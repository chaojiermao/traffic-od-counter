# -*- coding: utf-8 -*-
"""车型识别计数引擎。

这个模块**不导入 Streamlit**，所有视频处理/计数逻辑都在这里，可以：
  * 被 app.py（网页界面）作为后台线程调用；
  * 被命令行脚本直接调用；
  * 单独用脚本做回归测试。

核心约定
--------
* 工作线程绝不访问 ``st.session_state``（跨线程访问会直接抛 KeyError，
  这是旧版本最致命的问题）。线程和界面之间只通过 :class:`RunState` 通信，
  内部带锁，界面侧统一用 ``snapshot()`` 读快照。
* 分段统计用 ``seg_idx -> {车型: 数量}`` 的字典累积，最后一次性成表。
  旧版本靠"段切换时 append"，一旦中间有整段没车/跳段就会漏段、错段。
* 手动计数（额外/新能源）按**当前所在段**记账，而不是"段切换时快照上一次的
  累计值"，避免跨段串号。
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

VEHICLE_ORDER = [
    "其它小型客车",
    "其它大型客车",
    "公交车",
    "出租车",
    "小型货车",
    "大型货车",
    "摩托车",
]

COLOR_BOX = {
    "其它小型客车": (200, 200, 0),
    "其它大型客车": (0, 165, 255),
    "公交车": (0, 255, 0),
    "出租车": (0, 215, 255),
    "小型货车": (255, 0, 255),
    "大型货车": (0, 0, 255),
    "摩托车": (255, 255, 0),
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".ts", ".m4v", ".wmv")

DEFAULT_CFG = {
    "conf": 0.25,
    "line_y_ratio": 0.65,
    "truck_split": 150,
    "taxi_ratio": 0.15,
    "taxi_h_lo": 15,
    "taxi_h_hi": 40,
    "taxi_s_lo": 80,
    "taxi_v_lo": 80,
    "bus_white": 0.5,
    # 非过线帧每隔几帧投一次票（颜色判断没必要每帧做，省算力）
    "vote_stride": 3,
    # 推理尺寸与半精度：imgsz 越大小车越不容易漏，half=True 在 20/30/40 系显卡上基本白送 1.5 倍
    "imgsz": 640,
    "half": True,
}

_COCO_TARGETS = ("car", "motorcycle", "bus", "truck")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def find_videos(root, limit=50):
    """在 <root>/videos 里找视频；没有就退回 <root> 本身。"""
    for folder in (os.path.join(root, "videos"), root):
        if not os.path.isdir(folder):
            continue
        found = []
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(VIDEO_EXTS):
                found.append(path)
        if found:
            return found[:limit]
    return []


def coco_indices(names):
    """从 model.names 里找出 car / motorcycle / bus / truck 的类别 id。"""
    out = {}
    for key, value in (names or {}).items():
        name = str(value).strip().lower()
        if name in _COCO_TARGETS and name not in out:
            out[name] = int(key)
    return out


def probe_video(path):
    """读取视频基础信息。失败时返回 ok=False + error 文案，不抛异常。"""
    info = {
        "ok": False,
        "error": "",
        "path": path,
        "fps": 25.0,
        "width": 0,
        "height": 0,
        "total_frames": 0,
        "duration": 0.0,
        "mtime": 0.0,
    }
    if not path:
        info["error"] = "未填写视频路径"
        return info
    if not os.path.isfile(path):
        info["error"] = "文件不存在"
        return info

    info["mtime"] = os.path.getmtime(path)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            info["error"] = "无法打开（编码不支持或文件损坏）"
            return info

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # 某些容器（尤其是监控录制的 ts/avi）会返回 1000 或 0 这类假值
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            info["error"] = "读不出画面（文件可能不完整）"
            return info
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]

        total = max(total, 0)
        info.update(
            ok=True,
            fps=fps,
            width=width,
            height=height,
            total_frames=total,
            duration=(total / fps) if total > 0 else 0.0,
        )
        return info
    finally:
        cap.release()


def grab_frame(path, sec=0.0):
    """抓取指定秒数处的一帧（BGR）。失败返回 None。"""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        if sec and sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        return frame if ok and frame is not None else None
    finally:
        cap.release()


def color_ratio(hsv, h_lo, h_hi, s_lo, v_lo):
    if hsv is None or getattr(hsv, "size", 0) == 0:
        return 0.0
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    return float(cv2.countNonZero(mask)) / float(mask.size)


def classify(coco_id, box_w, roi_bgr, cmap, cfg):
    """COCO 粗类 + 框宽 + 车身颜色 -> 7 类之一；非目标类返回 None。"""
    if coco_id == cmap.get("motorcycle"):
        return "摩托车"

    if coco_id == cmap.get("bus"):
        white = 0.0
        if roi_bgr is not None and roi_bgr.size:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
            white = float(cv2.countNonZero(mask)) / float(mask.size)
        return "其它大型客车" if white >= cfg["bus_white"] else "公交车"

    if coco_id == cmap.get("truck"):
        return "大型货车" if box_w >= cfg["truck_split"] else "小型货车"

    if coco_id == cmap.get("car"):
        yellow = 0.0
        if roi_bgr is not None and roi_bgr.size:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            yellow = color_ratio(
                hsv, cfg["taxi_h_lo"], cfg["taxi_h_hi"], cfg["taxi_s_lo"], cfg["taxi_v_lo"]
            )
        return "出租车" if yellow >= cfg["taxi_ratio"] else "其它小型客车"

    return None


# --------------------------------------------------------------------------- #
# 线程安全共享状态
# --------------------------------------------------------------------------- #
class RunState:
    """工作线程与界面之间的唯一通道。所有读写都带锁。"""

    def __init__(self, run_id="run"):
        self.run_id = run_id
        self._lock = threading.RLock()

        self._frame = None          # 最近一帧（RGB）
        self._frame_ver = 0
        self._counts = {}           # 当前段各车型计数
        self._seg_idx = 0
        self._progress = 0.0
        self._log = ""

        self._manual = {}           # seg -> [额外, 新能源]
        self._stop = False
        self.running = False
        self.done = False
        self._error = None
        self._result = None
        self._started_at = time.time()
        self._finished_at = None

    # ---------------- 写 ----------------
    def set_frame(self, rgb):
        with self._lock:
            self._frame = rgb
            self._frame_ver += 1

    def publish(self, seg_idx, counts, progress, log):
        with self._lock:
            self._seg_idx = int(seg_idx)
            self._counts = dict(counts)
            self._progress = float(progress)
            self._log = log

    def inc_manual(self, kind, n=1):
        """kind: 'extra' | 'ev'，记到当前所在段。"""
        with self._lock:
            bucket = self._manual.setdefault(self._seg_idx, [0, 0])
            bucket[0 if kind == "extra" else 1] += n

    def clear_manual(self):
        with self._lock:
            self._manual = {}

    def request_stop(self):
        with self._lock:
            self._stop = True

    def should_stop(self):
        with self._lock:
            return self._stop

    def finish(self, result, error=None):
        with self._lock:
            self._result = result
            if error:
                self._error = str(error)
            self.running = False
            self.done = True
            self._finished_at = time.time()
            if error:
                self._log = f"处理中断：{error}"

    # ---------------- 读 ----------------
    def manual_of(self, seg):
        with self._lock:
            return tuple(self._manual.get(seg, (0, 0)))

    def manual_segments(self):
        with self._lock:
            return list(self._manual.keys())

    def snapshot(self):
        with self._lock:
            return {
                "run_id": self.run_id,
                "frame": self._frame,
                "frame_ver": self._frame_ver,
                "counts": dict(self._counts),
                "seg_idx": self._seg_idx,
                "progress": self._progress,
                "log": self._log,
                "running": self.running,
                "done": self.done,
                "error": self._error,
                "result": self._result,
                "elapsed": (self._finished_at or time.time()) - self._started_at,
            }


# --------------------------------------------------------------------------- #
# 计数主循环
# --------------------------------------------------------------------------- #
class Counter:
    def __init__(
        self,
        model,
        state,
        video_path,
        meta,
        cfg=None,
        roi_pts=None,
        start_sec=0.0,
        segment_min=5.0,
        speed=1,
        cmap=None,
        run_id="run",
        record_path=None,
        on_progress=None,
    ):
        self.model = model
        self.state = state
        self.video_path = video_path
        self.meta = meta
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.cmap = dict(cmap or {})
        self.roi_pts = list(roi_pts or [])
        self.start_sec = float(max(0.0, start_sec))
        self.segment_min = float(max(1, segment_min))
        self.speed = int(max(1, speed))
        self.run_id = run_id
        self.record_path = record_path
        self.on_progress = on_progress

    # ---- 外部入口（在线程里调用） ----
    def run(self):
        self.state.running = True
        try:
            self._loop()
        except Exception as exc:  # 任何异常都要落到界面上，不能让线程静默死掉
            self.state.finish(None, error=f"{type(exc).__name__}: {exc}")

    # ---- 内部 ----
    def _reset_tracker(self):
        """上一轮跑完后 ByteTrack 还留着旧 ID，不重置会串号/漏计。"""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        try:
            if trackers:
                for tracker in trackers:
                    tracker.reset()
            else:
                self.model.predictor = None
        except Exception:
            self.model.predictor = None

    def _loop(self):
        state = self.state
        meta = self.meta
        fps = float(meta["fps"])
        width = int(meta["width"])
        height = int(meta["height"])
        cfg = self.cfg

        line_y = int(round(height * cfg["line_y_ratio"]))
        line_y = min(max(line_y, 1), height - 1)

        poly = None
        if len(self.roi_pts) >= 3:
            poly = np.array(
                [[int(round(x * width)), int(round(y * height))] for x, y in self.roi_pts],
                dtype=np.int32,
            )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("视频打不开")

        self._reset_tracker()
        if self.start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)

        writer = None
        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.record_path, fourcc, max(fps / self.speed, 1.0), (width, height)
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"无法创建标注视频：{self.record_path}")

        seg_sec = self.segment_min * 60.0
        tracks = {}
        counted = set()
        vehicles = []
        seg_counts = defaultdict(lambda: defaultdict(int))
        valid = set(self.cmap.values())
        frame_no = 0          # 从起点开始读过的帧数
        processed = 0
        t0 = time.time()
        last_pub = 0.0
        cur_seg = 0
        stride = max(1, int(cfg.get("vote_stride", 3)))

        try:
            while True:
                if state.should_stop():
                    break
                frame_no += 1

                # 抽帧加速：只处理第 1、1+speed、1+2*speed… 帧。
                # 跳过的帧只 grab（不转 BGR、不拷贝），比 read 便宜约 3 倍；
                # 要处理的那一帧再 retrieve 拿图。
                if self.speed > 1 and (frame_no - 1) % self.speed != 0:
                    if not cap.grab():
                        frame_no -= 1
                        break
                    continue
                if not cap.grab():
                    frame_no -= 1
                    break
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame_no -= 1
                    break

                t_abs = self.start_sec + frame_no / fps
                cur_seg = int(max(0.0, t_abs - self.start_sec) // seg_sec)
                proc_counts = seg_counts[cur_seg]

                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=cfg["conf"],
                    imgsz=int(cfg.get("imgsz") or 640),
                    half=bool(cfg.get("half", True)),
                    verbose=False,
                )
                boxes = results[0].boxes

                if poly is not None:
                    cv2.polylines(frame, [poly], True, (0, 255, 255), 2)
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                    clss = boxes.cls.cpu().numpy().astype(int)
                    xywh = boxes.xywh.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)

                    for tid, cls, (cx, cy, bw, _bh), (x1, y1, x2, y2) in zip(
                        ids, clss, xywh, xyxy
                    ):
                        if cls not in valid:
                            continue

                        track = tracks.get(tid)
                        if track is None:
                            # 新车必须在 ROI 内出现才开始跟踪（未画 ROI 则全画面）
                            if poly is not None and cv2.pointPolygonTest(
                                poly, (float(cx), float(cy)), False
                            ) < 0:
                                continue
                            track = {
                                "prev_cy": float(cy),
                                "votes": defaultdict(int),
                                "crossed": False,
                                "last": frame_no,
                            }
                            tracks[tid] = track

                        track["last"] = frame_no
                        cy_f = float(cy)

                        crossed_now = (track["prev_cy"] < line_y <= cy_f) or (
                            track["prev_cy"] > line_y >= cy_f
                        )

                        need_vote = crossed_now or (not track["votes"]) or (
                            frame_no % stride == 0
                        )
                        if need_vote:
                            pad = 2
                            xi1, yi1 = max(0, x1 + pad), max(0, y1 + pad)
                            xi2, yi2 = min(width, x2 - pad), min(height, y2 - pad)
                            roi = frame[yi1:yi2, xi1:xi2]
                            vtype = classify(cls, float(bw), roi, self.cmap, cfg)
                            if vtype:
                                track["votes"][vtype] += 3 if crossed_now else 1

                        if crossed_now:
                            track["crossed"] = True

                        final = None
                        if track["votes"]:
                            final = max(track["votes"], key=track["votes"].get)

                        if track["crossed"] and tid not in counted and final:
                            counted.add(tid)
                            proc_counts[final] += 1
                            vehicles.append(
                                {
                                    "id": int(tid),
                                    "type": final,
                                    "time": float(t_abs),
                                    "seg": int(cur_seg),
                                    "width": float(bw),
                                }
                            )

                        track["prev_cy"] = cy_f

                        if final:
                            color = COLOR_BOX.get(final, (255, 255, 255))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                f"#{tid} {final}",
                                (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                processed += 1
                if processed % 200 == 0:
                    # 清掉长时间没再出现的轨迹，避免内存无限增长
                    stale = [k for k, v in tracks.items() if frame_no - v["last"] > 90]
                    for k in stale:
                        tracks.pop(k, None)

                extra, ev = state.manual_of(cur_seg)
                cv2.putText(
                    frame,
                    f"Extra:{extra} EV:{ev}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                state.set_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if writer is not None:
                    writer.write(frame)

                now = time.time()
                if now - last_pub > 0.1:
                    last_pub = now
                    if meta["total_frames"] > 0:
                        progress = min(t_abs / max(meta["duration"], 1e-6), 1.0)
                    else:
                        progress = 0.0
                    speed_fps = processed / max(now - t0, 0.1)
                    mm = int(t_abs // 60)
                    ss = int(t_abs % 60)
                    log = (
                        f"{processed} 帧已处理  {speed_fps:.1f} 帧/s  "
                        f"视频位置 {mm:02d}:{ss:02d}  第 {cur_seg + 1} 段"
                    )
                    state.publish(cur_seg, proc_counts, progress, log)
                    if self.on_progress is not None:
                        try:
                            self.on_progress(progress, log)
                        except Exception:
                            pass
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if frame_no == 0 and not state.should_stop():
            raise RuntimeError("没有读到任何画面：起点可能超过了视频长度")

        # ---- 汇总成表：补齐所有出现过的段（含只有手动计数的段） ----
        segments = set(seg_counts.keys()) | set(state.manual_segments()) | {cur_seg}
        last_seg = max(segments)
        rows = []
        for idx in range(last_seg + 1):
            extra, ev = state.manual_of(idx)
            counts = seg_counts.get(idx, {})
            row = {
                "段": idx + 1,
                "起始": self.start_sec + idx * seg_sec,
                "结束": self.start_sec + (idx + 1) * seg_sec,
                "counts": {vt: int(counts.get(vt, 0)) for vt in VEHICLE_ORDER},
                "额外计数": int(extra),
                "新能源": int(ev),
            }
            rows.append(row)

        result = {
            "rows": rows,
            "start_sec": self.start_sec,
            "segment_min": self.segment_min,
            "stopped": state.should_stop(),
            "frames_read": frame_no,
            "frames_processed": processed,
            "elapsed": time.time() - t0,
            "roi": self.roi_pts,
            "vehicles": vehicles,
        }
        state.finish(result)
