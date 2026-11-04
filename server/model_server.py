# -*- coding: utf-8 -*-
"""给网页版计数工具用的本机模型服务。

为什么要有它：网页里跑不了 PyTorch/CUDA，而 `weights\\yolov8x.pt` 有 130MB。
这里直接用本机 venv 里现成的 ultralytics + 显存，把 `src\\pipeline.py` 的计数引擎
包成一个 HTTP 服务，网页通过 http://127.0.0.1:8502 调用，所以模型判定规则
（COCO 映射 / 过线判定 / 颜色阈值 / 分车型）和你原来的系统完全一致。

启动：双击 `启动模型服务.bat`，或：
        venv\\Scripts\\python.exe model_server.py [端口]

接口
----
GET  /health                     服务与模型状态
POST /jobs?name=&line=&speed=&roi=   body=视频二进制，返回 {id}
GET  /jobs/<id>                   {status, progress, log, counts, total, ...}
POST /jobs/<id>/stop              请求停止
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipeline import Counter, RunState, VEHICLE_ORDER, coco_indices, probe_video  # noqa: E402

import cv2  # noqa: E402  pipeline 已经依赖它，这里用来把最新画面压成 JPEG

MODEL_PATH = os.environ.get("VEHICLE_WEIGHTS") or os.path.join(ROOT, "weights", "yolov8x.pt")
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, "weights", MODEL_PATH)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8502
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "vehicle_model_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 模型（后台线程加载，别挡住 /health）
# --------------------------------------------------------------------------- #
_model = None
_model_lock = threading.Lock()
_model_info = {"ready": False, "loading": False, "error": "", "device": "", "classes": []}


def load_model():
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        _model_info["loading"] = True
        try:
            import torch
            from ultralytics import YOLO

            if not os.path.isfile(MODEL_PATH):
                raise RuntimeError("找不到权重：" + MODEL_PATH)
            _model = YOLO(MODEL_PATH)
            _model_info["ready"] = True
            _model_info["device"] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            )
            _model_info["classes"] = [str(v) for v in (_model.names or {}).values()]
            print("[model] 就绪，设备：%s" % _model_info["device"], flush=True)
        except Exception as exc:  # 加载失败要能在 /health 里看到
            _model_info["error"] = "%s: %s" % (type(exc).__name__, exc)
            print("[model] 加载失败：%s" % _model_info["error"], flush=True)
        finally:
            _model_info["loading"] = False
        return _model


# --------------------------------------------------------------------------- #
# 任务
# --------------------------------------------------------------------------- #
JOBS = {}
JOBS_LOCK = threading.Lock()


class Job(object):
    def __init__(self, path, line_ratio, roi, speed, seg_min, name):
        self.id = uuid.uuid4().hex[:12]
        self.path = path
        self.name = name
        self.state = RunState(self.id)
        self.created = time.time()
        self.finished = None
        self.error = ""
        self.result = None
        self.thread = None
        # ---- 实时监看用 ----
        self.lock = threading.Lock()
        self.duration = 0.0        # 视频总长（秒）
        self.start = 0.0           # 从第几秒开始算
        self.clip = 0.0            # 只算多少秒（0 = 全片）
        self.segcounts = {}        # 段号 -> 该段各类计数
        self.prev_seg = -1
        self.prev_counts = {}      # 同一段内上一次的计数，用来推算“新过线”
        self.events = []           # 实时过线事件 {t, cat, n}


def _watch_clip(job, start, clip, duration):
    """只算一段：跑到 start+clip 秒就让计数器自己停下来。"""
    if duration <= 0 or clip <= 0:
        return
    end_t = start + clip
    while True:
        snap = job.state.snapshot()
        if snap["done"]:
            return
        if float(snap["progress"]) * duration >= end_t - 0.3:
            job.state.request_stop()
            return
        time.sleep(0.25)


def _run_job(job, line_ratio, roi, speed, seg_min, start=0.0, clip=0.0, imgsz=640, half=True):
    try:
        model = load_model()
        if model is None:
            job.state.finish(None, error="模型未就绪：" + (_model_info["error"] or "加载中"))
            return
        meta = probe_video(job.path)
        if not meta.get("ok"):
            job.state.finish(None, error=meta.get("error") or "视频读取失败")
            return
        job.duration = float(meta.get("duration") or 0)
        if clip and clip > 0:
            threading.Thread(
                target=_watch_clip, args=(job, start, clip, job.duration), daemon=True
            ).start()
        counter = Counter(
            model,
            job.state,
            job.path,
            meta,
            cfg={"line_y_ratio": float(line_ratio), "imgsz": int(imgsz), "half": bool(half)},
            roi_pts=roi,
            speed=int(speed),
            segment_min=float(seg_min),
            cmap=coco_indices(model.names),
            run_id=job.id,
            start_sec=float(start or 0.0),
        )
        counter.run()
    except Exception as exc:
        job.state.finish(None, error="%s: %s" % (type(exc).__name__, exc))
    finally:
        job.finished = time.time()
        print("[job %s] 结束" % job.id, flush=True)
        # 网页上传的副本跑完就删，别再把 C 盘塞满（本机路径 src= 的绝不动）
        try:
            if os.path.dirname(os.path.abspath(job.path)) == os.path.abspath(UPLOAD_DIR):
                os.remove(job.path)
                print("[job %s] 已清理上传临时文件" % job.id, flush=True)
        except OSError:
            pass


def job_payload(job):
    snap = job.state.snapshot()
    out = {
        "id": job.id,
        "name": job.name,
        "running": bool(snap["running"]) and not snap["done"],
        "done": bool(snap["done"]),
        "progress": round(float(snap["progress"]), 4),
        "log": snap["log"] or "",
        "elapsed": round(float(snap["elapsed"]), 1),
        "error": snap["error"] or "",
    }
    res = snap.get("result")
    if res:
        totals = {}
        for vt in VEHICLE_ORDER:
            totals[vt] = int(sum(int(r["counts"].get(vt, 0)) for r in res["rows"]))
        out["counts"] = totals
        out["total"] = int(sum(totals.values()))
        vehicles = res.get("vehicles") or []
        out["vehicles"] = len(vehicles)
        out["segments"] = len(res["rows"])
        out["finished"] = res.get("elapsed")
        # 每辆车的过线时刻 + 车型：网页按自己的分段重新归类，所以这里必须给明细
        out["events"] = [
            {"t": round(float(v["time"]), 2), "y": v["type"]}
            for v in vehicles[:20000]
        ]
    return out


def live_payload(job):
    """监看页用：当前段计数 + 各段累计 + 最近过线事件 + 帧版本号。"""
    snap = job.state.snapshot()
    seg = int(snap["seg_idx"])
    raw = snap["counts"] or {}
    counts = {vt: int(raw.get(vt, 0)) for vt in VEHICLE_ORDER}
    prog = float(snap["progress"])
    t = prog * job.duration if job.duration else 0.0

    with job.lock:
        if seg != job.prev_seg:
            job.prev_seg = seg
            job.prev_counts = {}
        job.segcounts[seg] = counts
        for vt in VEHICLE_ORDER:
            delta = counts[vt] - int(job.prev_counts.get(vt, 0))
            if delta > 0:
                job.events.append({"t": round(t, 2), "cat": vt, "n": delta})
                if len(job.events) > 1000:
                    del job.events[:300]
        job.prev_counts = dict(counts)
        segcounts = {str(k): {vt: int(v.get(vt, 0)) for vt in VEHICLE_ORDER}
                     for k, v in job.segcounts.items()}
        events = job.events[-40:]

    totals = {vt: int(sum(s.get(vt, 0) for s in segcounts.values())) for vt in VEHICLE_ORDER}
    return {
        "id": job.id,
        "name": job.name,
        "seg": seg + 1,
        "counts": counts,
        "totals": totals,
        "total": int(sum(totals.values())),
        "segcounts": segcounts,
        "events": events,
        "progress": round(prog, 4),
        "t": round(t, 2),
        "duration": round(job.duration, 2),
        "start": job.start,
        "clip": job.clip,
        "log": snap["log"] or "",
        "ver": int(snap["frame_ver"]),
        "running": bool(snap["running"]) and not snap["done"],
        "done": bool(snap["done"]),
        "error": snap["error"] or "",
        "elapsed": round(float(snap["elapsed"]), 1),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "VehicleModel/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---- 工具 ----
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    # ---- 路由 ----
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("/", "/health"):
            return self._json(
                {
                    "ok": True,
                    "service": "vehicle-model-server",
                    "model_ready": _model_info["ready"],
                    "model_loading": _model_info["loading"],
                    "model_error": _model_info["error"],
                    "device": _model_info["device"],
                    "weights": os.path.basename(MODEL_PATH),
                    "classes": _model_info["classes"],
                    "jobs": len(JOBS),
                }
            )
        if path == "/videos":
            return self._list_videos(query)
        if path == "/info":
            return self._video_info(query)
        if path == "/still":
            return self._still(query)
        if path.startswith("/jobs/"):
            parts = path.strip("/").split("/")
            job_id = parts[1] if len(parts) > 1 else ""
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self._json({"error": "任务不存在"}, 404)
            if len(parts) >= 3 and parts[2] == "live":
                return self._json(live_payload(job))
            if len(parts) >= 3 and parts[2] == "frame":
                return self._frame(job, query)
            return self._json(job_payload(job))
        return self._json({"error": "未知接口"}, 404)

    # ---- 本机视频目录（省掉几个 G 的上传） ----
    def _list_videos(self, query):
        folder = (query.get("dir") or [""])[0].strip() or r"D:\ra3"
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            return self._json({"ok": False, "error": "目录不存在：%s" % folder}, 400)
        exts = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm")
        items = []
        try:
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(exts):
                    continue
                full = os.path.join(folder, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                items.append({"name": name, "path": full,
                              "size": int(st.st_size), "mtime": int(st.st_mtime)})
        except OSError as exc:
            return self._json({"ok": False, "error": str(exc)}, 500)
        return self._json({"ok": True, "dir": folder, "items": items})

    def _video_info(self, query):
        src = (query.get("src") or [""])[0].strip()
        if not src or not os.path.isfile(src):
            return self._json({"ok": False, "error": "文件不存在"}, 400)
        meta = probe_video(src)
        if not meta.get("ok"):
            return self._json({"ok": False, "error": meta.get("error") or "读不出来"}, 400)
        return self._json({
            "ok": True,
            "name": os.path.basename(src),
            "duration": round(float(meta.get("duration") or 0), 2),
            "fps": round(float(meta.get("fps") or 0), 2),
            "width": int(meta.get("width") or 0),
            "height": int(meta.get("height") or 0),
            "frames": int(meta.get("total_frames") or 0),
        })

    # ---- 静帧（调线/调区域时当底图，不用上传视频） ----
    def _still(self, query):
        src = (query.get("src") or [""])[0].strip()
        if not src or not os.path.isfile(src):
            return self._json({"error": "文件不存在"}, 400)
        try:
            t = max(0.0, float((query.get("t") or ["0"])[0]))
        except ValueError:
            t = 0.0
        try:
            want = int((query.get("w") or ["1280"])[0])
        except ValueError:
            want = 1280
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return self._json({"error": "打不开视频"}, 400)
        try:
            if t > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return self._json({"error": "这一帧读不出来（是否超出片长？）"}, 400)
            h, w = frame.shape[:2]
            if want and w > want:
                frame = cv2.resize(frame, (want, max(1, int(h * want / float(w)))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                raise RuntimeError("编码失败")
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)
        finally:
            cap.release()
        data = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    # ---- 最新一帧（已画好框 / 线 / 区域） ----
    def _frame(self, job, query):
        snap = job.state.snapshot()
        frame = snap.get("frame")
        if frame is None:
            return self._json({"error": "还没有画面"}, 404)
        try:
            want = int((query.get("w") or ["0"])[0])
        except ValueError:
            want = 0
        try:
            h, w = frame.shape[:2]
            if want and w > want:
                frame = cv2.resize(frame, (want, max(1, int(h * want / float(w)))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if not ok:
                raise RuntimeError("编码失败")
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)
        data = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.startswith("/jobs/") and path.endswith("/stop"):
            job_id = path.split("/")[2]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self._json({"error": "任务不存在"}, 404)
            job.state.request_stop()
            return self._json({"ok": True})

        if path == "/jobs":
            return self._create_job(query)

        return self._json({"error": "未知接口"}, 404)

    def _create_job(self, query):
        def q(name, default=""):
            return (query.get(name) or [default])[0]

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0

        # 两种来源：本机路径（src=，不复制、不传输）或直接上传视频二进制
        src = (q("src") or "").strip()
        if src:
            tmp = os.path.abspath(src)
            if not os.path.isfile(tmp):
                return self._json({"error": "找不到文件：%s" % tmp}, 400)
            name = os.path.basename(tmp)
        else:
            if length <= 0:
                return self._json({"error": "没有收到视频数据"}, 400)
            name = os.path.basename(q("name") or "upload.mp4")
            ext = os.path.splitext(name)[1] or ".mp4"
            tmp = os.path.join(UPLOAD_DIR, uuid.uuid4().hex[:12] + ext)

            # 流式落盘，别把几百 MB 读进内存
            remaining = length
            with open(tmp, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            if remaining > 0:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return self._json({"error": "视频上传中断"}, 400)

        try:
            line_ratio = float(q("line", "0.65"))
        except ValueError:
            line_ratio = 0.65
        try:
            speed = max(1, min(10, int(q("speed", "2"))))
        except ValueError:
            speed = 2
        try:
            seg_min = max(1.0, float(q("seg", "5")))
        except ValueError:
            seg_min = 5.0

        roi = None
        raw_roi = q("roi", "")
        if raw_roi:
            pts = []
            ok = True
            for part in raw_roi.split(";"):
                if not part.strip():
                    continue
                try:
                    x, y = part.split(",")
                    pts.append((float(x), float(y)))
                except ValueError:
                    ok = False
                    break
            if ok and len(pts) >= 3:
                roi = pts

        try:
            start = max(0.0, float(q("start", "0") or 0))
        except ValueError:
            start = 0.0
        try:
            clip = max(0.0, float(q("clip", "0") or 0))
        except ValueError:
            clip = 0.0
        try:
            imgsz = max(320, min(1920, int(q("imgsz", "640") or 640)))
        except ValueError:
            imgsz = 640
        half = str(q("half", "1")).lower() not in ("0", "false", "no")

        job = Job(tmp, line_ratio, roi, speed, seg_min, name)
        job.start = start
        job.clip = clip
        with JOBS_LOCK:
            JOBS[job.id] = job
        job.thread = threading.Thread(
            target=_run_job, args=(job, line_ratio, roi, speed, seg_min, start, clip, imgsz, half), daemon=True
        )
        job.thread.start()
        print("[job %s] 开始 %s line=%.2f speed=%s imgsz=%s half=%s roi=%s"
              % (job.id, name, line_ratio, speed, imgsz, half, bool(roi)), flush=True)
        return self._json({"id": job.id, "name": name})


def main():
    print("=" * 60)
    print("车型识别模型服务  http://127.0.0.1:%d" % PORT)
    print("权重：%s" % MODEL_PATH)
    print("按 Ctrl+C 停止")
    print("=" * 60)
    threading.Thread(target=load_model, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
