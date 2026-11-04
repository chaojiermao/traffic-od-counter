# -*- coding: utf-8 -*-
r"""车型识别 - 命令行版（批处理用）。

网页界面（app.py）和这个脚本共用同一套计数内核 ``src/pipeline.py``，
所以两边的判定规则、分段口径、去重逻辑完全一致，不会再出现"两边结果不一样"。

用法示例::

    venv\Scripts\python.exe src\vehicle_classifier.py --video videos\xxx.mp4
    venv\Scripts\python.exe src\vehicle_classifier.py --video xxx.mp4 --seg 5 --speed 2
    venv\Scripts\python.exe src\vehicle_classifier.py --video xxx.mp4 ^
        --roi "0.05,0.30;0.95,0.30;0.95,0.90;0.05,0.90"

输出（默认 output\<视频名>\）：
    视频名_标注.mp4   画了框和车型的视频，可肉眼核对
    车辆明细.csv      每辆车过线的时刻、车型、框宽
    分类汇总.csv      7 类各多少辆
    分段统计.csv      按 --seg 分钟切段的数量表
"""

import argparse
import csv
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import pipeline as pl  # noqa: E402


def parse_roi(text):
    """'0.1,0.3;0.9,0.3;0.9,0.9' -> [(0.1,0.3), ...]"""
    if not text:
        return []
    pts = []
    for chunk in text.replace("；", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    return pts


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="监控视频车型识别计数")
    ap.add_argument("--video", required=True, help="监控视频路径")
    ap.add_argument("--weights", default=None, help="权重，默认 weights/yolov8x.pt")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 output/<视频名>")
    ap.add_argument("--line", type=float, default=0.65, help="计数线圈高度比例 0~1")
    ap.add_argument("--seg", type=float, default=5.0, help="每几分钟一段，默认 5")
    ap.add_argument("--speed", type=int, default=1, help="抽帧倍速，1=每帧都处理")
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度")
    ap.add_argument("--truck-split", type=float, default=150.0, help="大/小货车框宽分界 px")
    ap.add_argument("--taxi-ratio", type=float, default=0.15, help="出租车黄色占比阈值")
    ap.add_argument("--bus-white", type=float, default=0.5, help="白车身大巴阈值")
    ap.add_argument("--roi", default="", help="区域比例点，如 0.1,0.3;0.9,0.3;0.9,0.9")
    ap.add_argument("--no-video", action="store_true", help="不输出标注视频（更快）")
    args = ap.parse_args()

    from ultralytics import YOLO

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    weights = args.weights or osp.join(root, "weights", "yolov8x.pt")
    if not osp.isfile(weights):
        sys.exit(f"[错误] 找不到权重：{weights}")

    meta = pl.probe_video(args.video)
    if not meta["ok"]:
        sys.exit(f"[错误] 视频不可用：{meta['error']}")

    name = osp.splitext(osp.basename(args.video))[0]
    outdir = args.outdir or osp.join(root, "output", name)
    os.makedirs(outdir, exist_ok=True)

    print(f"[信息] 视频 {meta['width']}x{meta['height']} @{meta['fps']:.1f}fps "
          f"时长 {meta['duration'] / 60:.1f} 分钟")
    print(f"[信息] 加载模型 {weights}")
    model = YOLO(weights)
    cmap = pl.coco_indices(model.names)
    if not cmap:
        sys.exit("[错误] 权重里没有 car/bus/truck/motorcycle")
    print(f"[信息] 类别映射：{cmap}")

    cfg = {
        "line_y_ratio": args.line,
        "conf": args.conf,
        "truck_split": args.truck_split,
        "taxi_ratio": args.taxi_ratio,
        "bus_white": args.bus_white,
    }
    record = None if args.no_video else osp.join(outdir, f"{name}_标注.mp4")

    state = pl.RunState("cli")

    def on_progress(p, log):
        sys.stdout.write(f"\r[进度] {p * 100:5.1f}%  {log}    ")
        sys.stdout.flush()

    counter = pl.Counter(
        model=model,
        state=state,
        video_path=args.video,
        meta=meta,
        cfg=cfg,
        roi_pts=parse_roi(args.roi),
        start_sec=args.start,
        segment_min=args.seg,
        speed=args.speed,
        cmap=cmap,
        run_id="cli",
        record_path=record,
        on_progress=on_progress,
    )
    counter.run()
    print()

    snap = state.snapshot()
    if snap["error"]:
        sys.exit(f"[错误] {snap['error']}")

    res = snap["result"]
    if not res or not res["rows"]:
        sys.exit("[错误] 没有产生结果")

    # 逐车明细
    write_csv(
        osp.join(outdir, "车辆明细.csv"),
        ["跟踪ID", "车型", "过线时间(s)", "所在段", "框宽px"],
        [
            [v["id"], v["type"], round(v["time"], 2), v["seg"] + 1, round(v["width"], 1)]
            for v in res["vehicles"]
        ],
    )

    totals = {vt: 0 for vt in pl.VEHICLE_ORDER}
    for row in res["rows"]:
        for vt in pl.VEHICLE_ORDER:
            totals[vt] += row["counts"][vt]

    write_csv(
        osp.join(outdir, "分类汇总.csv"),
        ["车型", "数量"],
        [[vt, totals[vt]] for vt in pl.VEHICLE_ORDER] + [["总计", sum(totals.values())]],
    )

    def ts(sec):
        sec = int(round(sec))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    write_csv(
        osp.join(outdir, "分段统计.csv"),
        ["时间段"] + pl.VEHICLE_ORDER + ["额外计数", "新能源", "合计"],
        [
            [f"{ts(r['起始'])}-{ts(r['结束'])}"]
            + [r["counts"][vt] for vt in pl.VEHICLE_ORDER]
            + [r["额外计数"], r["新能源"],
               sum(r["counts"].values()) + r["额外计数"] + r["新能源"]]
            for r in res["rows"]
        ],
    )

    print("\n===== 分类汇总 =====")
    for vt in pl.VEHICLE_ORDER:
        print(f"  {vt}: {totals[vt]}")
    print(f"  总计: {sum(totals.values())}")
    print(f"\n[完成] 结果目录: {outdir}")
    if record:
        print(f"[完成] 标注视频: {record}")


if __name__ == "__main__":
    main()
