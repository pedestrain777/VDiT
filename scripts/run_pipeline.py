"""
统一入口脚本（当前：video->keyframes->RAFT打分->EDEN动态插帧->输出）。

示例：
python scripts/run_pipeline.py \
  --video_path examples/input.mp4 \
  --eden_config configs/eval_eden.yaml \
  --output_path interpolation_outputs/out.mp4 \
  --raft_ckpt /data/models/raft/raft-things.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许直接 `python scripts/run_pipeline.py ...` 运行时也能 import `src.*`
# （此时 sys.path[0] 是 scripts/，不会自动包含项目根目录）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--eden_config", type=str, required=True)
    p.add_argument("--output_path", type=str, default="interpolation_outputs/interpolated.mp4")
    p.add_argument("--log_file", type=str, default="interpolation_outputs/greedy_refinement.log")

    # RAFT 权重默认路径（和常见 RAFT demo 的参数习惯对齐）
    # 若你机器上权重在别处，运行时用 --raft_ckpt 覆盖即可。
    p.add_argument(
        "--raft_ckpt",
        type=str,
        default="/data/models/raft/raft-things.pth",
        help="restore checkpoint (default: /data/models/raft/raft-things.pth)",
    )
    p.add_argument("--raft_device", type=str, default="cuda:0")
    p.add_argument("--eden_device", type=str, default="cuda:0")
    p.add_argument("--use_split_gpu", action="store_true")

    p.add_argument("--target_fps", type=float, default=24.0)
    p.add_argument("--keyframe_mode", type=str, choices=["all", "uniform", "random"], default="all")
    p.add_argument("--keyframes_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk_ratio", type=float, default=0.1)

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    # 延迟导入：避免 `--help` 也触发重依赖（如 lpips/xformers）导入失败
    from src.pipeline.run_iframe import PipelineConfig, run_interpolation_pipeline

    cfg = PipelineConfig(
        eden_config=args.eden_config,
        raft_ckpt=args.raft_ckpt,
        raft_device=args.raft_device,
        eden_device=args.eden_device,
        use_split_gpu=args.use_split_gpu,
        target_fps=args.target_fps,
        keyframe_mode=args.keyframe_mode,
        keyframes_k=args.keyframes_k,
        seed=args.seed,
        topk_ratio=args.topk_ratio,
    )

    run_interpolation_pipeline(
        video_path=args.video_path,
        output_path=args.output_path,
        cfg=cfg,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()


