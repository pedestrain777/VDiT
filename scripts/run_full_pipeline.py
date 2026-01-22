# scripts/run_full_pipeline.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许直接 `python scripts/run_full_pipeline.py ...` 时 import vdit.*
_HERE = Path(__file__).resolve()
for _parent in [_HERE] + list(_HERE.parents):
    _src_dir = _parent / "src"
    if _src_dir.is_dir():
        _root = _src_dir.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        break


def main() -> None:
    p = argparse.ArgumentParser()

    # -------- WAN 输入 --------
    p.add_argument("--wan_ckpt_dir", type=str, default=None, help="WAN checkpoint directory (required if not using --input_video)")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt for WAN (required if not using --input_video)")
    
    # 可选：直接从已有视频开始（跳过 WAN 生成，节省时间）
    p.add_argument("--input_video", type=str, default=None, help="Input video path (skip WAN generation if provided)")
    p.add_argument("--input_fps", type=float, default=None, help="Input video fps (auto-detect if not provided)")

    # WAN 生成参数（baseline：t2v-1.3B）
    p.add_argument("--wan_task", type=str, default="t2v-1.3B")
    p.add_argument("--wan_size", type=str, default="832*480", choices=["832*480", "480*832"])
    p.add_argument("--wan_frame_num", type=int, default=81)
    p.add_argument("--wan_sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--wan_sample_steps", type=int, default=50)
    p.add_argument("--wan_sample_shift", type=float, default=5.0)
    p.add_argument("--wan_guide_scale", type=float, default=5.0)
    p.add_argument("--wan_seed", type=int, default=0)
    p.add_argument("--wan_offload_model", action="store_true", help="enable offload_model=True (recommended)")

    # WAN 内部“均匀/随机取帧”（按 fps 下采样，保持时长不变）
    p.add_argument("--wan_out_fps", type=float, default=None, help="e.g. 8 or 12; None means no downsample")
    p.add_argument(
        "--wan_frame_sample",
        type=str,
        default="uniform",
        choices=["uniform", "random", "stratified_random"],
    )
    p.add_argument("--wan_frame_sample_seed", type=int, default=0)
    p.add_argument("--generator", type=str, default="wan", help="generator backend name (default: wan)")

    # -------- 插帧参数（你原来的 pipeline 参数）--------
    p.add_argument("--eden_config", type=str, required=True)
    p.add_argument("--output_path", type=str, default="interpolation_outputs/final.mp4")
    p.add_argument("--log_file", type=str, default="interpolation_outputs/greedy_refinement.log")
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

    # 关键帧策略：如果你用了 WAN out_fps 做“先取帧”，建议这里用 all（避免二次采样）
    p.add_argument("--keyframe_mode", type=str, choices=["all", "uniform", "random"], default="all")
    p.add_argument("--keyframes_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk_ratio", type=float, default=0.1)

    # 可选：保存 WAN 中间视频
    p.add_argument("--save_wan_video", type=str, default=None, help="save WAN raw/downsampled video for debugging")
    p.add_argument(
        "--save_sampled_video",
        type=str,
        default=None,
        help="save sampled video (after uniform/random sampling stage)",
    )
    p.add_argument(
        "--save_keyframes_video",
        type=str,
        default=None,
        help="save keyframes-only preview video (after keyframe selection in VDiT)",
    )

    args = p.parse_args()

    # 参数验证
    if args.input_video is None and (args.wan_ckpt_dir is None or args.prompt is None):
        p.error("Must provide either --input_video or (--wan_ckpt_dir + --prompt)")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    if args.save_wan_video:
        os.makedirs(os.path.dirname(args.save_wan_video) or ".", exist_ok=True)
    if args.save_sampled_video:
        os.makedirs(os.path.dirname(args.save_sampled_video) or ".", exist_ok=True)
    if args.save_keyframes_video:
        os.makedirs(os.path.dirname(args.save_keyframes_video) or ".", exist_ok=True)

    # 延迟导入（与 run_pipeline.py 一致）
    from vdit.generators.wan_t2v import WanGenerateConfig
    from vdit.pipeline.full_pipeline import FullPipelineConfig, run_full_pipeline
    from vdit.pipeline.run_iframe import PipelineConfig

    # WAN 配置（仅在需要 WAN 生成时使用）
    wan_cfg = WanGenerateConfig(
        task=args.wan_task,
        size=args.wan_size,
        frame_num=args.wan_frame_num,
        sample_solver=args.wan_sample_solver,
        sample_steps=args.wan_sample_steps,
        sample_shift=args.wan_sample_shift,
        guide_scale=args.wan_guide_scale,
        seed=args.wan_seed,
        offload_model=(True if args.wan_offload_model else True),  # 默认 True
        out_fps=args.wan_out_fps,
        frame_sample=args.wan_frame_sample,
        frame_sample_seed=args.wan_frame_sample_seed,
        device_id=0,
        t5_cpu=False,
    )

    iframe_cfg = PipelineConfig(
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

    full_cfg = FullPipelineConfig(wan=wan_cfg, iframe=iframe_cfg, generator_name=args.generator)

    run_full_pipeline(
        prompt=args.prompt,
        wan_ckpt_dir=args.wan_ckpt_dir,
        input_video=args.input_video,
        input_fps=args.input_fps,
        output_path=args.output_path,
        cfg=full_cfg,
        log_file=args.log_file,
        save_sampled_video_path=args.save_sampled_video,
        save_keyframes_video_path=args.save_keyframes_video,
        save_wan_video_path=args.save_wan_video,
    )


if __name__ == "__main__":
    main()
