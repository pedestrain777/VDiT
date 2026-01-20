from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from src.interpolators.eden_local import EdenInterpolator
from src.metrics.fuse import IntervalScorer, ScorerWeights
from src.flow.raft_estimator import RaftEstimator
from src.scheduler.greedy_refine import greedy_refine
from src.pipeline.keyframes import random_keyframes, uniform_keyframes
from src.pipeline.video_io import read_video_tensor, write_video_tensor


@dataclass(frozen=True)
class PipelineConfig:
    eden_config: str
    raft_ckpt: Optional[str] = None
    raft_device: str = "cuda:0"
    eden_device: str = "cuda:0"
    use_split_gpu: bool = False
    target_fps: float = 24.0
    keyframe_mode: Literal["all", "uniform", "random"] = "all"
    keyframes_k: int = 8
    seed: int = 0
    topk_ratio: float = 0.1
    weights: ScorerWeights = ScorerWeights()


def run_interpolation_pipeline(
    *,
    video_path: str,
    output_path: str,
    cfg: PipelineConfig,
    log_file: Optional[str] = None,
) -> None:
    """
    当前阶段落地：关键帧 -> (score_fn=5信息融合) greedy_refine -> EDEN 插帧 -> 输出视频

    未来接 WAN:
    wan->生成视频->(本函数)
    """
    frames, info = read_video_tensor(video_path)
    duration = info.num_frames / info.fps
    target_len = int(duration * cfg.target_fps)
    if target_len < 2:
        raise ValueError("target_len < 2，输入视频太短或 target_fps 太小。")

    if cfg.keyframe_mode == "all":
        init_frames = [frames[i].unsqueeze(0).cpu() for i in range(info.num_frames)]
    elif cfg.keyframe_mode == "uniform":
        init_frames = uniform_keyframes(frames, k=cfg.keyframes_k)
    elif cfg.keyframe_mode == "random":
        init_frames = random_keyframes(frames, k=cfg.keyframes_k, seed=cfg.seed)
    else:
        raise ValueError(f"unknown keyframe_mode: {cfg.keyframe_mode}")

    if len(init_frames) > target_len:
        raise ValueError(
            f"initial frames ({len(init_frames)}) > target_len ({target_len}). "
            f"请先降低 keyframes_k 或调高 target_fps。"
        )

    eden = EdenInterpolator(
        config_path=cfg.eden_config,
        device=cfg.eden_device,
        use_split_gpu=cfg.use_split_gpu,
    )

    raft = None
    if cfg.raft_ckpt:
        raft = RaftEstimator(model_path=cfg.raft_ckpt, device=cfg.raft_device)

    scorer = IntervalScorer(raft=raft, weights=cfg.weights, topk_ratio=cfg.topk_ratio)

    def score_fn(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(scorer(a, b))

    def interp_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return eden.interpolate(a, b)

    out_frames = greedy_refine(
        init_frames,
        target_len=target_len,
        score_fn=score_fn,
        interp_fn=interp_fn,
        verbose=True,
        log_file=log_file,
    )

    out = torch.cat(out_frames, dim=0)  # [T,3,H,W]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_video_tensor(output_path, out, fps=cfg.target_fps)


