# src/pipeline/full_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vdit.generators.base import create_generator
from vdit.generators.wan_t2v import WanGenerateConfig
from vdit.pipeline.run_iframe import PipelineConfig, run_interpolation_pipeline_from_frames
from vdit.pipeline.video_io import read_video_tensor, write_video_tensor


@dataclass(frozen=True)
class FullPipelineConfig:
    # WAN 生成配置
    wan: WanGenerateConfig
    # 插帧配置（你原来的 PipelineConfig）
    iframe: PipelineConfig
    # 生成器名称（插件化）
    generator_name: str = "wan"


@torch.no_grad()
def run_full_pipeline(
    *,
    prompt: Optional[str] = None,
    wan_ckpt_dir: Optional[str] = None,
    input_video: Optional[str] = None,
    input_fps: Optional[float] = None,
    output_path: str,
    cfg: FullPipelineConfig,
    log_file: Optional[str] = None,
    save_sampled_video_path: Optional[str] = None,
    save_keyframes_video_path: Optional[str] = None,
    save_wan_video_path: Optional[str] = None,
) -> None:
    """
    完整 baseline：
      WAN(1.3B) -> 生成视频 -> (可选: 按 out_fps 均匀/随机取帧) -> 信息计算/自适应插帧 -> EDEN -> 输出
      
    或者从已有视频开始：
      已有视频 -> 信息计算/自适应插帧 -> EDEN -> 输出

    说明：
      - 如果提供 input_video，则跳过 WAN 生成，直接从该视频开始插帧（节省时间）
      - 如果使用 WAN 生成，"均匀/随机取帧"有两种方式：
        A) WAN 内部按 out_fps resample（cfg.wan.out_fps 不为 None 时启用）
        B) 插帧阶段 keyframe_mode=uniform/random （cfg.iframe.keyframe_mode）
      - 如果你想严格对应“先取帧再插帧”，建议：
          cfg.wan.out_fps=8/12 + cfg.wan.frame_sample=uniform/random
          并设置 cfg.iframe.keyframe_mode="all"
        因为这表示：先把 WAN 输出直接变成“关键帧序列”，插帧模块不再二次采样。
    """
    
    # 判断是从已有视频开始还是从 WAN 生成开始
    if input_video:
        # 从已有视频开始（跳过 WAN 生成）
        frames, info = read_video_tensor(input_video)
        fps_src = float(input_fps) if input_fps is not None else info.fps
    else:
        # 从 WAN 生成开始（原始流程）
        if prompt is None or wan_ckpt_dir is None:
            raise ValueError("Must provide either (input_video) or (prompt + wan_ckpt_dir)")
        generator = create_generator(cfg.generator_name, ckpt_dir=wan_ckpt_dir, cfg=cfg.wan)
        frames, fps_src = generator.generate(prompt)

    sampled_video_path = save_sampled_video_path or save_wan_video_path
    if sampled_video_path:
        write_video_tensor(sampled_video_path, frames, fps=float(fps_src))

    # 进入插帧 pipeline（不落盘中间 mp4，直接 tensor 传递）
    run_interpolation_pipeline_from_frames(
        frames=frames,
        fps_src=fps_src,
        output_path=output_path,
        cfg=cfg.iframe,
        log_file=log_file,
        save_keyframes_video_path=save_keyframes_video_path,
    )
