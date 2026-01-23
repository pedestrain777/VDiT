# src/vdit/generators/wan_t2v.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import sys
from pathlib import Path

import torch

from vdit.generators.base import register_generator

try:
    import wan  # type: ignore
except Exception:
    _ROOT = Path(__file__).resolve().parents[3]
    _WAN_ROOT = _ROOT / "third_party" / "wan"
    if _WAN_ROOT.exists():
        sys.path.insert(0, str(_WAN_ROOT))
    import wan  # type: ignore

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # type: ignore
from wan.utils.frame_sampling import resample_video_tensor  # type: ignore

FrameSampleMode = Literal["uniform", "random", "stratified_random"]


@dataclass(frozen=True)
class WanGenerateConfig:
    # -------- WAN 基本参数 --------
    task: str = "t2v-1.3B"  # baseline：wan1.3b
    size: str = "832*480"  # 1.3B 支持：480*832 / 832*480
    frame_num: int = 81  # WAN 默认视频帧数（通常 4n+1）

    # -------- 采样参数（对齐 wan-main/generate.py）--------
    sample_solver: str = "unipc"  # ["unipc", "dpm++"]
    sample_steps: int = 50  # t2v 默认 50
    sample_shift: float = 5.0
    guide_scale: float = 5.0
    seed: int = 0
    offload_model: bool = True  # 单卡默认 True，省显存

    # -------- 可选：生成后做均匀/随机取帧（按 fps 保持时长不变）--------
    out_fps: Optional[float] = None  # 例如 8 / 12；None 表示不下采样
    frame_sample: FrameSampleMode = "uniform"
    frame_sample_seed: int = 0

    # -------- Keyframe-by-entropy（真正裁剪 latent 时间维）--------
    keyframe_by_entropy: bool = False
    entropy_steps: int = 5
    entropy_mode: str = "mean"  # "last" | "mean" | "ema"
    entropy_ema_alpha: float = 0.6
    entropy_block_idx: int = -1  # -1 = last block
    keyframe_topk: int = 16
    keyframe_cover: bool = True
    use_nonkey_context: bool = True
    debug_dir: Optional[str] = None
    save_debug_pt: bool = True
    profile_timing: bool = True
    keyframe_out_fps: Optional[float] = None

    # -------- 设备相关 --------
    device_id: int = 0  # 与 wan-main/generate.py 的 device_id 对齐（int）
    t5_cpu: bool = False  # 如显存很紧，可 True 试试（会慢）


def _wan_video_to_vdit_frames(video: torch.Tensor) -> torch.Tensor:
    """
    WAN:   [C,T,H,W] float, 常用 value_range 约为 [-1,1]
    VDiT:  [T,3,H,W] float, value_range [0,1]
    """
    if video.ndim != 4:
        raise ValueError(f"Expect WAN video [C,T,H,W], got {tuple(video.shape)}")
    c, _t, _h, _w = video.shape
    if c != 3:
        raise ValueError(f"Expect C=3 RGB video, got C={c}")

    # [-1,1] -> [0,1]
    frames = (video.clamp(-1.0, 1.0) + 1.0) * 0.5
    frames = frames.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W] -> [T,3,H,W]
    return frames


@torch.no_grad()
def generate_wan_frames(
    *,
    prompt: str,
    ckpt_dir: str,
    cfg: WanGenerateConfig,
) -> Tuple[torch.Tensor, float]:
    """
    返回：
      frames: [T,3,H,W] float in [0,1] （CPU tensor）
      fps:    float (采样后 fps；若没设 out_fps 则为 WAN config 的 sample_fps)
    """
    if cfg.task not in WAN_CONFIGS:
        raise ValueError(f"Unknown task: {cfg.task}. Valid: {list(WAN_CONFIGS.keys())}")
    if cfg.size not in SIZE_CONFIGS:
        raise ValueError(f"Unknown size: {cfg.size}. Valid: {list(SIZE_CONFIGS.keys())}")

    wan_cfg = WAN_CONFIGS[cfg.task]
    fps_src = float(getattr(wan_cfg, "sample_fps", 24.0))
    fps_tgt = fps_src

    # 构建 WAN pipeline（单卡、非分布式：rank=0）
    model = wan.WanT2V(
        config=wan_cfg,
        checkpoint_dir=ckpt_dir,
        device_id=cfg.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=cfg.t5_cpu,
    )

    # 生成 WAN 原生视频张量：[C,T,H,W]
    video = model.generate(
        prompt,
        size=SIZE_CONFIGS[cfg.size],
        frame_num=cfg.frame_num,
        shift=cfg.sample_shift,
        sample_solver=cfg.sample_solver,
        sampling_steps=cfg.sample_steps,
        guide_scale=cfg.guide_scale,
        seed=cfg.seed,
        offload_model=cfg.offload_model,
        keyframe_by_entropy=cfg.keyframe_by_entropy,
        entropy_steps=cfg.entropy_steps,
        entropy_mode=cfg.entropy_mode,
        entropy_ema_alpha=cfg.entropy_ema_alpha,
        entropy_block_idx=cfg.entropy_block_idx,
        keyframe_topk=cfg.keyframe_topk,
        keyframe_cover=cfg.keyframe_cover,
        use_nonkey_context=cfg.use_nonkey_context,
        debug_dir=cfg.debug_dir,
        save_debug_pt=cfg.save_debug_pt,
        profile_timing=cfg.profile_timing,
    )

    if cfg.keyframe_by_entropy:
        t_out = int(video.shape[1])
        t_full = int(cfg.frame_num)
        if t_full > 0:
            fps_tgt = fps_src * (t_out / float(t_full))
        if cfg.keyframe_out_fps is not None:
            fps_tgt = float(cfg.keyframe_out_fps)

    # 可选：按 out_fps 做“均匀/随机取帧”（保持时长不变）
    if cfg.out_fps is not None and not cfg.keyframe_by_entropy:
        fps_tgt = float(cfg.out_fps)
        if fps_tgt > 0 and abs(fps_tgt - fps_src) > 1e-6:
            video, _idx = resample_video_tensor(
                video,
                fps_src=fps_src,
                fps_tgt=fps_tgt,
                mode=cfg.frame_sample,
                seed=cfg.frame_sample_seed,
            )

    frames = _wan_video_to_vdit_frames(video).float().cpu()

    # 主动释放 WAN 占用显存（对串联 pipeline 很关键）
    del video
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 确保返回的 fps 是 Python float 类型（不是 numpy.float64）
    return frames, float(fps_tgt)


@register_generator("wan")
class WanGenerator:
    def __init__(self, ckpt_dir: str, cfg: WanGenerateConfig):
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[torch.Tensor, float]:
        return generate_wan_frames(prompt=prompt, ckpt_dir=self.ckpt_dir, cfg=self.cfg)
