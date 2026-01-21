from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
import torchvision
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    height: int
    width: int


def read_video_tensor(path: str) -> Tuple[torch.Tensor, VideoInfo]:
    """
    返回 frames: [T,3,H,W] float in [0,1]
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    try:
        frames, _, info = torchvision.io.read_video(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read video from {path}: {e}") from e
    
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    # 强制转换为 Python float（避免 numpy.float64）
    fps_raw = info["video_fps"]
    if isinstance(fps_raw, np.floating):
        fps = float(fps_raw.item())
    elif hasattr(fps_raw, 'item'):
        fps = float(fps_raw.item())
    else:
        fps = float(fps_raw)
    t, _, h, w = frames.shape
    return frames, VideoInfo(fps=fps, num_frames=t, height=h, width=w)


def write_video_tensor(path: str, frames: torch.Tensor, *, fps: float) -> None:
    """
    frames: [T,3,H,W] float in [0,1]
    """
    frames_u8 = (frames.clamp(0.0, 1.0).permute(0, 2, 3, 1) * 255.0).byte().cpu()
    
    # PyAV does not like numpy scalars; ensure a plain int fps.
    fps_py = float(fps)
    fps_int = int(round(fps_py))
    torchvision.io.write_video(path, frames_u8, fps=fps_int)
