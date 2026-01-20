from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torchvision


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
    frames, _, info = torchvision.io.read_video(path)
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    fps = float(info["video_fps"])
    t, _, h, w = frames.shape
    return frames, VideoInfo(fps=fps, num_frames=t, height=h, width=w)


def write_video_tensor(path: str, frames: torch.Tensor, *, fps: float) -> None:
    """
    frames: [T,3,H,W] float in [0,1]
    """
    frames_u8 = (frames.clamp(0.0, 1.0).permute(0, 2, 3, 1) * 255.0).byte().cpu()
    torchvision.io.write_video(path, frames_u8, fps=float(fps))


