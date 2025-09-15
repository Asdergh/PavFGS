import torch
import numpy as np
import os

from moviepy import VideoFileClip
from typing import (
    Optional,
    Tuple
)


def read_avi(
    path: str, 
    fps: Optional[int]=100,
    sequence_len: Optional[int]=128,
    shuffle: Optional[bool]=False
) -> Tuple:


    
    clip = VideoFileClip(path)
    audio_clip = clip.audio

    imgs = np.asarray(list(clip.iter_frames(fps=fps)))
    imgs = (imgs / 255.0).astype(np.float32)
    idx = np.arange(0, sequence_len)
    if shuffle:
        pv = imgs.shape[0] - sequence_len
        idx = np.arange(pv, pv + sequence_len)

    imgs = imgs[idx]  
    sound = np.asarray(audio_clip.to_soundarray()).T.mean(axis=0)

    return (imgs, sound)