import os
import h5py 
import subprocess as sps 
import numpy as np
import torch 
import warnings
warnings.filterwarnings("ignore")

from moviepy import VideoFileClip
from urllib.parse import urlparse
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
import librosa as lib
from torchvision.transforms import Resize
from typing import (
    Optional,
    Tuple
)


import matplotlib.pyplot as plt
plt.style.use("dark_background")

gen_p = str(Path(__file__).parent.parent.parent)
gen_p = os.path.join(gen_p, "data")


def _gen_path(path: str) -> None:

    path = os.path.join(gen_p, "files")
    if not os.path.exists(path):
        os.mkdir(path)
    
    return path

def load_h5split(
    url: str,
    target_p: Optional[str]=None
) -> None:

    
    path = _gen_path("files")
    if target_p is None:
        target_p = urlparse(url)
        target_p = os.path.basename(target_p.path)
        target_p = os.path.join(path, target_p)
    
    print(target_p)
    cmd = ["wget", "-P", target_p, url]
    sps.run(cmd)




def parse_avi(
    path: str, 
    target_path: str, 
    img_size: Tuple[int]=(128, 128),
    step: int=100,
    batch_size: int=32
) -> None:

    res_tf = Resize(img_size)
    clip = VideoFileClip(path)
    total_imgs = []
    total_sounds = []

    samples_n = 0
    img_batch = []
    sound_batch = []
    for audio, frame in zip(
        clip.audio.iter_frames(fps=step),
        clip.iter_frames(fps=step)
    ):
        
        if samples_n == batch_size:
            samples_n = 0
            img_batch = torch.cat(img_batch, dim=0).permute(1, 0, 2, 3)
            sound_batch = torch.cat(sound_batch, dim=0)
            total_imgs += [img_batch.unsqueeze(dim=0), ]
            total_sounds += [sound_batch.unsqueeze(dim=0), ]

            img_batch = []
            sound_batch = []

        img = res_tf((torch.Tensor(frame) / 256).permute(-1, 0, 1).unsqueeze(dim=0))
        sound = torch.Tensor(audio).view(1, 2)
        img_batch += [img, ]
        sound_batch += [sound, ]

        samples_n += 1

    total_imgs = torch.cat(total_imgs, dim=0)
    total_sounds = torch.cat(total_sounds, dim=0)
    return total_imgs, total_sounds
    
    

    
        
    # clip.write_videofile(f"{target_path}.mp4")
    # audio_clip.write_audiofile(f"{target_path}.wav")

if __name__ == "__main__":

    
    from torchvision.utils import make_grid
    # url = "https://github.com/facebookresearch/FAIR-Play/blob/main/splits/split1/train.h5"
    # load_h5split(url=url)
    # path = "/home/ram/Downloads/test(1).h5"
    # parse_h5(path)
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.background_color = [0, 0, 0]

    path = "/home/ram/Downloads/video/001.AVI"
    # path = "/home/ram/Desktop/own_projects/PavFGS/test.wav"
    img_seq, sound = parse_avi(path, target_path="test")
    print(img_seq.size(), sound.size())
    # plotter.add_volume(img_seq[0, ...].numpy(), cmap="turbo", opacity="geom")
    # plotter.show()

    # img_seq = img_seq.permute(1, 0, 2, 3)[:16, ...]
    # grid = make_grid(img_seq).permute(1, 2, 0)
    # print(grid.size())
    # _, axis = plt.subplots()
    # axis.imshow(grid)
    # plt.show()

    # signal, _ = lib.load(path)
    # signal = torch.Tensor(signal).unsqueeze(dim=0)
    # mel = MelSpectrogram()
    # out = mel(signal)
    
    # import matplotlib.pyplot as plt
    # plt.style.use("dark_background")
    # _, axis = plt.subplots()
    # axis.imshow(out.squeeze(), cmap="turbo")
    # plt.show()