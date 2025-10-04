import numpy as np
import torch 
import os
import cv2

from pathlib import Path
from moviepy import VideoFileClip
from utils.data import read_avi
from tqdm import tqdm
from typing import (
    Optional,
    Tuple
)
from .base import BaseDataset

import matplotlib.pyplot as plt
plt.style.use("dark_background")



class Dh1kDataset(BaseDataset):

    def __init__(
        self,
        data_path: str,
        fps_per_video: Optional[int]=100,
        n_samples: Optional[int]=100,
        device: Optional[str]="cpu",
        width: Optional[int]=128,
        height: Optional[int]=128,
        sequence_len: Optional[int]=128,
        shuffle: Optional[bool]=False
    ) -> None:
        
        super(BaseDataset, self).__init__()
        self.W, self.H = (width, height)
        self.n_samples = n_samples
        self.seq_len = sequence_len
        self.fps = fps_per_video
        self.shuffle = shuffle
        self.device = device

        self.path = data_path
        if self.path is None:
            base_path = Path(__file__).parent
            self.path = os.path.join(base_path, "dh1k_data")
    
        self.get_filepaths()
        self.avi_paths = np.asarray(os.listdir(self.avi_root))
        self.gts_paths = np.asarray(os.listdir(self.gt_root))
        Ns = min(len(self.avi_paths), len(self.gts_paths))

        idx = np.random.randint(0, Ns, (n_samples, ))
        
        self.avi_paths = [
            os.path.join(self.avi_root, path)
            for path in self.avi_paths[idx.tolist()]
        ]
        self.gts_paths = [
            os.path.join(self.gt_root, path)
            for path in self.gts_paths[idx.tolist()]
        ]

        print(self.avi_paths)
        print(self.gts_paths)
        
    

    def __len__(self) -> int:
        Ntotal = 0
        for sub_path in self.gts_paths:
            Ntotal += len(os.listdir(sub_path))

        return Ntotal

 
    def get_filepaths(self) -> None:
        self.gt_root = os.path.join(self.path, "annotation")
        self.avi_root = os.path.join(self.path, "avi_samples")
    

    def _load_avi(self, path: str, N: int) -> None:
        
        clip = VideoFileClip(path)
        audio = clip.audio.to_soundarray().T.mean(axis=0)
        print(audio.shape)
        t = clip.duration
        fps = t / N
        
        imgs = np.asarray(list(clip.iter_frames(1 / fps)))
        return (imgs, audio)
        

    def _load_gt_maps(self, path: str) -> None:
        
        paths = os.listdir(path)
        _buffer = []
        N = len(paths)
        for gtf in os.listdir(path):

            gtf = os.path.join(path, gtf)
            
            gtm = cv2.imread(gtf)
            gtm = cv2.cvtColor(gtm, cv2.COLOR_BGR2GRAY)
            gtm = cv2.resize(gtm, (self.W, self.H))
            _buffer += [np.expand_dims(gtm, axis=0), ]
        
        return (np.concatenate(_buffer, axis=0), N)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:


        avi_path = self.avi_paths[idx]
        gts_path = os.path.join(self.gts_paths[idx], "maps")

        gtm, Ngt = self._load_gt_maps(gts_path)
        imgs, sound = self._load_avi(avi_path, Ngt)
        
        print(imgs.shape, sound.shape)
        return (
            torch.Tensor(imgs).to(self.device),
            torch.Tensor(gtm).to(self.device),
            torch.Tensor(sound)
        )
        

        
if __name__ == "__main__":

    dataset = Dh1kDataset(data_path="/home/ram/Downloads/DH1FK_dataset")
    rgb, gt, sound = dataset[10]
    print(rgb.size(), gt.size(), sound.size())
    



              



                



        


