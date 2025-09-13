import torch
import torch.nn as nn

from src.layers import (
    ResampleLayer,
    Mlp
)
from typing import (
    Optional,
    Callable
)
from torchaudio.transforms import MelSpectrogram


class AudioEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hiden_features: Optional[int]=None,
        drop_rate: Optional[float]=0.32,
        sample_rate: Optional[int]=22050,
        patch_size: Optional[int]=32,
        n_mels: Optional[int]=128,
        max_sequence_len: Optional[int]=256,
        hiden_act_conv: Callable[..., nn.Module]=nn.GELU,
        hiden_act_fcn: Callable[..., nn.Module]=nn.Sigmoid,
        use_res: Optional[bool]=False,
        resample_fl: Optional[list]=[128, (-1, -1), 64, (-1, -1), 1, (-1, -1)]
    ) -> None:
        
        super().__init__()
        self.mel = MelSpectrogram(
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        self.sr = sample_rate
        
        self.patch_size = patch_size
        self.sequence_len = max_sequence_len
        self.resample = ResampleLayer(
            in_features=1,
            patch_size=(n_mels, max_sequence_len),
            feature_values=resample_fl,
            hiden_act=hiden_act_conv,
            use_res=use_res,
            return_scales=None,
            start_with_patch=True
        )
        
        vs_n = 0
        for v in resample_fl:
            if isinstance(v, tuple):
                vs_n += 1
        
        lms = n_mels // (2 ** vs_n)
        lmss = max_sequence_len // (2 ** vs_n)
        self.ft = nn.Flatten()
        self.fcn = Mlp(
            in_features=(lms * lmss),
            hiden_features=hiden_features,
            out_features=embedding_dim,
            act=hiden_act_fcn,
            drop_rate=drop_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mel(x).unsqueeze(dim=1)
        print(x.size())
        x = self.resample(x)
        print(x.size())
        x = self.ft(x)
        print(x.size())
        x = self.fcn(x)
        return x





