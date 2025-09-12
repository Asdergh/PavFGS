import torch
import torch.nn as nn
import librosa as lib
from typing import (
    Union,
    Optional,
    Callable
)
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from torchaudio.transforms import (
    MelSpectrogram,
    Spectrogram
)



mel = MelSpectrogram(n_mels=256)
spec = Spectrogram()
path = "/home/ram/Downloads/file_example_WAV_1MG.wav"
signal, sr = lib.load(path)
signal = torch.Tensor(signal)
N = 11
w_size = signal.size()[0] // N
signals = []
for idx in range(N):

    try:
        sslice = signal[:, idx * w_size: (idx + 1) * w_size]
        sslice = sslice.view(*sslice.size(), 1)
        signals += [sslice, ]
    
    except BaseException:
        pass

signals = torch.cat(signals)
mel_map = mel(signals)
print(mel_map.size())




# class AudioAlternatingAgregator(nn.Module):

#     def __init__(
#         self,
#         sample_rate: Optional[int]=22050,
#         patch_size: Optional[int]=32,
#         n_mels: Optional[int]=128,
#         max_sequence_len: Optional[int]=200,
#     ) -> None:
        
#         super().__init__()
#         self.mel = MelSpectrogram(n_mels=n_mels)
#         self.sr = sample_rate
        
#         self.patch_size = patch_size
#         self.sequence_len = max_sequence_len
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         pass 
