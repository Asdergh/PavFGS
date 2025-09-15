import torch
import numpy as np
from torch.utils.data import Dataset
from typing import (
    Optional,
    Union,
    Tuple
)
from torchvision.transforms import Resize
from abc import (
    ABC,
    abstractmethod
)


class BaseDataset(ABC, Dataset):

    def __init__(
        self,
        data_path: str,
        device: Optional[str]="cuda",
        width: Optional[int]=128,
        height: Optional[int]=128,
        sequence_len: Optional[int]=128
    ) -> None:

        super().__init__()
        self.path = data_path
        self.device = device
        self.tf = Resize((width, height))
        self.seq_len = sequence_len

    @abstractmethod
    def get_filepaths(self) -> None:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        return self.imgs.shape[0] // self.seq_len

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        pass

     

    
        
        
    