import torch 
from dataclasses import dataclass
from typing import Optional


@dataclass 
class BasePcd:
    pts: torch.Tensor
    colors: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]