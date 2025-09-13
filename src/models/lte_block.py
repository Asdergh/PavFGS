import torch 
import torch.nn as nn

from src.layers import (
    Gap,
    Mlp
)
from typing import (
    Optional,
    Callable
)

class LTEBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lt_size: int,
        fcn_act: Callable[..., nn.Module]=nn.ReLU
    ) -> None:
        
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Conv3d(in_features, out_features, (3, 3, 3), 1, 1),
            nn.Sigmoid()
        )
        self.gap = Gap()
        self.fcn = nn.Sequential(
            Mlp(
                in_features=out_features, 
                out_features=out_features,
                act=fcn_act
            ),
            nn.Softmax(dim=-1)
        )
        self.learnable_tokens = nn.Parameter(
            torch.normal(0, 1, (1, out_features, *[lt_size, ] * 3)),
            requires_grad=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.in_block(x)
        gap_scores = self.gap(x)
        weights = self.fcn(gap_scores)
        B, C = weights.size()
        
        P = self.learnable_tokens * weights.view(B, C, 1, 1, 1)
        x = x * P + x
        return x



if __name__ == "__main__":

    D, W, H = (64, 64, 64)
    test = torch.normal(0, 1, (1, 1, D, W, H))
    block = LTEBlock(
        in_features=1,
        out_features=3,
        lt_size=64
    )
    out = block(test)
    print(out.size())

        
