import torch
import torch.nn as nn
from typing import (
    Optional,
    Callable,
    Tuple
)


class ResampleLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        start_with_patch: Optional[bool]=False,
        patch_size: Optional[Tuple[int, int]]=[128, 128],
        feature_values: Optional[list]=[128, (1, 1), 64, (1, 1), 3],
        hiden_act: Callable[..., nn.Module]=nn.Sigmoid,
        use_res: Optional[bool]=True,
        return_scales: Optional[bool]=False,
        space_dim: Optional[str]="2d"
    ) -> None:
        
        super().__init__()
        
        self.res = use_res
        self.rs = return_scales
        self.fv = feature_values
        
        in_f = in_features
        self.blocks = []
        if start_with_patch:
            self.patch_sample = nn.Upsample(size=patch_size)

        for v in self.fv:
            if space_dim == "2d":
                if isinstance(v, tuple):
                    block = nn.Upsample(scale_factor=(
                        2 ** v[0], 
                        2 ** v[1]
                    ))
                
                elif isinstance(v, int):
                    print(v)
                    block = nn.Sequential(
                        nn.Conv2d(in_f, v, (3, 3), 1, 1),
                        hiden_act(),
                        nn.BatchNorm2d(v)
                    )
                    in_f = v
            
            elif space_dim == "3d":
                if isinstance(v, tuple):
                    block = nn.Upsample(scale_factor=(
                        2 ** v[0], 
                        2 ** v[1],
                        2 ** v[2]
                    ))
                
                elif isinstance(v, int):
                    block = nn.Sequential(
                        nn.Conv3d(in_f, v, (3, 3, 3), 1, 1),
                        hiden_act(),
                        nn.BatchNorm3d(v)
                    )
                    in_f = v

            
            self.blocks += [block, ]
        
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        features = []
        sc_features = []
        if hasattr(self, "patch_sample"):
            x = self.patch_sample(x)

        for idx, block in enumerate(self.blocks):
            
            x = block(x)
            if self.res:
                if (idx != 0 and
                    not isinstance(self.fv[idx - 1], tuple) and
                    not isinstance(self.fv[idx], tuple)):
                    x += features[-1]

            if self.rs:
                if ((idx == len(self.blocks) - 1) or 
                    isinstance(self.fv[idx + 1], tuple)):
                    sc_features += [x, ]
            
            features += [x, ]
        
        if self.rs:
            return sc_features[::-1]
        
        return x



if __name__ == "__main__":

    model = ResampleLayer(3, return_scales=True, feature_values=[
        128, (-1, -1),
        64, (-1, -1),
        32, (-1, -1)
    ])
    test = torch.normal(0, 1, (1, 3, 256, 256))
    out = model(test)
    for f in out:
        print(f.size())


