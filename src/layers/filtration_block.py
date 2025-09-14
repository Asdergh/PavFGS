import torch 
import torch.nn as nn
from typing import (
    Optional,
    Tuple,
    Union,
    Callable
)


class FilterBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        hiden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        filt_depth: Optional[int]=2,
        filt_type: Optional[str]="local", #[local, global]
        features_only: Optional[bool]=False,
        hiden_act: Callable[..., nn.Module]=nn.GELU,
    ) -> None:
        
        N = filt_depth
        _conv_params = {
            "global": {
                "kernel_size": (5, 5, 5),
                "stride": (1, 1, 1),
                "padding": (2, 2, 2)
            },
            "local": {
                "kernel_size": (3, 3, 3),
                "stride": (1, 1, 1),
                "padding": (1, 1, 1)
            }
        }

        super().__init__()
        hiden_features = (
            hiden_features
            if hiden_features is not None
            else in_features
        )
        out_features = (
            out_features
            if out_features is not None
            else in_features
        )
        self.f_only = features_only

        self.blocks = []
        in_f = in_features
        out_f = out_features
        for idx in range(N):

            if idx == 0:
               in_f = in_features
               out_f = hiden_features
            
            elif idx == (N - 1):
                print("lol")
                in_f = hiden_features
                out_f = out_features

            block = nn.ModuleDict({
                "main": nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_f,
                            out_channels=out_f,
                            **_conv_params[filt_type]
                        ),
                        hiden_act(),
                        nn.BatchNorm3d(out_f)
                    ),
                "film": nn.Sequential(
                    nn.Conv3d(
                        in_channels=out_f,
                        out_channels=out_f * 2,
                        **_conv_params[filt_type]
                    ),
                    nn.Sigmoid()
                )                
            })
            self.blocks += [block, ] 
        
        self.blocks = nn.ModuleList(self.blocks)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.f_only:
            features = []

        for block in self.blocks:
            
            x = block["main"](x)
            x_film  = block["film"](x)

            b, c, d, w, h = x_film.size()
            x_film = x_film.view(b, 2, c // 2, d, w, h)
            scale, shift = x_film.unbind(1)
            x = scale * x + shift

            if self.f_only:
                features.append(x)
        
        if self.f_only:
            return features
        
        return x
        

