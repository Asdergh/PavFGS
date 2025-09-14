import torch 
import torch.nn as nn
from src.layers import (
    FilterBlock,
    Gap,
    Mlp
)
from typing import (
    Optional,
    Union,
    Callable,
    Tuple
)


class AmfBlock(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int]=None,
        filt_depth: Optional[int]=1,
        hiden_act: Callable[..., nn.Module]=nn.GELU,
        filtration_order: Optional[list]=["local", "audio", "global"]
    ) -> None:
        
        super().__init__()
        out_features = (
            out_features
            if out_features is not None
            else in_features
        )
        self.f_order = filtration_order
        l_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "filt_type": "local",
            "filt_depth": filt_depth,
            "hiden_act": hiden_act
        }
        g_kwargs = l_kwargs.copy()
        g_kwargs["filt_type"] = "global"

        self.blocks = nn.ModuleDict({
            "global": FilterBlock(**g_kwargs),
            "local": FilterBlock(**l_kwargs),
            "audio": FilterBlock(**l_kwargs)
        })

        self.mlp_blocks = nn.ModuleDict({
            "global": Mlp(out_features * 3, out_features),
            "local": Mlp(out_features * 3, out_features),
            "audio": Mlp(out_features * 3, out_features)
        })

        self.gap = Gap()

    
  
    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        x, Af = input
        B = x.size()[0]
        features = {}
        for f_type in self.f_order:
            f = self.blocks[f_type](x)
            if f_type == "audio":
                f += Af.view(*Af.size(), 1, 1, 1)

            features.update({f_type: f})
        
        F = torch.cat(list(features.values()), dim=1)
        gap_score = self.gap(F)
        weights = []
        for f_type in self.f_order:
            w = self.mlp_blocks[f_type](gap_score)
            weights += [w, ]
        
        weights = torch.cat(weights, dim=-1)
        C = weights.size()[-1] // 3
        weights = torch.sigmoid(weights).view(B, C, 3)
        (Wl, Wg, Wau) = weights.unbind(dim=-1)

        Fl = Wl.view(C, 1, 1, 1) * features["local"] 
        Fg = Wg.view(C, 1, 1, 1) * features["global"] 
        Fau = Wau.view(C, 1, 1, 1) * features["audio"]

        return Fl + Fg + Fau
        

        
        

        
        
        
            
        
        