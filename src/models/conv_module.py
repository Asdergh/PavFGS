import torch
import torch.nn as nn
from typing import (
    Optional,
    Union,
    Callable
)


class ConvModule(nn.Module):

    def __init__(
        self,
        in_features: int,
        features_values: Optional[list]=[128, (1, 1), 64, (1, 1), 32],
        hiden_act: Callable[..., nn.Module]=nn.Sigmoid,
        use_res: Optional[bool]=True,
        return_features: Optional[bool]=False
    ) -> None:
        
        super().__init__()
        
        self.res = use_res
        self.rf = return_features
        in_f = in_features
        self.blocks = []
        self.fv = features_values

        for v in features_values:
            
            if isinstance(v, tuple):
                block = nn.Upsample(scale_factor=(2 ** v[0], 2 ** v[1]))
            
            elif isinstance(v, int):
                print(v)
                block = nn.Sequential(
                    nn.Conv2d(in_f, v, (3, 3), 1, 1),
                    hiden_act(),
                    nn.BatchNorm2d(v)
                )
                in_f = v

            self.blocks += [block, ]
        
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        features = []
        for idx, block in enumerate(self.blocks):
            
            x = block(x)
            if self.res:
                if (features and 
                    features[-1].size()[1] == x.size()[1] and 
                    idx != 0 and
                    not isinstance(self.fv[idx - 1], tuple) and
                    not isinstance(self.fv[idx], tuple)):
                    x += features[-1]
            
            features += [x, ]
        
        if self.rf:
            return features
        
        return x
            



if __name__ == "__main__":

    test = torch.normal(0, 1, (10, 1, 128, 128))
    model = ConvModule(
        in_features=1,
        features_values=[3, (1, 1), 64, (-1, -1), 128],
        use_res=True,
        return_features=False
    )

    print(model(test).size())