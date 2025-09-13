import torch as th
import torch.nn as nn
from typing import Optional, Union, Callable


class Mlp(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int]=None,
        hiden_features: Optional[int]=None,
        act: Callable[..., nn.Module]=nn.GELU,
        drop_rate: Optional[float]=0.32,
    ):
        
        super().__init__()
        hiden_features = (hiden_features if hiden_features is not None else in_features)
        out_features = (out_features if out_features is not None else hiden_features)

        self.fc1 = nn.Linear(in_features, hiden_features)
        self.fc2 = nn.Linear(hiden_features, out_features) 
        self.dp = nn.Dropout(drop_rate)

        if isinstance(act, nn.Softmax):
            self.act = act(dim=-1)
        
        else:
            self.act = act()
        
        self.layer_norm_fc1 = nn.LayerNorm(hiden_features)
        self.layer_norm_fc2 = nn.LayerNorm(out_features)

    def forward(self, x: th.Tensor) -> th.Tensor:

        x = self.fc1(x)
        x = self.dp(x)
        x = self.act(x)
        self.layer_norm_fc1(x)

        x = self.fc2(x)
        x = self.dp(x)
        x = self.act(x)
        x = self.layer_norm_fc2(x)

        return x

        