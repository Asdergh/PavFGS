import torch 
import torch.nn as nn

class Gap(nn.Module): 

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C = x.size()[:2]
        features = []
        for batch in range(B):
            weights = []
            for level in range(C):
                sample = x[:, level, ...]
                avg_score = torch.sigmoid(sample.mean()).view(1, )
                weights += [avg_score, ]
            
            weights = torch.cat(weights, dim=0).view(1, C)
            features += [weights, ]
        
        return torch.cat(features, dim=0).view(B, C)



            