import torch
import torch.nn as nn

from argparse import ArgumentParser
from torch.optim import Adam
from gsplat import rasterization


class GShead(nn.Module):

    def __init__(
        self,
        in_features: int,
    ) -> None:
        
        super().__init__()
        self.geom_head = nn.Sequential(
            nn.Linear(in_features, 10),
            nn.ReLU()
        )
        self.color_head = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Sigmoid()
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        means, scales, quats = self.geom_head(x).unbind([3, 3, 4], dim=-1)
        colors, opacities = self.color_head(x).unbind([3, 1], dim=-1)
        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "colors": colors,
            "opacities": opacities
        }


def run_test(
    noise_dim: int,
    n_epochs: int,
    width: int, height: int,
    weights_path: str
):
    
    gs_head = GShead(noise_dim)
    optim = Adam(params=gs_head.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    noise = torch.normal(0, 1, (12, noise_dim))
    gt_img = torch.normal(0, 1, (1, 3, width, height))
    Ks = torch.Tensor([
        [64.0, 0, 63.5],
        [0, 64.0, 63.5],
        [0, 0, 1]
    ])
    ViewMat = torch.eye(4)
    ViewMat[2, -1] = 8.0
    losses = []
    for idx in range(n_epochs):
        
        optim.zero_grad()
        gs_item = gs_head(noise)
        rendered_img = rasterization(
            **gs_head,
            Ks=Ks[None],
            viewmats=ViewMat[None],
            width=width, height=height
        )[0][0]
        print(rendered_img.size())
        
        loss = loss_fn(gt_img, rendered_img)
        loss.backward()
        optim.step()
        print(f"EPOCH [{idx}] finished with loss: {loss.item()}")

    state_dict = gs_head.state_dict()
    torch.save(state_dict, weights_path)

parser = ArgumentParser()
parser.add_argument("--weights-path", type=str, default="weights.pt")
parser.add_argument("--noise-dim", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)

args = parser.parse_args()





        