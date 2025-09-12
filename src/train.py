import torch 
import torch.nn as nn
import os
import shutil
import yaml

from argparse import ArgumentParser
from torch.optim import Adam
from gsplat import rasterization

parser = ArgumentParser()
parser.add_argument("--hyper-params", help="input hyper parameters")
parser.add_argument("--train-log", help="path with final results")
args = parser.parse_args()

print(args)
with open(args.hyper_params) as file:
    hyper_cfg = yaml.load(file, Loader=yaml.SafeLoader)

print(list(hyper_cfg.keys()))
W, H = hyper_cfg["WH"]
Npts = hyper_cfg["points_n"]

features = {
    "means": nn.Parameter(torch.rand((Npts, 3))),
    "quats": nn.Parameter(torch.rand(Npts, 4)),
    "scales": nn.Parameter(torch.rand(Npts, 3)),
    "colors": nn.Parameter(torch.rand(Npts, 3)),
    "opacities": nn.Parameter(torch.rand(Npts, ))
}
_ViewMat = torch.eye(4)
_ViewMat[:3, -1] = torch.tensor([0, 0, 8])
_Ks = torch.Tensor([
    [64.0, 0, 63.5],
    [0, 64.0, 63.5],
    [0, 0, 1]
])

gt_img = torch.normal(0, 1, (3, W, H))
optim = Adam(params=list(features.values()), lr=0.1)
loss_fn = nn.MSELoss()

for _ in range(hyper_cfg["epochs"]):
    
    optim.zero_grad()
    rendered_img = rasterization(
        **features,
        viewmats=_ViewMat[None],
        Ks=_Ks[None],
        width=W, height=H,
        packed=False
    )[0][0]
    
    loss = loss_fn(rendered_img, gt_img)
    loss.backward()
    optim.step()


log_path = args.train_log.split(".")[0]
arch_type = args.train_log.split(".")[1]
if not os.path.exists(log_path):
    os.mkdir(log_path)

for name, features in features.items():
    feature = feature.detach().cpu()
    ff = os.path.join(log_path, f"{name}.pt")
    torch.save(feature, ff)

shutil.make_archive(log_path, log_path, arch_type)





