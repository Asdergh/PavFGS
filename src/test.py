import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from tqdm import tqdm
from scipy.spatial.transform.rotation import Rotation as R
from PIL import Image
from torchvision.transforms import functional as Fv
from gsplat import rasterization
from torch.optim import Adam, SGD
from matplotlib.animation import FuncAnimation, ArtistAnimation
from src.submodules.VGGT.vggt.models.vggt import VGGT
from src.utils.functional import ssim 



device = "cuda"

W, H = (224, 224)
POINTS_N = W * H
NOISE_DIM = 128
EPOCHS_N = 1000


ViewMat = torch.eye(4).to(device)
ViewMat[3, -1] = 500.0
K = torch.Tensor([
    [6.0, 0.0, 111.5],
    [0.0, 6.0, 111.5],
    [0.0, 0.0, 1]
]).to(device)

gt_img_path = "/media/test/T7/mmpr_dataset_1/mav0/cam0/data/000001681944017000.png"
gt_img = Image.open(gt_img_path)
gt_img = Fv.pil_to_tensor(gt_img) / 255.0
gt_img = Fv.resize(gt_img, (W, H)).to(device)




class TestModel(nn.Module):

    def __init__(self, in_features: int) -> None:

        super().__init__()
        self.hiden_fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU()
        )
        self.geom_head = nn.Sequential(
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.LayerNorm(10)
        )
        self.color_head = nn.Sequential(
            nn.Linear(32, 4),
            nn.Sigmoid(),
            nn.LayerNorm(4)
        )
    
    def forward(self, x: torch.Tensor) -> dict: 

        h = self.hiden_fc(x)
        means, scales, quats = self.geom_head(h).split([3, 3, 4], dim=-1)
        colors, opacities = self.color_head(h).split([3, 1], dim=-1)
        opacities = opacities.squeeze()

        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "colors": colors,
            "opacities": opacities
        }

noise_vector = torch.normal(0, 1, (POINTS_N, NOISE_DIM)).to(device)
frames_buffer = []
losses = []


# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()


weights_path = "/media/test/T7/model.pt"
gt_img = gt_img.view(1, 1, *gt_img.size())
model = VGGT(
    enable_point=True,
    enable_track=True,
    enable_depth=True,
    enable_camera=True
).to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True))

print(gt_img.size())
model_out = model(gt_img)
print(list(model_out.keys()))

depth = model_out["depth"].squeeze().cpu().detach()
pts = model_out["world_points"].squeeze().cpu().detach()
pose = model_out["pose_enc"].squeeze().cpu().detach()


# t, quat = pose[:3], pose[3:-2]
# Rmat = torch.Tensor(R.from_quat(quat).as_matrix())
# ViewMat = torch.zeros(4, 4)
# ViewMat[:3, :3] = Rmat
# ViewMat[:3, -1] = t
# ViewMat = ViewMat.to(device)



means =torch.flatten(pts.detach(), end_dim=-2)
quats = torch.cat([
    torch.normal(0, 1, (POINTS_N, 3)),
    torch.ones(POINTS_N, 1)
], dim=-1).requires_grad_(True)
scales = torch.ones(POINTS_N, 3)
colors = torch.flatten(gt_img.squeeze().permute(1, 2, 0), end_dim=-2)
opacities = torch.ones((POINTS_N, ))

means = nn.Parameter(means.to(device).requires_grad_(True))
quats = nn.Parameter(quats.to(device).requires_grad_(True))
scales = nn.Parameter(scales.to(device).requires_grad_(True))
colors = nn.Parameter(colors.to(device).requires_grad_(True))
opacities = nn.Parameter(opacities.to(device).requires_grad_(True))

gt_img = gt_img.squeeze()

optim = Adam([
    {"params": [means], "lr": 0.01, "name": "means"},
    {"params": [quats], "lr": 0.01, "name": "quats"},
    {"params": [scales], "lr": 0.01},
    {"params": [colors], "lr": 0.01},
    {"params": [opacities], "lr": 0.01}
], lr=0.001)

for group in optim.param_groups:
    if "name" in list(group.keys()):
        state = optim.state.get(group["params"][0], None)
        print(group["params"][0].size())
        print(state["exp_avg"].size())
        print(state["exp_avg_sq"].size())
    




# _, axis = plt.subplots(ncols=5)
# axis[0].imshow(gt_img.squeeze().cpu().permute(1, 2, 0))
# axis[1].imshow(depth, cmap="turbo")
# axis[2].imshow(pts[..., 0], cmap="turbo")
# axis[3].imshow(pts[..., 1], cmap="turbo")
# axis[4].imshow(pts[..., 2], cmap="turbo")
# plt.show()


# with tqdm(
#     desc="GS Training ...",
#     colour="green",
#     ascii=":>",
#     total=EPOCHS_N
# ) as pbar:
#     for _ in range(EPOCHS_N):

#         optim.zero_grad()
#         rendered_image = rasterization(
#             means=means,
#             quats=quats,
#             scales=scales,
#             colors=colors,
#             opacities=opacities,
#             width=W, height=H,
#             Ks=K[None],
#             viewmats=ViewMat[None],
#             packed=False
#         )[0].squeeze()

#         rendered_image = rendered_image.permute(-1, 0, 1)
#         L1_term = loss_fn(gt_img, rendered_image)
#         Dssim_term = 1 - ssim(gt_img, rendered_image, device=device)
#         loss = L1_term + Dssim_term

#         loss.backward()

#         optim.step()
#         losses.append(loss.item())
        
#         img = (rendered_image.permute(1, 2, 0) * 255.0).cpu().detach().numpy()
#         img = img.astype(np.uint8)
#         img = Image.fromarray(img)
#         frames_buffer.append(img)

#         pbar.update(1)




# path = "out_vis.gif"
# frames_buffer[0].save(
#     path,
#     save_all=True,
#     append_images=frames_buffer[1:],
#     loop=0
# )

# _, axis = plt.subplots()
# axis.plot(losses, color="red", linestyle="--")
# plt.show()






    




