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
from src.scene.gaussian_model import GaussianModel



device = "cuda"

W, H = (112, 112)
POINTS_N = W * H
NOISE_DIM = 128
EPOCHS_N = 1000

gs = GaussianModel(device=device)

ViewMat = torch.eye(4).to(device)
ViewMat[3, -1] = 500.0
K = torch.Tensor([
    [51.0, 0.0, 56.5],
    [0.0, 51.0, 56.5],
    [0.0, 0.0, 1]
]).to(device)

gt_img_path = "/media/test/T7/alimi_img.png"
gt_img = Image.open(gt_img_path)
gt_img = Fv.pil_to_tensor(gt_img) / 255.0
gt_img = Fv.resize(gt_img, (W, H)).to(device)



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



means = torch.flatten(pts.detach(), end_dim=-2)
gs.load_from_pts(means)
gs.setup_optimizer([0.01, 0.01, 0.01, 0.01])

gt_img = gt_img.squeeze()

    




# _, axis = plt.subplots(ncols=5)
# axis[0].imshow(gt_img.squeeze().cpu().permute(1, 2, 0))
# axis[1].imshow(depth, cmap="turbo")
# axis[2].imshow(pts[..., 0], cmap="turbo")
# axis[3].imshow(pts[..., 1], cmap="turbo")
# axis[4].imshow(pts[..., 2], cmap="turbo")
# plt.show()

losses = []
frames_buffer = []
with tqdm(
    desc="GS Training ...",
    colour="green",
    ascii=":>",
    total=EPOCHS_N
) as pbar:
    for idx in range(EPOCHS_N):

        gs.optimizer.zero_grad()
        gs_items = gs.get_items()
        rendered_image = rasterization(
            **gs_items,
            width=W, height=H,
            Ks=K[None],
            viewmats=ViewMat[None],
            packed=False
        )[0].squeeze()
    

        rendered_image = rendered_image.permute(-1, 0, 1)
        L1_term = loss_fn(gt_img, rendered_image)
        Dssim_term = 1 - ssim(gt_img, rendered_image, device=device)
        loss = L1_term + Dssim_term
        loss.backward()

        gs.optimizer.step()
        losses.append(loss.item())

        
        img = (rendered_image.permute(1, 2, 0) * 255.0).cpu().detach().numpy()
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        frames_buffer.append(img)

        pbar.update(1)





path = "out_vis.gif"
frames_buffer[0].save(
    path,
    save_all=True,
    append_images=frames_buffer[1:],
    loop=0
)
gs.save_ply("/media/test/T7/test_gs.ply")

_, axis = plt.subplots()
axis.plot(losses, color="red", linestyle="--")
plt.show()






    




