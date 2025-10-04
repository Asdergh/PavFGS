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
from src.utils import (
    ssim,
    eval_sh
)
# from src.scene.gaussian_model import GaussianModel
from src.scene.gaussian_model_origin import (
    GaussianModel,
    BasicPointCloud,
    TrainingConfig
)

device = "cuda"

W, H = (112, 112)
POINTS_N = W * H
NOISE_DIM = 128
EPOCHS_N = 2000
DENSIFY_ITER = 300
DENSIFY_GRAD_MAX = 0.02
DENSIFY_OPACITY_MIN = 0.005
DENSIFY_SCENE_EXTENT = 12.3
DENSIFICATION_EPOCHS = [100, 300]



# ViewMat = torch.eye(4).to(device)
# ViewMat[3, -1] = 500.0
K = torch.Tensor([
    [5.0, 0.0, 56.0],
    [0.0, 5.0, 56.0],
    [0.0, 0.0, 1]
]).to(device)

gt_img_path = "/media/test/T7/mmpr_dataset_1/mav0/cam0/data/000001735217372000.png"
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
gt_img = gt_img.squeeze()
print(list(model_out.keys()))

depth = model_out["depth"].squeeze().cpu().detach()
pts = model_out["world_points"].squeeze().cpu().detach()
pose = model_out["pose_enc"].squeeze().cpu().detach()


t, quat = pose[:3], pose[3:-2]
Rmat = torch.Tensor(R.from_quat(quat).as_matrix())
ViewMat = torch.zeros(4, 4)
ViewMat[:3, :3] = Rmat
ViewMat[:3, -1] = t
ViewMat = ViewMat.to(device)



# print(gt_img.min(), gt_img.max())
pts = torch.flatten(pts.detach(), end_dim=-2).numpy()
pcd = BasicPointCloud(
    points=pts,
    colors=torch.flatten(gt_img.permute(1, 2, 0), end_dim=-2).cpu().numpy(),
    normals=np.zeros_like(pts)
)
# gs.load_from_pts(means)
# gs.setup_optimizer([0.01, 0.01, 0.01, 0.01])
tr_args = TrainingConfig()
gs = GaussianModel(3)
gs.create_from_pcd(pcd, 0.1, 1, 1.0)
gs.training_setup(tr_args)




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

        gs_items = gs.get_items()
        features_dc, features_rest = gs_items["colors"]
        colors = features_dc 
        # + eval_sh(
        #     deg=gs.max_sh_degree,
        #     sh=features_rest.permute(0, 2, 1),
        #     dirs=gs_items["means"]
        # )
        # print(colors.size())
        gs_items["colors"] = colors
        

        rendered_image = rasterization(
            **gs_items,
            width=W, height=H,
            Ks=K[None],
            viewmats=ViewMat[None],
            packed=False
        )[0].squeeze()
        print(78 * "=")
        for item in gs_items:
            attr = gs_items[item]
            print(item, attr.requires_grad, attr.min().item(), attr.mean().item(), attr.max().item())

        # _, axis = plt.subplots()
        # axis.imshow(rendered_image.squeeze().detach().cpu())
        # plt.show()
        if idx == 10:
            break
        


        rendered_image = rendered_image.permute(-1, 0, 1)
        L1_term = loss_fn(gt_img, rendered_image)
        Dssim_term = 1 - ssim(gt_img, rendered_image, device=device)
        loss = L1_term + Dssim_term
        loss.backward()

        # with torch.no_grad():
        #     xyz_grad = gs._xyz
        #     gs.add_densification_stats(xyz_grad, torch.ones(xyz_grad.size()[0], dtype=torch.bool))
        #     if idx in DENSIFICATION_EPOCHS:
        #         # print(xyz_grad.size())
        #         gs.densify_and_prune(
        #             max_grad=DENSIFY_GRAD_MAX,
        #             min_opacity=DENSIFY_OPACITY_MIN,
        #             extent=DENSIFY_SCENE_EXTENT,
        #             max_screen_size=112
        #         )

        gs.optimizer.step()
        gs.optimizer.zero_grad()
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






    




