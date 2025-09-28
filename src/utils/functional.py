import torch 
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from typing import (
    Union,
    Optional,
    Tuple
)
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as Fv
    
    


def _gauss_kernel(kernel_size: int) -> Tuple:

    labels = np.linspace(-1, 1, kernel_size)
    exp = np.exp(labels ** 2 / 2)
    GoP = np.outer(exp, exp) 
    GoP /= GoP.sum()

    return torch.Tensor(GoP)

def _sobel_kernel(kernel_size: int, return_full: Optional[bool]=False) -> Tuple:

    b = []
    for i in range(kernel_size):
        C = math.comb(kernel_size - 1, i)
        b.append(C)


    k = (kernel_size - 1) / 2
    _neg_d = []
    _pos_d = []
    while k >= 0:

        _neg_d.append(-k)
        _pos_d.append(k)
        k -= 1
    
    d = _neg_d[:-1] + _pos_d[::-1]
    GxOp = torch.Tensor(np.outer(b, d))
    GyOp = torch.Tensor(np.outer(d, b))
    
    if return_full:
        return (GxOp @ GyOp)

    return (GxOp, GyOp)


def ssim(
    Img1: torch.Tensor,
    Img2: torch.Tensor,
    K1: Optional[float]=0.01,
    K2: Optional[float]=0.03,
    L: Optional[float]=255.0,
    kernel_size: Optional[int]=3,
    get_ssim_map: Optional[bool]=False,
    device: Optional[str]="cpu"
) -> Union[Tuple, torch.Tensor]:
    


    GoP = _gauss_kernel(kernel_size=kernel_size)
    GoP = GoP.view(1, 1, *GoP.size())
    GoP = GoP.repeat(1, 3, 1, 1).to(device)

    mu_x = F.conv2d(Img1, GoP)
    mu_y = F.conv2d(Img2, GoP)
    sigma_xx = F.conv2d(Img1 * Img1, GoP) - (mu_x.pow(2))
    sigma_yy = F.conv2d(Img2 * Img2, GoP) - (mu_y.pow(2))
    sigma_xy = F.conv2d(Img1 * Img2, GoP) - (mu_x * mu_y)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    S1_denom = 2 * (mu_x * mu_y) + C1
    S1_nom = (mu_x.pow(2) + mu_y.pow(2) + C1)
    S1 = S1_denom / S1_nom

    S2_denom = (2 * sigma_xy + C2)
    S2_nom = (sigma_xx + sigma_yy + C2)
    S2 = S2_denom / S2_nom
 

    SSIM_map = (S1 * S2).squeeze()
    ssim_score = SSIM_map.mean()
    
    if get_ssim_map:
        return (ssim_score, SSIM_map)

    return ssim_score

     


# W, H = (128, 128)
# img1 = Image.open("/media/test/T7/test_img2.png")
# img1 = Fv.pil_to_tensor(img1)
# img1 = Fv.resize(img1, (W, H)).unsqueeze(dim=0).to(torch.float32)

# img2 = Image.open("/media/test/T7/test_img2.png")
# img2 = Fv.pil_to_tensor(img2)
# img2 = Fv.resize(img2, (W, H)).unsqueeze(dim=0).to(torch.float32)

# score, map = ssim(
#     Img1=img1,
#     Img2=img2,
#     get_ssim_map=True,
#     kernel_size=3
# )
# print(map.size())
# print(score)
# _, axis = plt.subplots(ncols=3)
# axis[0].imshow(img1.squeeze().permute(1, 2, 0) / 255.0)
# axis[1].imshow(img2.squeeze().permute(1, 2, 0) / 255.0)
# axis[2].imshow(map, cmap="turbo")


# plt.show()


    


