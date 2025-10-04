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
    



def gauss_kernel(kernel_size: int) -> Tuple:

    labels = np.linspace(-1, 1, kernel_size)
    exp = np.exp(labels ** 2 / 2)
    GoP = np.outer(exp, exp) 
    GoP /= GoP.sum()

    return torch.Tensor(GoP)

def sobel_kernel(kernel_size: int, return_full: Optional[bool]=False) -> Tuple:

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
    device: Optional[str]="cpu",
    kernel_type: Optional[str]="gauss" #[gauss, sobel]
) -> Union[Tuple, torch.Tensor]:
    

    if kernel_type == "gauss":
        GoP = gauss_kernel(kernel_size)
    
    elif kernel_type == "sobel":
        GoP = sobel_kernel(kernel_size, return_full=True)
    
    else:
        raise ValueError("unknown kernel_type!!")
    
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


def build_rotation(r):

    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)
    

def inverse_sigmoid(x):
    return torch.log(x/(1-x))



    


