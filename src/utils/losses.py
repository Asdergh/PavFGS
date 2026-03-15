import torch
import torch.nn as nn
import numpy as np
import math

from typing import Optional, Tuple, Union
from torch.nn import functional as F


def gauss_kernel(kernel_size: int) -> Tuple:

    labels = np.linspace(-1, 1, kernel_size)
    exp = np.exp(labels**2 / 2)
    GoP = np.outer(exp, exp)
    GoP /= GoP.sum()

    return torch.Tensor(GoP)


def sobel_kernel(kernel_size: int, return_full: Optional[bool] = False) -> Tuple:

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
        return GxOp @ GyOp

    return (GxOp, GyOp)


class DSSIMLoss(nn.Module):
    def __init__(
        self,
        K1: Optional[float] = 0.01,
        K2: Optional[float] = 0.03,
        L: Optional[float] = 255.0,
        kernel_size: Optional[int] = 3,
        get_ssim_map: Optional[bool] = False,
        device: Optional[str] = "cuda",
        kernel_type: Optional[str] = "gauss",  # [gauss, sobel]
    ) -> Union[Tuple, torch.Tensor]:

        super().__init__()
        if kernel_type == "gauss":
            self.GoP = gauss_kernel(kernel_size)

        elif kernel_type == "sobel":
            self.GoP = sobel_kernel(kernel_size, return_full=True)

        else:
            raise ValueError("unknown kernel_type!!")

        self.get_ssim_map = get_ssim_map
        self.GoP = self.GoP.view(1, 1, *self.GoP.size())
        self.GoP = self.GoP.repeat(1, 3, 1, 1).to(device)
        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2

    def forward(self, Img1: torch.Tensor, Img2: torch.Tensor) -> torch.Tensor:

        mu_x = F.conv2d(Img1, self.GoP)
        mu_y = F.conv2d(Img2, self.GoP)
        sigma_xx = F.conv2d(Img1 * Img1, self.GoP) - (mu_x.pow(2))
        sigma_yy = F.conv2d(Img2 * Img2, self.GoP) - (mu_y.pow(2))
        sigma_xy = F.conv2d(Img1 * Img2, self.GoP) - (mu_x * mu_y)

        S1_denom = 2 * (mu_x * mu_y) + self.C1
        S1_nom = mu_x.pow(2) + mu_y.pow(2) + self.C1
        S1 = S1_denom / S1_nom

        S2_denom = 2 * sigma_xy + self.C2
        S2_nom = sigma_xx + sigma_yy + self.C2
        S2 = S2_denom / S2_nom

        SSIM_map = (S1 * S2).squeeze()
        ssim_score = SSIM_map.mean()

        if self.get_ssim_map:
            return (ssim_score, SSIM_map)

        return ssim_score
