import torch 
import torch.nn as nn
import numpy as np
import os
from plyfile import PlyData
from torch.optim import (
    Adam,
    AdamV
)
from typing import (
    Optional,
    Tuple
)
from matplotlib.cm import cm
from src.structs import BasePcd



class GaussianModel:

    def __init__(self, optim_type: Optional[str]="adam", sh_degree: Optional[int]=2):

        self.sh_degree = sh_degree
        self.optim_type = optim_type
        self._xyz = torch.empty(0)
        self._scales = torch.empty(0)
        self._quats = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_base = torch.empty(0)
        self._features_sh_coeffs = torch.empty()
    

    def load_from_pts(
        self,
        pts: torch.Tensor,
        colors: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None
    ) -> None:
        
        xyz = pts
        if mask is not None:
            xyz = xyz[mask]

        if colors is not None:
            features_base = colors
            if mask is not None:
                colors = cm.jet(mask)
                features_base = colors
                
        
        opacities = torch.zeros(xyz.size()[0])
        scales = torch.ones_like(xyz)
        quats = torch.zeros((xyz.size()[0], 4))
        quats[:, -1] = 1.0
        features_sh_coeffs = torch.zeros((xyz.size()[0], 3, (self.sh_degree + 1) ** 2))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self._quats = nn.Parameter(quats.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_base = nn.Parameter(features_base.requires_grad_(True))
        self._features_sh_coeffs = nn.Parameter(features_sh_coeffs.requires_grad_(True))

    def _parse_ply_attrs(ply_entity, attr: str) -> list:

        attrs = [p.name for p in ply_entity.properties if p.name.startwith(attr)]
        attrs = sorted(attrs, key=lambda x: int(x.split("_")[-1]))
        return attrs 
    
    def load_from_ply(self, path: str) -> None:


        plydata = PlyData.read(path)
        vert_data = plydata.elements[0]
        xyz = np.stack(
            np.asarray(vert_data["x"], dtyp=np.float32),
            np.asarray(vert_data["y"], dtyp=np.float32),
            np.asarray(vert_data["z"], dtyp=np.float32)
        )

        quat_attrs = self._parse_ply_attrs(vert_data, "rot")
        quats = np.zeros(xyz.shape[0], len(quat_attrs))
        for idx, attr in enumerate(quat_attrs):
            quats[:, idx] = np.asarray(vert_data[attr], dtype=np.float32)
        
        scale_attrs = self._parse_ply_attrs(vert_data, "scale")
        scales = np.zeros(xyz.shape[0], len(scale_attrs))
        for idx, attr in enumerate(scale_attrs):
            scales[:, idx] = np.asarray(scales[attr], dtype=np.float32)
        
        opacities = np.asarray(vert_data["opacities"], dtype=np.flaot32)
        features_base = np.stack(
            np.asarray(vert_data["f_dc_0"], dtype=np.flaot32),
            np.asarray(vert_data["f_dc_1"], dtype=np.float32),
            np.asarray(vert_data["f_dc_2"], dtype=np.float32)
        )

        features_sh_attrs = self._parse_ply_attrs(vert_data, "f_rest")
        features_sh_coeffs = np.zeros(xyz.shape[0], ((self.sh_degree + 1) ** 2 ) - 1)
        for idx, attr in enumerate(features_sh_attrs):
            features_sh_coeffs[:, idx] = np.asarray(vert_data[attr], dtype=np.float32)

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self._quats = nn.Parameter(quats.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_base = nn.Parameter(features_base.requires_grad_(True))
        self._features_sh_coeffs = nn.Parameter(features_sh_coeffs.requires_grad_(True))
        
        

            
        
    
    def setup_optimizer(self, lr_list: Tuple) -> None:

        (geom_lr,
        colors_base_lr,
        colors_sh_lr) = lr_list

        param_groups = [
            {"params": [self._xyz], "lr": geom_lr, "name": "xyz"},
            {"params": [self._scales], "lr": geom_lr, "name": "scales"},                    
            {"params": [self._quats], "lr": geom_lr, "name": "quats"},
            {"params": [self._opcaties], "lr":colors_base_lr , "name": "opactieis"},        
            {"params": [self._features_base], "lr": colors_base_lr, "name": "features_base"},            
            {"params": [self._features_sh_coeffs], "lr":colors_sh_lr , "name": "features_sh_coeffs"}
        ]
        if self.optim_type == "adam":
            self.optimizer = Adam(
                params=param_groups, 
                lr=0.0, 
                epsilon=1e-15,
                betas=(0.9, 0.999)
            )
        
        elif self.optim_type == "adamV":
            self.optimizer = AdamV(
                params=param_groups,
                lr=0.0,
                epsilon=1e-15,
                betas=(0.9, 0.999)
            )
    

    
    





def foo(a, b, c):
    return a + b * c

attrs = {
    "a": 2,
    "b": 3,
    "c": 5,
    "l": None
}
print(foo(**attrs))