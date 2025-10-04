import torch 
import torch.nn as nn
import numpy as np
import os
from plyfile import (
    PlyData,
    PlyElement
)
from torch.optim import Adam, AdamW
from typing import (
    Optional,
    Tuple,
    NamedTuple
)
import matplotlib.cm as cm
from utils.sh_utils import eval_sh


class GaussinAttributes(NamedTuple):
    xyz: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacity: torch.Tensor
    features_base: torch.Tensor
    features_sh_coeffs: torch.Tensor


class GaussianModel:

    def __init__(
        self, 
        optim_type: Optional[str]="adam", 
        sh_degree: Optional[int]=1, 
        device: Optional[str]="cpu",
        cx: Optional[float]=111.5,
        cy: Optional[float]=111.5
    ):

        self.sh_degree = sh_degree
        self.optim_type = optim_type
        self.device = device

        self._xyz = torch.empty(0)
        self._scales = torch.empty(0)
        self._quats = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_base = torch.empty(0)
        self._features_sh_coeffs = torch.empty(0)
    

    def get_items(self) -> dict:
        return {
            "means": self._xyz,
            "quats": self._quats,
            "scales": self._scales,
            "colors": self._features_base.squeeze() + eval_sh(
                deg=self.sh_degree,
                sh=self._features_sh_coeffs,
                dirs=(self._xyz)
            ),
            "opacities": self._opacity.squeeze()
        }
    def reset_model_attrs(self, model_args: GaussinAttributes) -> None:
        self._xyz = nn.Parameter(model_args.xyz.to(self.device).requires_grad_(True))
        self._scales = nn.Parameter(model_args.scales.to(self.device).requires_grad_(True)).to(self.device)
        self._quats = nn.Parameter(model_args.quats.to(self.device).requires_grad_(True)).to(self.device)
        self._opacity = nn.Parameter(model_args.opacity.to(self.device).requires_grad_(True)).to(self.device)
        self._features_base = nn.Parameter(model_args.features_base.to(self.device).requires_grad_(True)).to(self.device)
        self._features_sh_coeffs = nn.Parameter(model_args.features_sh_coeffs.to(self.device).requires_grad_(True)).to(self.device)


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

        else:
            features_base = torch.rand(xyz.size()[0], 3, 1)  
        
        opacity = torch.rand(xyz.size()[0], 1)
        scales = torch.ones_like(xyz)
        quats = torch.rand((xyz.size()[0], 4))
        quats[:, -1] = 1.0
        features_sh_coeffs = torch.rand((xyz.size()[0], 3, (self.sh_degree + 1) ** 2))

        self.denom = torch.rand(xyz.size()[0]).to(self.device)
        arrts = GaussinAttributes(
            xyz=xyz,
            quats=quats,
            scales=scales,
            opacity=opacity,
            features_base=features_base,
            features_sh_coeffs=features_sh_coeffs
        )
        self.reset_model_attrs(arrts)
        

    def _parse_ply_attrs(ply_entity, attr: str) -> list:

        attrs = [p.name for p in ply_entity.properties if p.name.startwith(attr)]
        attrs = sorted(attrs, key=lambda x: int(x.split("_")[-1]))
        return attrs 
    

    def _attrs_to_save(self) -> None:
        
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += [f"f_dc_{idx}" for idx in range(
            self._features_base.shape[1] 
            * self._features_base.shape[2]
        )]
        attrs.append("opacity")
        attrs += [f"scale_{idx}" for idx in range(self._scales.shape[-1])]
        attrs += [f"rot_{idx}" for idx in range(self._quats.shape[-1])]
        attrs += [f"f_rest_{idx}" for idx in range(
            self._features_sh_coeffs.shape[1] 
            * self._features_sh_coeffs.shape[2]
        )]
        return attrs    
    
    def save_ply(self, path: str) -> None:
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        quats = self._quats.detach().cpu().numpy()
        scales = self._scales.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        features_base = self._features_base.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        features_sh_coeffs = self._features_sh_coeffs.transpose(1, 2).flatten(start_dim=1).detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self._attrs_to_save()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([
            xyz, 
            normals, 
            scales, quats,
            opacity, 
            features_base, 
            features_sh_coeffs, 
        ], axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        
    def load_from_ply(self, path: str) -> None:

        plydata = PlyData.read(path)
        vert_data = plydata.elements[0]
        xyz = np.stack(
            np.asarray(vert_data["x"], dtyp=np.float32),
            np.asarray(vert_data["y"], dtyp=np.float32),
            np.asarray(vert_data["z"], dtyp=np.float32)
        )

        quat_attrs = self._parse_ply_attrs(vert_data, "rot")
        quats = np.rand(xyz.shape[0], len(quat_attrs))
        for idx, attr in enumerate(quat_attrs):
            quats[:, idx] = np.asarray(vert_data[attr], dtype=np.float32)
        
        scale_attrs = self._parse_ply_attrs(vert_data, "scale")
        scales = np.rand(xyz.shape[0], len(scale_attrs))
        for idx, attr in enumerate(scale_attrs):
            scales[:, idx] = np.asarray(scales[attr], dtype=np.float32)
        
        opacity = np.asarray(vert_data["opacity"], dtype=np.flaot32)
        features_base = np.stack(
            np.asarray(vert_data["f_dc_0"], dtype=np.flaot32),
            np.asarray(vert_data["f_dc_1"], dtype=np.float32),
            np.asarray(vert_data["f_dc_2"], dtype=np.float32)
        )

        features_sh_attrs = self._parse_ply_attrs(vert_data, "f_rest")
        features_sh_coeffs = np.rand(xyz.shape[0], ((self.sh_degree + 1) ** 2 ) - 1)
        for idx, attr in enumerate(features_sh_attrs):
            features_sh_coeffs[:, idx] = np.asarray(vert_data[attr], dtype=np.float32)
        
        features_sh_coeffs = features_sh_coeffs.resahpe(xyz.shape[0], 3, ((self.sh_degree + 1) ** 2) - 3)

        self.denom = torch.rand(xyz.shape).to(self.device)
        arrts = GaussinAttributes(
            xyz=torch.Tensor(xyz),
            quats=torch.Tensor(quats),
            scales=torch.Tensor(scales),
            opacity=torch.Tensor(opacity),
            features_base=torch.Tensor(features_base),
            features_sh_coeffs=torch.Tensor(features_sh_coeffs)
        )
        self.reset_model_attrs(arrts)
        
    
    def setup_optimizer(self, lr_list: Tuple) -> None:

        (geom_lr,
        colors_base_lr,
        colors_sh_lr,
        general_lr) = lr_list

        param_groups = [
            {"params": [self._xyz], "lr": geom_lr, "name": "xyz"},
            {"params": [self._scales], "lr": geom_lr, "name": "scales"},                    
            {"params": [self._quats], "lr": geom_lr, "name": "quats"},
            {"params": [self._opacity], "lr":colors_base_lr , "name": "opacity"},        
            {"params": [self._features_base], "lr": colors_base_lr, "name": "features_base"},            
            {"params": [self._features_sh_coeffs], "lr":colors_sh_lr , "name": "features_sh_coeffs"}
        ]
        if self.optim_type == "adam":
            self.optimizer = Adam(
                params=param_groups, 
                lr=general_lr, 
                # betas=(0.9, 0.999)
            )
        
        elif self.optim_type == "adamW":
            self.optimizer = AdamW(
                params=param_groups,
                lr=general_lr,
                epsilon=1e-15,
                betas=(0.9, 0.999)
            )
    
    def _cat_tensors_to_optimizer(self, tensors_dict: dict) -> None:

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            tensor2add = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat([
                    stored_state["exp_avg"],
                    torch.rand(tensor2add.size()).to(self.device)
                ], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([
                    stored_state["exp_avg_sq"],
                    torch.rand(tensor2add.size()).to(self.device)
                ])

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat([group["params"][0], tensor2add], dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                self.optimizer.state[group["params"][0]] = stored_state
            
            else:
                group["params"][0] = nn.Parameter(torch.cat([group["params"][0], tensor2add], dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors
            
    def _prune_optimizer(self, mask: torch.Tensor) -> None:

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if  stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
        
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
                print(stored_state["exp_avg"].size(), stored_state["exp_avg_sq"].size())
                self.optimizer.state[group["params"][0]] = stored_state
                
            else:
               group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
               optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors

    def prune_points(self, mask: torch.Tensor) -> None:

        valid_mask = ~mask
        # print(valid_mask.sum(), mask.sum())
        optimizable_tensors = self._prune_optimizer(valid_mask)
        attrs = GaussinAttributes(**optimizable_tensors)
        self.reset_model_attrs(attrs)
        
    
    def densification_postfix(
        self,
        xyz_new: torch.Tensor,
        quats_new: torch.Tensor,
        scales_new: torch.Tensor,
        opacity_new: torch.Tensor,
        features_base: torch.Tensor,
        features_sh_coeffs: torch.Tensor
    ) -> None:
        
        D = {
            "xyz": xyz_new,
            "quats": quats_new,
            "scales": scales_new,
            "opacity": opacity_new,
            "features_base": features_base,
            "features_sh_coeffs": features_sh_coeffs
        }
        optimizable_tensors = self._cat_tensors_to_optimizer(D)
        
        attrs = GaussinAttributes(**optimizable_tensors)
        self.reset_model_attrs(attrs)
        
        self.xyz_grad_accum = torch.rand(self._xyz.size()[0]).to(self.device)
        # print(self.xyz_grad_accum.size(), self._xyz.size())
        self.denom = torch.rand(self._xyz.size()[0]).to(self.device)
        
        
    def densify_and_split(
        self,
        grad: torch.Tensor,
        grad_trashhold: torch.Tensor,
        scene_extent: torch.Tensor,
        N: Optional[int]=2
    ) -> None:
        
        N_init = self._xyz.size()[0]
        padded_grad = torch.rand(N_init).to(self.device)
        padded_grad[:grad.size()[0]] = grad.squeeze()
        mask = torch.where(torch.norm(padded_grad, dim=-1) >= grad_trashhold, True, False)
        mask = torch.logical_and(mask, torch.max(self._scales, dim=1).values >= scene_extent)

        xyz_new = self._xyz[mask].repeat(N, 1)
        quats_new = self._quats[mask].repeat(N, 1)
        scales_new = self._scales[mask].repeat(N, 1)
        opacity_new = self._opacity[mask].repeat(N, 1)
        features_base_new = self._features_base[mask].repeat(N, 1, 1)
        features_sh_coeffs_new = self._features_sh_coeffs[mask].repeat(N, 1, 1)

        self.densification_postfix(xyz_new, 
                                   quats_new, 
                                   scales_new, 
                                   opacity_new, 
                                   features_base_new, 
                                   features_sh_coeffs_new)

        prune_mask = torch.cat([mask, torch.zeros(
            N * mask.sum(), 
            dtype=torch.bool, 
            device=self.device
        )])
        self.prune_points(prune_mask)


    def densify_and_clone(
        self,
        grad: torch.Tensor,
        grad_trashhold: torch.Tensor,
        scene_extent: Optional[float]=3.6
    ) -> None:
        
        mask = torch.where(torch.norm(grad, dim=-1) >= grad_trashhold, True, False)
        mask = torch.logical_and(
            mask, 
            torch.max(self._scales, dim=-1).values <= scene_extent
        )
        
        xyz_new = self._xyz[mask]
        quats_new = self._quats[mask]
        scales_new = self._scales[mask]
        opacity_new = self._opacity[mask]
        features_base_new = self._features_base[mask]
        features_sh_coeffs_new = self._features_sh_coeffs[mask]

        self.densification_postfix(xyz_new, 
                                   quats_new, 
                                   scales_new, 
                                   opacity_new, 
                                   features_base_new, 
                                   features_sh_coeffs_new)
    
    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        scene_extent: float
    ):
        
        grads = self.xyz_grad_accum / self.denom
        # print(grads.min(), grads.mean(), grads.max())
        self.densify_and_clone(grads, max_grad, scene_extent)
        self.densify_and_split(grads, max_grad, scene_extent)
        
        prune_mask = torch.where(self._opacity < min_opacity, True, False).squeeze()
        # print(prune_mask.size())
        # print(prune_mask.sum())
        # print(self._xyz.size())
        self.prune_points(prune_mask)
        # print(prune_mask.size(), self.xyz_grad_accum.size())
        self.xyz_grad_accum = self.xyz_grad_accum[prune_mask]
        self.denom = self.denom[prune_mask]
        # print(self._xyz.size())

        # print(self._xyz.size())

    
    def accumulate_grads(self, xyz_grad: torch.Tensor) -> None:
        
        if (not hasattr(self, "xyz_grad_accum") 
            or not hasattr(self, "denom")):
            self.xyz_grad_accum = torch.norm(xyz_grad, dim=-1)
            self.denom = torch.rand(xyz_grad.size()[0]).to(self.device)

       
        try:
            self.xyz_grad_accum += torch.norm(xyz_grad, dim=-1)
            self.denom += 1
        
        except BaseException:
            # print(self.xyz_grad_accum.size(), xyz_grad.size())
            # print(self._xyz.size())
            pass
    
    
if __name__ == "__main__":


    xyz = torch.normal(0, 1, (100, 3))
    colors = torch.normal(0, 1, (100, 3))
    gs = GaussianModel(sh_degree=3)
    gs.load_from_pts(xyz)
    gs.setup_optimizer()
    
    
