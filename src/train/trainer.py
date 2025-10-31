import torch
import torch.nn as nn
import numpy as np 
import os
import matplotlib.cm as cm
from torch.nn import functional as F
from torchvision.transforms import functional as Fv

from dataclasses import dataclass, field
from gsplat.rendering import rasterization
from gsplat.strategy import (
    DefaultStrategy, 
    MCMCStrategy
)
from gsplat import export_splats
from src.scene.pts_scene import BasicPointCloudScene
from torch.optim import (
    Optimizer,
    Adam,
    AdamW,
    SparseAdam
)
from typing import (
    Optional,
    Tuple,
    Dict,
    Union,
    List
)
from tqdm import tqdm
from torch.nn import (
    MSELoss,
    L1Loss
)
from src.utils import (
    DSSIMLoss,
    eval_sh
)
from torchvision.io import write_video





__optimizers__ = {
    "adam": Adam,
    "adamw": AdamW,
    "sparse_adam": SparseAdam
}
@dataclass
class Config:
    epochs_n: Optional[int]=15000
    densification_start: Optional[int]=1000
    densification_until: Optional[int]=10000
    densification_interval: Optional[int]=100
    prune_opa: Optional[float]=0.005
    grow_scale3d: Optional[float]=0.01
    prune_scale3d: Optional[float]=0.1
    packed: Optional[bool]=True
    xyz_lr: Optional[float]=0.01
    geom_lr: Optional[float]=0.01
    colors_lr: Optional[float]=0.1
    densification_strat: Optional[str]="default" # [default, mcmc]
    scene_idx: Optional[int]=0 # idx of scene from BasicPointcloudScene
    sh_degree: Optional[int]=3 # [1, 2, 3]
    apply_sh: Optional[bool]=False
    split_gs_optimizers: Optional[bool]=True
    optim_type: Optional[str]="adam" # [adam, adamw, sparse_adam]
    mse_loss: Optional[bool]=False
    mae_loss: Optional[bool]=False
    dssim_loss: Optional[bool]=False
    mse_coeff: Optional[float]=1.0
    mae_coeff: Optional[float]=1.0
    dssim_loss: Optional[float]=1.0
    save_samples: Optional[bool]=True
    save_ecposhs: Optional[List]=field(default_factory=lambda: [0, 200, 1000, 4000, 5000, 6000, 700, 9000, 11000, 14999])
    log_dir: Optional[str]=None


class PavGSTrainer:

    def __init__(
        self, 
        cfg: Config, 
        basic_scene: BasicPointCloudScene, 
        device: Optional[str]="cuda"
    ) -> None:


        self.device = device
        self.cfg = cfg
        self.base_scene = basic_scene

        (self.splats, self.opt, 
        self.gt_rgb, self.viewmats, 
        scene_scale, self.Ks) = self.parse_scene()
        self.viewmats[..., :3, -1] = F.sigmoid(self.viewmats[..., :3, -1])
        print(self.viewmats[..., :3, -1].min(), self.viewmats[..., :3, -1].mean(), self.viewmats[..., :3, -1].max())
        print(self.viewmats[..., :3, :3].min(), self.viewmats[..., :3, :3].mean(), self.viewmats[..., :3, :3].max())
        print(self.splats["means"].min(), self.splats["means"].mean(), self.splats["means"].max())
    
        if self.cfg.densification_strat.lower() == "default":
            self.strategy = DefaultStrategy(
                refine_start_iter=self.cfg.densification_start,
                refine_stop_iter=self.cfg.densification_until,
                refine_every=self.cfg.densification_interval,
                grow_scale3d=self.cfg.grow_scale3d,
                prune_scale3d=self.cfg.prune_scale3d,
                prune_opa=self.cfg.prune_opa
            )
            print(scene_scale)
            self.strat_state = self.strategy.initialize_state(scene_scale)
        
        elif self.cfg.densification_strat.lower() == "mcmc":
            self.strategy = MCMCStrategy(
                refine_start_iter=self.cfg.densification_start,
                refine_stop_iter=self.cfg.densification_until,
                refine_every=self.cfg.densification_interval
            )
            self.strat_state = self.strategy.initialize_state()
    
        
    

    def parse_scene(self) -> Tuple[
        nn.ParameterDict, 
        Union[Dict[str, Optimizer], Optimizer]
    ]:

        scene_items = self.base_scene[self.cfg.scene_idx]
        scene_scale = np.linalg.norm(scene_items["bbox_extent"])
        gt_rgb = scene_items["gt_imgs"].to(self.device)
        if self.cfg.save_samples:
            print(gt_rgb.size())
            self.gt_ref_idx = np.random.randint(0, gt_rgb.size()[0])
            self.imgs2save = []
            self.ssim_maps2save = []

        viewmats = scene_items["viewmats"].to(self.device)
        Ks = torch.Tensor(self.base_scene.K)[None].repeat(gt_rgb.size()[0], 1, 1).to(self.device)

        pts = torch.Tensor(scene_items["pts"])
        quats = torch.rand(pts.size()[0], 4)
        quats[:, -1] = 1.0
        scales = torch.Tensor(scene_items["initial_scales"])
        print(scales.max())
        opactieis = torch.rand(pts.size()[0], )

        colors = torch.zeros(pts.size()[0], (self.cfg.sh_degree + 1) ** 2, 3)
        colors[:, 0, :] = torch.Tensor(scene_items["colors"])
        colors = colors.to(self.device)

        splats_attrs = [
            ("means", nn.Parameter(F.sigmoid(pts).to(self.device)), self.cfg.xyz_lr),         
            ("quats", nn.Parameter(quats.to(self.device)), self.cfg.geom_lr),
            ("scales", nn.Parameter(F.sigmoid(scales).to(self.device)), self.cfg.geom_lr),
            ("opacities", nn.Parameter(F.sigmoid(opactieis).to(self.device)), self.cfg.colors_lr),
            ("sh0", nn.Parameter(colors[:, :1, :]), self.cfg.colors_lr),
            ("shN", nn.Parameter(colors[:, 1:, :]), self.cfg.colors_lr)      
        ]
        splats = nn.ParameterDict({n: p for (n, p, _) in splats_attrs})
        
        if self.cfg.optim_type.lower() in __optimizers__:
            opt = __optimizers__[self.cfg.optim_type.lower()]
        
        else:
            raise ValueError("config: optim_type can be only: [Adam, AdamW, Sparse_Adam]")
        
        if self.cfg.split_gs_optimizers:
            opt = {n: opt([p], lr=l) for (n, p, l) in splats_attrs}

        else:
            opt = opt([
                {"params": [p], "lr": p, "name": n}
                for (n, p, l) in splats_attrs
            ], lr=0.1)
        
        return (splats, opt, gt_rgb, viewmats, scene_scale, Ks)
    

    def train(self) -> None:
        
        if (not self.cfg.mse_loss and 
            not self.cfg.mae_loss and
            not self.cfg.dssim_loss):
            raise ValueError(f"""
                you must choose atleast one loss. 
                Curent loss states: 
                    [mse_loss: {self.cfg.mse_loss},
                    mae_loss: {self.cfg.mae_loss},
                    dssim_loss: {self.cfg.dssim_loss}]""")

        loss_fns = []
        if self.cfg.mse_loss:
            mse_loss = MSELoss()
            loss_fns.append((mse_loss, self.cfg.mse_coeff))
        
        if self.cfg.mae_loss:
            mae_loss = L1Loss()
            loss_fns.append((mae_loss, self.cfg.mae_coeff))
        
        if self.cfg.dssim_loss:
            dssim_loss = DSSIMLoss()
            loss_fns.append((dssim_loss, self.cfg.dssim_loss))
            
        
        with tqdm(
            desc="Scene Reconstraction...",
            colour="red",
            ascii=":>",
            total=self.cfg.epochs_n
        ) as pbar:
            for idx in range(self.cfg.epochs_n):
                
            
                if self.cfg.apply_sh:
                   colors = eval_sh(
                        deg=self.cfg.sh_degree,
                        sh=torch.cat([
                            self.splats["sh0"], 
                            self.splats["shN"]
                        ], dim=1).permute(0, 2, 1),
                        dirs=self.splats["means"]
                    )
                
                else:
                    colors = self.splats["sh0"].squeeze()

                rendered_rgb, _, meta = rasterization(
                    means=self.splats["means"],
                    quats=self.splats["quats"],
                    scales=self.splats["scales"],
                    opacities=self.splats["opacities"],
                    colors=colors,
                    width=self.base_scene.w, height=self.base_scene.h,
                    viewmats=self.viewmats,
                    Ks=self.Ks,
                    absgrad=(
                        self.strategy.absgrad
                        if isinstance(self.strategy, DefaultStrategy)
                        else None
                    ),
                    packed=self.cfg.packed
                )

                if self.cfg.save_samples:
                    ref_render = rendered_rgb[self.gt_ref_idx, ...].permute(-1, 0, 1)
                    ref_gt = self.gt_rgb[self.gt_ref_idx, ...]
                    if self.cfg.dssim_loss:
                        _, ssim_heatmap = dssim_loss(
                            ref_gt, ref_render, 
                            get_ssim_map=True
                        )
                        ssim_heatmap = Fv.resize(
                            ssim_heatmap.view(1, *ssim_heatmap.size()), 
                            (self.base_scene.w, self.base_scene.h)
                        )
                        self.ssim_maps2save.append(ssim_heatmap)
                    self.imgs2save.append(ref_render)
                    
                rendered_rgb = rendered_rgb.squeeze().permute(0, -1, 1, 2)
                if self.cfg.split_gs_optimizers:
                    if self.cfg.densification_strat.lower() != "none":
                        self.strategy.step_pre_backward(
                            params=self.splats,
                            optimizers=self.opt,
                            state=self.strat_state,
                            step=idx,
                            info=meta
                        )

                loss = 0.0
                for (loss_fn, coeff) in loss_fns:
                    loss += coeff * loss_fn(self.gt_rgb, rendered_rgb)

                loss.backward()
                if self.cfg.split_gs_optimizers:
                                    
                    if isinstance(self.strategy, DefaultStrategy):
                        gaussian_ids = meta["gaussian_ids"]
                        for k in self.splats.keys():
                            grad = self.splats[k].grad
                            if grad is None or grad.is_sparse:
                                continue
                            self.splats[k].grad = torch.sparse_coo_tensor(
                                indices=gaussian_ids[None],  # [1, nnz]
                                values=grad[gaussian_ids],  # [nnz, ...]
                                size=self.splats[k].size(),  # [N, ...]
                                is_coalesced=len(self.Ks) == 1,
                            )
                    
                    for param in self.opt:
                        self.opt[param].step()
                        self.opt[param].zero_grad()

                    if idx != 0:
                        if self.cfg.densification_strat.lower() != "none":
                            if isinstance(self.strategy, DefaultStrategy):
                                self.strategy.step_post_backward(
                                    params=self.splats,
                                    optimizers=self.opt,
                                    state=self.strat_state,
                                    info=meta,
                                    step=idx,
                                    packed=self.cfg.packed
                                )
                                if (idx > self.cfg.densification_start and 
                                    idx < self.cfg.densification_until and 
                                    idx % self.cfg.densification_interval == 0):

                                    print((32 * "=") + "... Denficiation results ..." + (32 * "="))
                                    print(f"means: [{self.splats['means'].size()}, {self.splats['means'].min()}, {self.splats['means'].max()}]")
                                    print(f"opacities: [{self.splats['opacities'].size()}, {self.splats['opacities'].min()}, {self.splats['opacities'].max()}]")
                                    print(f"scales: [{self.splats['scales'].size()}, {self.splats['scales'].min()}, {self.splats['scales'].max()}]")
                                    print((32 * "=") + "... Denficiation results ..." + (32 * "="), "\n")

                            elif isinstance(self.strategy, MCMCStrategy):
                                self.strategy.step_post_backward(
                                    params=self.splats,
                                    optimizers=self.opt,
                                    state=self.strat_state,
                                    info=meta,
                                    lr=self.cfg.geom_lr,
                                    step=idx
                                )
                
                
                if idx in self.cfg.save_ecposhs:
                    if self.cfg.save_samples:
                        splats_dir = os.path.join(self.cfg.log_dir, "splats")
                        video_f = os.path.join(self.cfg.log_dir, "render_results.mp4")
                        if not os.path.exists(splats_dir):
                            os.makedirs(splats_dir)
                        path = os.path.join(splats_dir, f"Splats{idx}.ply")
                        
                        export_splats(
                            means=self.splats["means"],
                            scales=self.splats["scales"],
                            quats=self.splats["quats"],
                            opacities=self.splats["opacities"],
                            sh0=self.splats["sh0"],
                            shN=self.splats["shN"],
                            format="ply_compressed",
                            save_to=path
                        )
                        
                        rgb2save = (torch.stack(self.imgs2save, dim=0) * 255.0).cpu()
                        if self.cfg.dssim_loss:
                            map2save = torch.stack(self.ssim_maps2save, dim=0).detach().cpu().numpy()
                            colorized_maps = torch.Tensor(cm.jet(map2save))
                            print(colorized_maps.size())
                            colorized_maps = 255 * (colorized_maps.squeeze(dim=0).permute(0, -1, 1, 2)[:, :3, ...])
                            # print(colorized_maps.min(), colorized_maps.mean(), colorized_maps.max())
                        
                        samples2save = torch.cat([rgb2save, colorized_maps], axis=-1)
                        print(samples2save.size())    
                        write_video(video_f, samples2save.permute(0, 2, 3, 1), fps=1.0)
                        
                
                pbar.update(1)
                pbar.set_description(f"GS Reconstraction Loss: ...[{loss.item()}]...")
               
            


if __name__ == "__main__":

    video_path = "/media/test/T7/test_video3.mp4"
    vggt_weights = "/media/test/T7/model.pt"
    log_dir = "/media/test/T7/ply_collection"

    basic_pts = BasicPointCloudScene()
    basic_pts.create_from_colmap(
        path="/media/test/T7/ply_collection/gerrard-hall",
        partition_size=10,
        partitions_n=40,
        shuffle=True,
        max_radii=28.5,
        inv_poses=True
    )
    config = Config(
        log_dir=log_dir,
        densification_strat="default",
        apply_sh=True,
        optim_type="sparse_adam",
        prune_opa=0.005,
        grow_scale3d=0.1,
        prune_scale3d=0.9,
        xyz_lr=0.001,
        geom_lr=0.001,
        colors_lr=0.001
    )

    trainer = PavGSTrainer(config, basic_pts, "cuda")
    trainer.train()






    
        
                
        
        