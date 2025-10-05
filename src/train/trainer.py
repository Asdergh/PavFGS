import torch
import torch.nn as nn
import numpy as np 
import os

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
    AdamW
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



@dataclass
class Config:
    epochs_n: Optional[int]=7000
    densification_start: Optional[int]=1000
    densification_until: Optional[int]=3000
    densification_interval: Optional[int]=1000
    xyz_lr: Optional[float]=0.01
    geom_lr: Optional[float]=0.01
    colors_lr: Optional[float]=0.1
    densification_strat: Optional[str]="default" # [default, mcmc]
    scene_idx: Optional[int]=0 # idx of scene from BasicPointcloudScene
    sh_degree: Optional[int]=3 # [1, 2, 3]
    split_gs_optimizers: Optional[bool]=True
    optim_type: Optional[str]="adam" # [adam, adamw]
    mse_loss: Optional[bool]=False
    mae_loss: Optional[bool]=False
    dssim_loss: Optional[bool]=False
    mse_coeff: Optional[float]=1.0
    mae_coeff: Optional[float]=1.0
    dssim_loss: Optional[float]=1.0
    save_samples: Optional[bool]=True
    save_ecposhs: Optional[List]=field(default_factory=lambda: [4000, 5000, 6000, 6999])
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
    
        if self.cfg.densification_strat.lower() == "default":
            self.strategy = DefaultStrategy(
                refine_start_iter=self.cfg.densification_start,
                refine_stop_iter=self.cfg.densification_until,
                refine_every=self.cfg.densification_interval
            )
            self.strat_state = self.strategy.initialize_state(scene_scale)
        
        elif self.cfg.densification_strat.lower() == "mcmc":
            self.tratagy = MCMCStrategy(
                refine_start_iter=self.cfg.densification_start,
                refine_stop_iter=self.cfg.densification_until,
                refine_every=self.cfg.densification_interval
            )
            self.strat_state = self.strategy.initialize_state()
        
        else:
            raise ValueError("config: densification_strat can only be: [default, mcmc]")
        
    

    def parse_scene(self) -> Tuple[
        nn.ParameterDict, 
        Union[Dict[str, Optimizer], Optimizer]
    ]:

        scene_items = self.base_scene[self.cfg.scene_idx]
        scene_scale = np.linalg.norm(scene_items["bbox_extent"])
        gt_rgb = scene_items["gt_imgs"].to(self.device)
        viewmats = scene_items["viewmats"].to(self.device)
        Ks = torch.Tensor(self.base_scene.K)[None].repeat(gt_rgb.size()[0], 1, 1).to(self.device)

        pts = torch.Tensor(scene_items["pts"])
        quats = torch.rand(pts.size()[0], 4)
        quats[:, -1] = 1.0
        scales = torch.ones_like(pts)
        opactieis = torch.rand(pts.size()[0], )

        colors = torch.zeros(pts.size()[0], (self.cfg.sh_degree + 1) ** 2, 3)
        colors[:, 0, :] = torch.Tensor(scene_items["colors"])
        colors = colors.to(self.device)

        splats_attrs = [
            ("means", nn.Parameter(pts.to(self.device)), self.cfg.xyz_lr),         
            ("quats", nn.Parameter(quats.to(self.device)), self.cfg.geom_lr),
            ("scales", nn.Parameter(scales.to(self.device)), self.cfg.geom_lr),
            ("opacities", nn.Parameter(opactieis.to(self.device)), self.cfg.colors_lr),
            ("sh0", nn.Parameter(colors[:, :1, :]), self.cfg.colors_lr),
            ("shN", nn.Parameter(colors[:, 1:, :]), self.cfg.colors_lr)      
        ]
        splats = nn.ParameterDict({n: p for (n, p, _) in splats_attrs})
        
        if self.cfg.split_gs_optimizers:
            if self.cfg.optim_type.lower() == "adam":
                opt = {
                    n: Adam([{"params": [p], "lr": l, "name": n}], lr=0.1) 
                    for (n, p, l) in splats_attrs
                }
            elif self.cfg.optim_type.lower() == "adamw":
                opt = {
                    n: AdamW([{"params": [p], "lr": l, "name": n}], lr=0.1) 
                    for (n, p, l) in splats_attrs
                }
            else:
                raise ValueError("config: optim_type can be only: [Adam, AdamW]")
        
        else:
            if self.cfg.optim_type.lower() == "adam":
                opt = Adam([
                    {"params": [p], "lr": p, "name": n}
                    for (n, p, l) in splats_attrs
                ], lr=0.1)

            elif self.cfg.optim_type.lower() == "adamw":
                opt = AdamW([
                    {"params": [p], "lr": p, "name": n}
                    for (n, p, l) in splats_attrs
                ], lr=0.1)

            else:
                raise ValueError("config: optim_type can be only: [Adam, AdamW]")

        

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
                
               
                rendered_rgb, _, meta = rasterization(
                    means=self.splats["means"],
                    quats=self.splats["quats"],
                    scales=self.splats["scales"],
                    opacities=self.splats["opacities"],
                    colors=eval_sh(
                        deg=self.cfg.sh_degree,
                        sh=torch.cat([
                            self.splats["sh0"], 
                            self.splats["shN"]
                        ], dim=1).permute(0, 2, 1),
                        dirs=self.splats["means"]
                    ),
                    width=self.base_scene.w, height=self.base_scene.h,
                    viewmats=self.viewmats,
                    Ks=self.Ks
                )
                # meta["radii"] = meta["radii"].unsqueeze(dim=-1)
                
                rendered_rgb = rendered_rgb.permute(0, -1, 1, 2)
                if self.cfg.split_gs_optimizers:
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
                    for param in self.opt:
                        self.opt[param].step()
                        self.opt[param].zero_grad()

                    # if isinstance(self.strategy, DefaultStrategy):
                    #     print(list(meta.keys()))
                    #     print(meta["radii"].size())
                    #     self.strategy.step_post_backward(
                    #         params=self.splats,
                    #         optimizers=self.opt,
                    #         state=self.strat_state,
                    #         info=meta,
                    #         step=idx
                    #     )
                    
                    # elif isinstance(self.strategy, MCMCStrategy):
                    #     self.strategy.step_post_backward(
                    #         param=self.splats,
                    #         optimizers=self.opt,
                    #         state=self.strat_state,
                    #         info=meta,
                    #         lr=self.cfg.geom_lr,
                    #         step=idx
                    #     )
                
                
                if idx in self.cfg.save_ecposhs:
                    if self.cfg.save_samples:
                        path = os.path.join(self.cfg.log_dir, f"splats")
                        if not os.path.exists(path):
                            os.makedirs(path)
                        path = os.path.join(path, f"Splats{idx}.ply")
                        
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
                
                pbar.update(1)
                pbar.set_description(f"GS Reconstraction Loss: ...[{loss.item()}]...")
               
            


if __name__ == "__main__":

    video_path = "/media/test/T7/test_video3.mp4"
    vggt_weights = "/media/test/T7/model.pt"
    log_dir = "/media/test/T7/ply_collection"

    basic_pts = BasicPointCloudScene()
    basic_pts.create_from_video([video_path], 2.0)
    config = Config(log_dir=log_dir)

    trainer = PavGSTrainer(config, basic_pts, "cuda")
    trainer.train()






    
        
                
        
        