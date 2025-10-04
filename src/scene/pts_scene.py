import torch 
import numpy as np
import open3d 
import os
import rerun as rr

from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import (
    Union,
    Optional,
    List
)
from open3d.geometry import (
    PointCloud,
    KDTreeSearchParamKNN,
    KDTreeSearchParamHybrid
)
from open3d.utility import Vector3dVector as vec
from torchvision.transforms import functional as Fv
from torchvision.io import read_video
from src.submodules import VGGT
from open3d.io import write_point_cloud


class BasicPCD:

    def __init__(
        self, 
        width: Optional[int]=112, 
        height: Optional[int]=112,
        vggt_weights: Optional[str]=None,
        base_rotation: Optional[np.ndarray]=None,
        base_translation: Optional[np.ndarray]=None,
    ) -> None:

        self.w, self.h = (width, height)
        self.pts = torch.empty(0)
        self.normals = torch.empty(0)
        self.colors = torch.empty(0)     

        self._vggt = VGGT()
        if vggt_weights is not None:
            weights = torch.load(vggt_weights, weights_only=True)
            self._vggt.load_state_dict(weights)
        
        self.base_Rmat = base_rotation
        self.base_t = base_translation
        
    

    def _vggt2pcd(self, pts, colors) -> PointCloud:

        points = torch.flatten(pts, end_dim=-2)
        colors = torch.flatten(colors.permute(1, 2, 0), end_dim=-2)
        
        prune_mask = torch.where(torch.norm(points, dim=-1) == 0, True, False)
    
        prune_points = points[~prune_mask].detach().numpy()
        prune_colors = colors[~prune_mask].numpy()
        del points, colors

        if self.base_Rmat is not None:
            prune_points = (prune_points @ self.base_Rmat)
        
        if self.base_t is not None:
            prune_points[..., 0] += self.base_t[0]
            prune_points[..., 1] += self.base_t[1]
            prune_points[..., 2] += self.base_t[2]

        return (prune_points, prune_colors)
    
    def create_from_tensor(
        self, 
        inputs: torch.Tensor,
        search_param: Optional[str | KDTreeSearchParamKNN | KDTreeSearchParamHybrid]="knn",
        k_nns: Optional[int]=30,
        radius: Optional[int]=0.1
    ) -> None:

        self.pcds = []
        inputs = Fv.resize(inputs, (self.w, self.h))
        if isinstance(search_param, str):
            if search_param == "knn":
                    search_param = KDTreeSearchParamKNN(knn=k_nns)
                
            elif search_param == "hybrid":
                search_param = KDTreeSearchParamHybrid(radius, k_nns)

        if len(inputs.size()) == 5:
            vggt_item_ = self._vggt(inputs)
            for batch_idx in range(inputs.size()[0]):
                points_ = []
                colors_ = []
                for frame_idx in range(inputs.size()[1]):

                    points = vggt_item_["world_points"][batch_idx, frame_idx, ...]
                    colors = inputs[batch_idx, frame_idx, ...]
                    
                    prune_points, prune_colors = self._vggt2pcd(points, colors)
                    points_.append(prune_points)
                    colors_.append(prune_colors)
                
                points_ = np.vstack(points_)
                colors_ = np.vstack(colors_)
                pcd = PointCloud()
                pcd.points = vec(points_)
                pcd.colors = vec(colors_)
                pcd.estimate_normals(search_param)

                self.pcds.append(pcd)
        else:
            vggt_item_ = self._vggt(inputs.unsqueeze(dim=0))
            points_ = []
            colors_ = []
            for frame_idx in range(inputs.size()[1]):
                points = vggt_item_["world_points"][frame_idx, ...]
                colors = inputs[frame_idx, ...]
                prune_points, prune_colors = self._vggt2pcd(points, colors)
                points_.append(prune_points)
                colors_.append(prune_colors)
            
            points_ = np.vstack(points_)
            colors_ = np.vstack(colors_)
            pcd = PointCloud()
            pcd.points = vec(points_)
            pcd.colors = vec(colors_)
            pcd.estimate_normals(search_param)
            self.pcds = [pcd]

            
   
    def create_from_video(
        self, 
        paths: List[str], 
        fps: Optional[float]=20.0, 
        search_param: Optional[str]="knn",
        k_nns: Optional[int]=30,
        radius: Optional[int]=0.1
    ) -> None:


        if search_param == "knn":
                search_param = KDTreeSearchParamKNN(knn=k_nns)
            
        elif search_param == "hybrid":
            search_param = KDTreeSearchParamHybrid(radius, k_nns)

        clips = [VideoFileClip(path) for path in paths]
        frames_n = int(max([clip.duration for clip in clips]) * fps)
        batches_n = len(paths)
        frames_ = torch.zeros(batches_n, frames_n, 3, self.w, self.h)
        for clip_idx, clip in enumerate(clips):
            # print(clip)
            # video_tensor, _, _ = read_video(clip)
            # print(video_tensor.size())
            frames = clip.iter_frames(fps=fps)
            for frame_idx, frame in enumerate(frames):

                frame = torch.Tensor(frame).permute(-1, 0, 1)
                frame = (frame / 255.0).to(torch.float32)
                frame = Fv.resize(frame, (self.w, self.h))
                frames_[clip_idx, frame_idx, ...] = frame
        
        print(frames_.size())
        self.create_from_tensor(frames_, search_param, k_nns, radius)
            
    
    def save_ply(self, path: str) -> None:
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        for idx, pcd in enumerate(self.pcds):
            pcd_f = os.path.join(path, f"scene_{idx}.ply")
            write_point_cloud(pcd_f, pcd)
            
            
            
    def show(self) -> None:
        
        path = "pcd_origin"
        rr.init(path, spawn=True)
        for idx in range(len(self)):
        
            pcd_path = f"{path}/Scene{idx}"
            scene_items = self[idx]
            # print(scene_items["colors"].min(), scene_items["colors"].max())
            rr.log(
                f"{pcd_path}/rgb_pts",
                rr.Points3D(
                    positions=scene_items["pts"],
                    colors=(scene_items["colors"] + 1) / 2,
                    radii=[0.002]
                )
            )
            rr.log(
                f"{pcd_path}/normals_pts",
                rr.Points3D(
                    positions=scene_items["pts"],
                    colors=(scene_items["normals"] * 2) - 1,
                    radii=[0.002]
                )
            )
            rr.log(
                f"{pcd_path}/bbox",
                rr.Boxes3D(
                    centers=[scene_items["bbox_center"]],
                    half_sizes=[scene_items["bbox_estent"] / 2],
                    quaternions=[rr.Quaternion(xyzw=scene_items["bbox_rotation"])],
                    colors=[(0, 255, 0)],
                    labels=f"Scen{idx}"
                )
            )
    
    def __len__(self) -> int:
        return len(self.pcds)
        
    def __getitem__(self, idx: int) -> dict:

        pcd = self.pcds[idx]
        bbox = pcd.get_oriented_bounding_box()
        print(np.asarray(pcd.get_axis_aligned_bounding_box()))
        return {
            "pts": np.asarray(pcd.points),
            "colors": np.asarray(pcd.colors),
            "normals": np.asarray(pcd.normals),
            "bbox_center": np.asarray(bbox.center),
            "bbox_estent": np.asarray(bbox.extent),
            "bbox_rotation": R.from_matrix(bbox.R).as_quat()
        }



if __name__ == "__main__":

    from scipy.spatial.transform import Rotation as R
    rot_vec = np.array([1, 0.0, 0.0]) * 90.0
    Rmat = R.from_rotvec(rot_vec, degrees=True).as_matrix()
    # Rmat = None
    # video = "/media/test/T7/video_test.mp4"
    video = "/media/test/T7/test_video3.mp4"
    weights = "/media/test/T7/model.pt"
    
    pcd = BasicPCD(vggt_weights=weights, base_rotation=Rmat)
    pcd.create_from_video([video], 2.0)
    # draw_geometries([pcd[0], pcd[1], pcd[2]])
    pcd.save_ply("/media/test/T7/ply_collection")
    pcd.show()

    
    
            
        

        
            
        

        
        
        
            
            
        
        