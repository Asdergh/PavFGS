import torch 
import numpy as np
import open3d 
import os
import rerun as rr

from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Tuple
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
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors


class BasicPointCloudScene:

    def __init__(
        self, 
        width: Optional[int]=112, 
        height: Optional[int]=112,
        vggt_weights: Optional[str]=None,
        base_rotation: Optional[np.ndarray]=None,
        base_translation: Optional[np.ndarray]=None,
        base_K: Optional[np.ndarray]=np.array([
            [56.0, 0.0, 56.0],
            [0.0, 56.0, 56.0],
            [0.0, 0.0, 1.0]
        ]),
        n_neighbors: Optional[int]=3
    ) -> None:

        self.w, self.h = (width, height)
        self.K = base_K
        self.pts = torch.empty(0)
        self.normals = torch.empty(0)
        self.colors = torch.empty(0)     

        self._vggt = VGGT()
        if vggt_weights is not None:
            weights = torch.load(vggt_weights, weights_only=True)
            self._vggt.load_state_dict(weights)
        
        self.base_Rmat = base_rotation
        self.base_t = base_translation
        self.nn_searcher = NearestNeighbors(n_neighbors=n_neighbors)
        
    

    def _vggt2pcd(self, pts, colors) -> Tuple[np.ndarray, np.ndarray]:

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
    
    def _handle_pose_encs(self, t, quats) -> torch.Tensor:
        
        viewmats = torch.zeros(t.size()[0], 4, 4)
        for idx in range(t.size()[0]):

            tmp_t = t[idx, :]
            quat = quats[idx, :].numpy()
            Rmat = torch.Tensor(R.from_quat(quat).as_matrix())
            viewmats[idx, :-1, -1] = tmp_t
            viewmats[idx, :3, :3] = Rmat
        
        return viewmats
            
    def _handle_frames_strack(
        self, 
        pts: torch.Tensor, 
        inputs: torch.Tensor, 
        search_param: Union[KDTreeSearchParamHybrid, KDTreeSearchParamKNN] ,
        batch_idx: Optional[int]=0,
    ) -> PointCloud:

        points_ = []
        colors_ = []
        for frame_idx in range(inputs.size()[1]):

            points = pts[batch_idx, frame_idx, ...]
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

        return pcd


    def create_from_tensor(
        self, 
        inputs: torch.Tensor,
        search_param: Optional[str | KDTreeSearchParamKNN | KDTreeSearchParamHybrid]="knn",
        k_nns: Optional[int]=30,
        radius: Optional[int]=0.1
    ) -> None:

        self.pcds = []
        self.gt_imgs = []
        self.viewmats = []
        inputs = Fv.resize(inputs, (self.w, self.h))
        if isinstance(search_param, str):
            if search_param == "knn":
                    search_param = KDTreeSearchParamKNN(knn=k_nns)
                
            elif search_param == "hybrid":
                search_param = KDTreeSearchParamHybrid(radius, k_nns)

        if len(inputs.size()) == 5:
            with torch.no_grad():
                vggt_item_ = self._vggt(inputs)
            for batch_idx in range(inputs.size()[0]):
                pcd = self._handle_frames_strack(
                    inputs=inputs,
                    pts=vggt_item_["world_points"],
                    search_param=search_param,
                    batch_idx=batch_idx
                )

                self.pcds.append(pcd)
                self.gt_imgs.append(inputs[batch_idx, ...])

                t, quats = vggt_item_["pose_enc"][batch_idx, :, :-2].split([3, 4], dim=-1)
                viewmats = self._handle_pose_encs(t, quats)
                self.viewmats.append(viewmats)

        else:
            vggt_item_ = self._vggt(inputs.unsqueeze(dim=0))
            pcd = self._handle_frames_strack(
                inputs=inputs,
                pts=vggt_item_["world_points"],
                search_param=search_param
            )
            self.pcds = [pcd]
            self.gt_imsg = [inputs.squeeze()]
            
            t, quats = vggt_item_["pose_enc"][0, :, :-2].split([3, 4], dim=-1)
            viewmats = self._handle_pose_encs(t, quats)
            self.viewmats = [viewmats]

            
   
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
                    half_sizes=[scene_items["bbox_extent"] / 2],
                    quaternions=[rr.Quaternion(xyzw=scene_items["bbox_rotation"])],
                    colors=[(0, 255, 0)],
                    labels=f"Scen{idx}"
                )
            )
    
    def __len__(self) -> int:
        return len(self.pcds)
        
    def __getitem__(self, idx: int) -> dict:

        if idx > len(self):
            pcd = self.pcd[len(self) - 1]
            gt_imgs = self.gt_imgs[len(self) - 1]
            viewmats = self.viewmats[len(self) - 1]
        
        else:
            pcd = self.pcds[idx]
            gt_imgs = self.gt_imgs[idx]
            viewmats = self.viewmats[idx]

        bbox = pcd.get_oriented_bounding_box()

        pts = np.asarray(pcd.points)
        self.nn_searcher.fit(pts)
        dists, _ = self.nn_searcher.kneighbors(pts)

        initial_scales = np.stack([
            dists.min(axis=-1),
            dists.min(axis=-1),
            dists.min(axis=-1)
        ], axis=-1)
        


        return {
            "pts": pts,
            "initial_scales": initial_scales,
            "colors": np.asarray(pcd.colors),
            "normals": np.asarray(pcd.normals),
            "bbox_center": np.asarray(bbox.center),
            "bbox_extent": np.asarray(bbox.extent),
            "bbox_rotation": R.from_matrix(bbox.R).as_quat(),
            "gt_imgs": gt_imgs,
            "viewmats": viewmats
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]




if __name__ == "__main__":

    
    rot_vec = np.array([1, 0.0, 0.0]) * 90.0
    Rmat = R.from_rotvec(rot_vec, degrees=True).as_matrix()
    # Rmat = None
    # video1 = "/media/test/T7/video_test.mp4"
    # video2 = "/media/test/T7/test_video3.mp4"
    video3 = "/media/test/T7/sber_indoor.mp4"
    weights = "/media/test/T7/model.pt"
    
    pcd = BasicPointCloudScene(vggt_weights=weights, base_rotation=Rmat)
    pcd.create_from_video([video3], 5.0)
    # draw_geometries([pcd[0], pcd[1], pcd[2]])
    pcd.save_ply("/media/test/T7/ply_collection")
    pcd.show()

    
    
            
        

        
            
        

        
        
        
            
            
        
        