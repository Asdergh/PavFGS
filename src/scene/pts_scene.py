import torch 
import numpy as np
import open3d 
import os
import rerun as rr
import pandas as pd

from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Tuple
)
from PIL import Image
from open3d.geometry import (
    PointCloud,
    KDTreeSearchParamKNN,
    KDTreeSearchParamHybrid
)
from open3d.io import write_point_cloud
from open3d.utility import Vector3dVector as vec

from torchvision.transforms import functional as Fv
from torchvision.io import read_video

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from src.submodules import VGGT



class BasicPointCloudScene:

    def __init__(
        self, 
        width: Optional[int]=112, 
        height: Optional[int]=112,
        vggt_weights: Optional[str]=None,
        base_transform: Optional[np.ndarray]=None,
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
        
        self.base_Twc = base_transform
        if (self.base_Twc is not None and 
            isinstance(self.base_Twc, np.ndarray)):
            self.base_Twc = torch.Tensor(self.base_Twc)
        
        self.nn_searcher = NearestNeighbors(n_neighbors=n_neighbors)
        
    

    def _aply_tranform(self, pts, viewmat) -> Union[np.ndarray, torch.Tensor]:

        pts = (viewmat[:3, :3] @ pts.T).T
        pts[..., 0] += viewmat[0, -1]
        pts[..., 1] += viewmat[1, -1]
        pts[..., 2] += viewmat[3, -1]
        return pts

    def _vggt2pcd(self, pts, colors, viewmat) -> Tuple[np.ndarray, np.ndarray]:

        points = torch.flatten(pts, end_dim=-2)
        colors = torch.flatten(colors.permute(1, 2, 0), end_dim=-2)
        
        prune_mask = torch.where(torch.norm(points, dim=-1) == 0, True, False)
    
        prune_points = points[~prune_mask].detach()
        prune_points = self._aply_tranform(prune_points, viewmat).numpy()
        prune_colors = colors[~prune_mask].numpy()
        del points, colors
        
        if self.base_Twc is not None:
            prune_points = self._aply_tranform(prune_points, self.base_Twc)

        return (prune_points, prune_colors)
    
    def _handle_pose_encs(self, t, quats) -> torch.Tensor:
        
        viewmats = torch.zeros(t.size()[0], 4, 4)
        for idx in range(t.size()[0]):

            tmp_t = t[idx, :]
            quat = quats[idx, :].numpy()
            Rmat = torch.Tensor(R.from_quat(quat).as_matrix())

            viewmats[idx, :-1, -1] = tmp_t
            viewmats[idx, :3, :3] = Rmat
            if self.base_Twc:
                viewmats = (self.base_Twc @ viewmats)
        
        return viewmats
            
    def _handle_frames_stack(
        self, 
        pts: torch.Tensor, 
        inputs: torch.Tensor, 
        viewmats: torch.Tensor,
        search_param: Union[KDTreeSearchParamHybrid, KDTreeSearchParamKNN],
        batch_idx: Optional[int]=0,
        
    ) -> PointCloud:

        points_ = []
        colors_ = []
        for frame_idx in range(inputs.size()[1]):

            points = pts[batch_idx, frame_idx, ...]
            colors = inputs[batch_idx, frame_idx, ...]
            viewmat = viewmats[frame_idx, ...]
            
            prune_points, prune_colors = self._vggt2pcd(points, colors, viewmat)
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

                t, quats = vggt_item_["pose_enc"][batch_idx, :, :-2].split([3, 4], dim=-1)
                viewmats = self._handle_pose_encs(t, quats)
                self.viewmats.append(viewmats)

                pcd = self._handle_frames_stack(
                    inputs=inputs,
                    pts=vggt_item_["world_points"],
                    search_param=search_param,
                    batch_idx=batch_idx,
                    viewmats=viewmats                    
                )

                self.pcds.append(pcd)
                self.gt_imgs.append(inputs[batch_idx, ...])

            
        else:
            vggt_item_ = self._vggt(inputs.unsqueeze(dim=0))
            pcd = self._handle_frames_stack(
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
            
    
    def _parse_colmap_path(self, path: str) -> Tuple:

        imgs_path = os.path.join(path, "images")
        cameras_path = os.path.join(path, "sparse/images.txt")
        cameras_info = pd.read_csv(cameras_path)
        points_path = os.path.join(path, "sparse/points3D.txt")

        return (imgs_path, cameras_info, points_path)

    def create_from_colmap(
        self,
        path: str,
        partition_size: Optional[int]=1000,
        partitions_n: Optional[int]=32,
        shuffle: Optional[bool]=True,
        search_param: Optional[str | KDTreeSearchParamKNN | KDTreeSearchParamHybrid]="knn",
        k_nns: Optional[int]=30,
        radius: Optional[int]=0.1
    ) -> None:

        imgs_path, cams, points_f = self._parse_colmap_path(path)
        if isinstance(search_param, str):
            if search_param == "knn":
                search_param = KDTreeSearchParamKNN(knn=k_nns)
                
            elif search_param == "hybrid":
                search_param = KDTreeSearchParamHybrid(radius, k_nns)

        points_ = np.zeros(partition_size * partitions_n, 3)
        colors_ = np.zeros(partition_size * partitions_n, 3)
        imgs_fs = set()
        with tqdm(
            desc="Loading Collmap Partitions ...",
            colour="green",
            ascii=":>",
            total=(partition_size * partitions_n)
        ) as pbar:
            with open(points_f, "r") as file:
                data_strings = file.readlines()[3:]
                for _ in range(partitions_n):
                    idx = np.random.randint(0, len(data_strings) - partition_size)
                    for raw_idx in range(idx, idx + partition_size):
                        
                        raw = data_strings[raw_idx].split(" ")
                        xyz = np.asarray([float(val) for val in raw[1:4]])
                        rgb = np.asarray([int(val) for val in raw[4:7]])

                        cam_ids = [int(val) for val in raw[9::2]]
                        cam_fs = set(cams[cams["IMAGE_ID"] == cam_ids]["NAME"].tolist())

                        points_[idx, ...] = xyz
                        colors_[idx, ...] = rgb
                        imgs_fs.update(cam_fs)

                        del data_strings[idx:(idx + partition_size)]
                        pbar.update(1)
        
        pcd = PointCloud()
        pcd.points = vec(points_)
        pcd.colors = vec(colors_)
        pcd.estimate_normals(search_param)

        
        with tqdm(
            desc="Reading imgs ...",
            colour="green",
            ascii=":>",
            total=len(imgs_fs)
        ) as pbar:
            
            quats = np.zeros((len(imgs_fs), 4))
            txyzs = np.zeros((len(imgs_fs, 3)))
            gt_imgs = []
            
            for idx, mg_f in enumerate(imgs_fs):

                cam_raw = cams[cams["NAME"] == img_f]
                img_f = os.path.join(imgs_path, img_f)
                img = Image.open(img_f)
                img = (Fv.pil_to_tensor(img) / 255.0).to(torch.float)
                img = Fv.resize(img, (self.w, self.h))

                quat = np.asarray([float(val) for val in cam_raw[1:5]])
                txyz = np.asarray([float(val) for val in cam_raw[5:8]])
                
                gt_imgs.append(img)
                quats[idx] = quat
                txyzs[idx] = txyz
        
        self.pcds = [pcd]
        self.gt_imgs = [torch.vstack(gt_imgs, dim=0)]
        self.viewmats = [self._handle_pose_encs(quats, txyzs)]
        

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
            for idx, viewmat in enumerate(scene_items["viewmats"]):

                gt_img = scene_items["gt_imgs"][idx]
                if gt_img.shape[0] == 3:
                    gt_img = gt_img.permute(1, 2, 0)

                rr.log(
                    f"{pcd_path}/Frame{idx}",
                    rr.Transform3D(
                        mat3x3=viewmat[:3, :3],
                        translation=viewmat[:3, -1]
                    )
                )
                rr.log(
                    f"{pcd_path}/Frame{idx}/ImgRgb",
                    rr.Pinhole(
                        focal_length=0.5 * (self.K[0, 0].item() + self.K[1, 1].item()),
                        width=self.w, 
                        height=self.h
                    ),
                    rr.Image(gt_img)
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
    
    pcd = BasicPointCloudScene(vggt_weights=weights)
    pcd.create_from_video([video3], 5.0)
    # draw_geometries([pcd[0], pcd[1], pcd[2]])
    pcd.save_ply("/media/test/T7/ply_collection")
    pcd.show()

    
    
            
        

        
            
        

        
        
        
            
            
        
        