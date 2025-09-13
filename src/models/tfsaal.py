import torch
import torch.nn as nn

from typing import (
    Optional,
    Tuple,
    Callable
)
from src.layers import (
    ResampleLayer,
    Mlp
)
from .audio_encoder import AudioEncoder
from .lte_block import LTEBlock
from .amf_block import AmfBlock


class TfSal(nn.Module):

    def __init__(
        self,
        in_features: int,
        hiden_features: Optional[int]=32,
        out_features: Optional[int]=1,
        features_path_size: Optional[Tuple[int]]=(32, 32, 32),
        total_resamplings: Optional[int]=4,
        return_hiden_maps: Optional[bool]=False,
        hiden_act_fn: Callable[..., nn.Module]=nn.ReLU,
        lte_act_fn: Callable[..., nn.Module]=nn.ReLU,
        out_act_fn: Callable[..., nn.Module]=nn.Sigmoid,
        decoding_act_fn: Callable[..., nn.Module]=nn.Sigmoid,
        amf_filt_order: Optional[list]=["local", "global", "audio"],
        amf_filt_depth: Optional[int]=1
    ) -> None:
        

        super().__init__()
        self.rdm = return_hiden_maps

        gen_v_list = [
            hiden_features, (-1, -1, -1),
            hiden_features, (-1, -1, -1),
            hiden_features, (-1, -1, -1),
            hiden_features, (-1, -1, -1),
            hiden_features,
        ]
        up_v_list = [
            out_features, (1, 1, 1),
            out_features, (1, 1, 1),
            out_features, (1, 1, 1),
            out_features
        ]
       
        self.resp_block = ResampleLayer(
            in_features=in_features,
            feature_values=gen_v_list,
            hiden_act=hiden_act_fn,
            use_res=True,
            return_scales=True,
            space_dim="3d"
        )
        gen_v_list = gen_v_list[1:]

        self.hiden_blocks = []
        for idx in range(0, 7, 2):
            if idx == 0:
                block = ResampleLayer(
                    in_features=hiden_features,
                    feature_values=gen_v_list,
                    hiden_act=hiden_act_fn,
                    use_res=True,
                    return_scales=False,
                    space_dim="3d"
                )
            
            else:
                block = ResampleLayer(
                    in_features=hiden_features,
                    feature_values=gen_v_list[:-idx],
                    hiden_act=hiden_act_fn,
                    use_res=True,
                    return_scales=False,
                    space_dim="3d"
                )
            self.hiden_blocks  += [block, ]
        
        self.hiden_blocks = nn.ModuleList(self.hiden_blocks)
        self.decoding_blocks = nn.ModuleList([
            ResampleLayer(
                in_features=hiden_features,
                feature_values=up_v_list,
                hiden_act=decoding_act_fn,
                use_res=True,
                return_scales=False,
                space_dim="3d"
            )
            for _ in range(4)
        ])
        self.lte_blocks = nn.ModuleList([
            LTEBlock(
                in_features=hiden_features,
                out_features=hiden_features,
                lt_size=32,
                fcn_act=lte_act_fn,
            )
            for _ in range(4)
        ])
        self.amf_blocks = nn.ModuleList([
            AmfBlock(
                in_features=hiden_features,
                out_features=hiden_features,
                filt_depth=amf_filt_depth,
                filtration_order=amf_filt_order
            )
            for _ in range(4)
        ])
        
        self.au_encoder = AudioEncoder(hiden_features)
        self.out_block = nn.Sequential(
            nn.Conv3d(4, out_features, (3, 3, 3), 1, 1),
            out_act_fn()
        )
    

    def forward(self, input: Tuple[torch.Tensor]) -> torch.Tensor:

        x, Au = input
        Au = self.au_encoder(Au)

        scale_features = self.resp_block(x)
        dense_maps = []
        hiden_features = []
        for idx, scale_f in enumerate(scale_features):
            
            fx = self.hiden_blocks[idx](scale_f)
            if hiden_features:
                for hf in hiden_features:
                    fx += hf
            
            lte_fx = self.lte_blocks[idx](fx)
            amf_fx = self.amf_blocks[idx]([lte_fx, Au])
            dense_map = self.decoding_blocks[idx](amf_fx)

            dense_maps += [dense_map, ]
            hiden_features += [fx, ]
        
        dense_map = torch.cat(dense_maps, dim=1)
        dense_map = self.out_block(dense_map)
        if self.rdm:
            return (dense_map, dense_maps)

        return dense_map
        
            

             
        

        
if __name__ == "__main__":
    
    import librosa as lib

    path = "/home/ram/Downloads/file_example_WAV_1MG.wav"
    signal = torch.Tensor(lib.load(path)[0]).unsqueeze(dim=0)
    print(signal.size())
    noise = torch.normal(0, 1, (1, 3, 128, 128, 128))
    model = TfSal(3, return_hiden_maps=False)
    
    out = model([noise, signal])
    print(out.size())
    

    

        
            


        
            