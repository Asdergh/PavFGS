import torch 

REPO_DIR = "/home/ram/Desktop/own_projects/dinov3"
WEIGHTS_F = "/home/ram/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
dinov3_vit7b16 = torch.hub.load(
    REPO_DIR, 
    'dinov3_vit7b16', 
    source='local', 
    verbose=True
)

print(dinov3_vit7b16)
