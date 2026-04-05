import torch
from src.model import model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 加载 DINO ViT-B/8
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    dino.eval()
    
    # localnet 输入特征维度为 1536
    local_net = model.localnet(len_feature=1536)
    
    dino_params = count_parameters(dino)
    localnet_params = count_parameters(local_net)
    total_params = dino_params + localnet_params

    def params_to_mb(params):
        return params * 4 / 1024 / 1024

    print(f"DINO 参数量: {dino_params:,} ({params_to_mb(dino_params):.2f} MB)")
    print(f"localnet 参数量: {localnet_params:,} ({params_to_mb(localnet_params):.2f} MB)")
    print(f"总参数量: {total_params:,} ({params_to_mb(total_params):.2f} MB)")
    for name, param in local_net.named_parameters():
        print(name, param.dtype)
