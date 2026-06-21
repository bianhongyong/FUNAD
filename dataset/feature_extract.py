import sys
import warnings

from dataset import dataset_extract
import AnomalyCLIP_lib
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def str_or_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {'none', 'null', ''}:
        return None
    return value


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y', 't'}:
        return True
    if value in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, e.g. true/false')


def get_subdatasets(data_path, subdataset):
    if subdataset is not None:
        return [subdataset]

    class_names = []
    for name in sorted(os.listdir(data_path)):
        class_dir = os.path.join(data_path, name)
        train_dir = os.path.join(class_dir, 'train')
        if os.path.isdir(class_dir) and os.path.isdir(train_dir):
            class_names.append(name)

    if not class_names:
        raise ValueError(f'No valid class folders found under data_path: {data_path}')

    return class_names

def set_device():
    use_cuda = torch.cuda.is_available()
    return torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('Feature_extract')
    parser.add_argument('--data_path', type=str,default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10")
    parser.add_argument('--feature_path', type=str,default="/media/honeywell/E/bhy/FUNAD/feature/MVTec/no_cls_token")
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'], default='mvtec')
    parser.add_argument('--noise', type=str, choices=['0%', '1%', '2%', '3%', '5%', '10%', '20%'], default='10%')   
    parser.add_argument('-d', '--subdataset', type=str, default=None)
    parser.add_argument('--use_cls_token', type=str2bool, default=False,
                        help='Whether to concatenate cls token with patch tokens (true/false).')
    parser.add_argument('--feature_model', type=str, choices=['dino', 'clip'], default='dino',
                        help='Backbone for feature extraction: dino or clip (AnomalyCLIP).')
    parser.add_argument('--clip_model_name', type=str, default='ViT-L/14@336px',
                        help='CLIP model name for AnomalyCLIP_lib.load().')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24],
                        help='Feature layers used in AnomalyCLIP encode_image().')
    parser.add_argument('--dpam_layer', type=int, default=24,
                        help='DPAM layer used by AnomalyCLIP visual encoder.')
    parser.add_argument('--depth', type=int, default=9, help='learnable_text_embedding_depth for AnomalyCLIP load.')
    parser.add_argument('--n_ctx', type=int, default=12, help='Prompt_length for AnomalyCLIP load.')
    parser.add_argument('--t_n_ctx', type=int, default=4, help='learnable_text_embedding_length for AnomalyCLIP load.')
    return parser.parse_args()


def _convert_imagenet_norm_to_clip_norm(input_tensor):
    mean_imagenet = torch.tensor(IMAGENET_MEAN, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    std_imagenet = torch.tensor(IMAGENET_STD, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    mean_clip = torch.tensor(CLIP_MEAN, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    std_clip = torch.tensor(CLIP_STD, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)

    rgb_01 = input_tensor * std_imagenet + mean_imagenet
    rgb_01 = torch.clamp(rgb_01, 0.0, 1.0)
    clip_tensor = (rgb_01 - mean_clip) / std_clip
    return clip_tensor


def extract_feature(input, feature_extractor, use_cls_token=True, feature_model='dino', features_list=None, dpam_layer=24):
    with torch.no_grad():
        feature_extractor.eval()
        if feature_model == 'clip':
            clip_input = _convert_imagenet_norm_to_clip_norm(input)
            image_features, _, _, patch_projections = feature_extractor.encode_image(
                clip_input, features_list, DPAM_layer=dpam_layer
            )
            x_norm = image_features
            x_prenorm = patch_projections[-1]
        else:
            feature = feature_extractor.get_intermediate_layers(input)[0]
            x_norm = feature[:, 0, :]
            x_prenorm = feature[:, 1:, :]

    if x_norm.dim() == 1:
        x_norm = x_norm.unsqueeze(0)
    if x_prenorm.dim() == 2:
        x_prenorm = x_prenorm.unsqueeze(0)

    if use_cls_token:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], dim=-1)

    if x_prenorm.shape[0] == 1:
        x_prenorm = x_prenorm.squeeze(0)

    return x_prenorm

def extract_and_save_features(feature_extractor, loader, save_path, class_name, noise, is_train=True,
                              use_cls_token=True, feature_model='dino', features_list=None, dpam_layer=24):
    os.makedirs(save_path, exist_ok=True)
    device = set_device()

    if is_train:
        features = []
        for x in loader:
            x = x.to(device)
            feature = extract_feature(
                x, feature_extractor,
                use_cls_token=use_cls_token,
                feature_model=feature_model,
                features_list=features_list,
                dpam_layer=dpam_layer
            )
            feature = feature.detach().cpu().numpy()

            features.append(feature)
        
        features = np.stack(features)
        if noise != '10%':
            os.makedirs(os.path.join(save_path, noise), exist_ok=True)
            np.save(os.path.join(save_path, noise, class_name+'.npy'), features)
        else:
            np.save(os.path.join(save_path, class_name+'.npy'), features)

    else:
        features = []
        label = []
        gt_mask = []

        for x, y, mask in loader:
            x = x.to(device)
            feature = extract_feature(
                x, feature_extractor,
                use_cls_token=use_cls_token,
                feature_model=feature_model,
                features_list=features_list,
                dpam_layer=dpam_layer
            )
            feature = feature.detach().cpu().numpy()

            features.append(feature)
            label.append(y.detach().numpy())
            gt_mask.append(mask.squeeze().detach().numpy())

        features = np.stack(features)
        gt = np.concatenate(label)
        mask = np.stack(gt_mask)

        np.save(os.path.join(save_path, class_name+'_test.npy'), features)
        np.save(os.path.join(save_path, class_name+'_gt.npy'), gt)
        np.save(os.path.join(save_path, class_name+'_mask.npy'), mask)

def main():
    args = parse_args()
    device = set_device()
    class_names = get_subdatasets(args.data_path, args.subdataset)

    if args.feature_model == 'clip':
        anomalyclip_parameters = {
            'Prompt_length': args.n_ctx,
            'learnabel_text_embedding_depth': args.depth,
            'learnabel_text_embedding_length': args.t_n_ctx
        }
        feature_extractor, _ = AnomalyCLIP_lib.load(
            args.clip_model_name,
            device=device,
            design_details=anomalyclip_parameters
        )
        feature_extractor.eval()
        feature_extractor.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
    else:
        feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        feature_extractor = feature_extractor.to(device)

    for class_name in class_names:
        train_set = dataset_extract.MyDataset(dataset_path=args.data_path, dataset=args.dataset, class_name=class_name, is_train=True)
        train_loader = DataLoader(train_set, batch_size=1, pin_memory=True)

        extract_and_save_features(feature_extractor, train_loader, args.feature_path, class_name, args.noise,
                      is_train=True, use_cls_token=args.use_cls_token,
                      feature_model=args.feature_model, features_list=args.features_list, dpam_layer=args.dpam_layer)

        if (args.dataset == 'mvtec') and (args.noise == '10%'):
            test_set = dataset_extract.MyDataset(dataset_path=args.data_path, dataset=args.dataset, class_name=class_name, is_train=False)
            test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)

            extract_and_save_features(feature_extractor, test_loader, args.feature_path, class_name, args.noise,
                                      is_train=False, use_cls_token=args.use_cls_token,
                                      feature_model=args.feature_model, features_list=args.features_list, dpam_layer=args.dpam_layer)

# ---------------------------------------------------------------------------
# DINOv3 特征提取（供 self_train_ad_multiclass_dinov3.py 使用）
# ---------------------------------------------------------------------------

DINOV3_FEATURE_MODEL_REGISTRY = {
    # ---- DINOv3 (patch 16) ----
    "dinov3_vits16": {"hub_entry": "dinov3_vits16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vits16plus": {"hub_entry": "dinov3_vits16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitb16": {"hub_entry": "dinov3_vitb16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitl16": {"hub_entry": "dinov3_vitl16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitl16plus": {"hub_entry": "dinov3_vitl16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vith16plus": {"hub_entry": "dinov3_vith16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vit7b16": {"hub_entry": "dinov3_vit7b16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    # ---- DINOv2 (patch 14, 来自 facebookresearch/dinov2) ----
    "dinov2_vits14":     {"hub_entry": "dinov2_vits14",     "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitb14":     {"hub_entry": "dinov2_vitb14",     "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitl14":     {"hub_entry": "dinov2_vitl14",     "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitg14":     {"hub_entry": "dinov2_vitg14",     "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vits14_reg": {"hub_entry": "dinov2_vits14_reg", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitb14_reg": {"hub_entry": "dinov2_vitb14_reg", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitl14_reg": {"hub_entry": "dinov2_vitl14_reg", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
    "dinov2_vitg14_reg": {"hub_entry": "dinov2_vitg14_reg", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov2_main"},
}

_FEATURE_MODEL_CHOICES = tuple(DINOV3_FEATURE_MODEL_REGISTRY.keys())


def _is_valid_local_torch_hub_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "hubconf.py"))


def _is_dinov2_model(feature_model: str) -> bool:
    """Check if the model key belongs to DINOv2 family."""
    return feature_model.startswith("dinov2_")


def resolve_dinov3_local_hub_dir(feature_model: str):
    if feature_model not in DINOV3_FEATURE_MODEL_REGISTRY:
        raise KeyError(f"Unknown feature_model={feature_model!r}")
    entry = DINOV3_FEATURE_MODEL_REGISTRY[feature_model]
    explicit = entry.get("hub_repo_dir")
    if explicit:
        p = os.path.expanduser(str(explicit))
        if _is_valid_local_torch_hub_dir(p):
            return p
    env_var = "DINOV2_HUB_DIR" if _is_dinov2_model(feature_model) else "DINOV3_HUB_DIR"
    env_dir = os.environ.get(env_var)
    if env_dir:
        p = os.path.expanduser(env_dir)
        if _is_valid_local_torch_hub_dir(p):
            return p
    repo_dir_name = "facebookresearch_dinov2_main" if _is_dinov2_model(feature_model) else "facebookresearch_dinov3_main"
    default_dir = os.path.join(torch.hub.get_dir(), repo_dir_name)
    if _is_valid_local_torch_hub_dir(default_dir):
        return default_dir
    legacy = os.path.expanduser(f"~/.cache/torch/hub/{repo_dir_name}")
    if _is_valid_local_torch_hub_dir(legacy):
        return legacy
    return None


def build_dinov3_feature_extractor(feature_model: str, device: torch.device):
    """Build a DINOv2/DINOv3 feature extractor from torch.hub.

    Supports both ``dinov3_*`` and ``dinov2_*`` model keys registered
    in *DINOV3_FEATURE_MODEL_REGISTRY*.
    """
    entry = DINOV3_FEATURE_MODEL_REGISTRY[feature_model]
    hub_entry = entry["hub_entry"]
    repo_dir = resolve_dinov3_local_hub_dir(feature_model)

    is_dinov2 = _is_dinov2_model(feature_model)
    model_family = "DINOv2" if is_dinov2 else "DINOv3"
    gh_repo = "facebookresearch/dinov2" if is_dinov2 else "facebookresearch/dinov3"

    # DINO imports `trunc_normal_` via `from utils import ...`.
    # This project also has `utils.py`, so temporarily unshadow it.
    local_utils_module = sys.modules.get("utils")
    should_restore_utils = (
        local_utils_module is not None
        and os.path.abspath(getattr(local_utils_module, "__file__", "")).endswith(
            os.path.join("FUNAD", "utils.py")
        )
    )
    if should_restore_utils:
        del sys.modules["utils"]
    try:
        if repo_dir is not None:
            print(f"[{model_family}] load {hub_entry} from local hub: {repo_dir}")
            feature_extractor = torch.hub.load(
                repo_dir,
                hub_entry,
                source="local",
                pretrained=True,
            )
        else:
            print(f"[{model_family}] load {hub_entry} from GitHub: {gh_repo}")
            feature_extractor = torch.hub.load(
                gh_repo,
                hub_entry,
                pretrained=True,
            )
    finally:
        if should_restore_utils:
            sys.modules["utils"] = local_utils_module
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def resolve_dino_block_indices(feature_extractor, dino_layer_indices):
    """Resolve 1-based layer indices to 0-based block indices."""
    num_blocks = len(getattr(feature_extractor, "blocks", []))
    raw = list(dino_layer_indices)
    requested_layers = [num_blocks if x == -1 else x for x in raw]

    for lid in requested_layers:
        if lid < 1 or lid > num_blocks:
            raise ValueError(
                f"dino_layer_indices {lid} (1-based) is out of range. "
                f"Model has {num_blocks} blocks (1-based: 1..{num_blocks}). "
                f"Received: {raw}"
            )

    return sorted({int(lid) - 1 for lid in requested_layers})


def extract_dinov3_feature_batch(input_tensor, feature_extractor, dino_layer_indices, class_indices=None, use_cls_token=False, return_cls_token=False):
    """Extract DINOv3 patch features from a batch of images.

    Args:
        dino_layer_indices: list[int] (global, used for all samples),
            or dict[int, list[int]] mapping class_idx -> layer_indices (per-class).
        class_indices: Tensor[B], required when dino_layer_indices is a dict.
    """
    with torch.no_grad():
        feature_extractor.eval()

        is_per_class = isinstance(dino_layer_indices, dict)
        if is_per_class:
            # Per-class mode: compute union of all needed layers, extract once,
            # then per-sample select and average only its class's layers.
            assert class_indices is not None, "class_indices required when dino_layer_indices is a dict"
            all_unique = sorted({idx for lst in dino_layer_indices.values() for idx in lst})
            selected_blocks = resolve_dino_block_indices(feature_extractor, all_unique)
            selected_layers = feature_extractor.get_intermediate_layers(
                input_tensor, n=selected_blocks, return_class_token=True,
            )
            # [L, B, N, C] and [L, B, C]
            all_patches = torch.stack([p for p, _ in selected_layers], dim=0)
            all_cls = torch.stack([c for _, c in selected_layers], dim=0)

            cls_to_blocks = {
                cls: set(resolve_dino_block_indices(feature_extractor, idxs))
                for cls, idxs in dino_layer_indices.items()
            }
            batch_size = input_tensor.shape[0]
            patch_list, cls_list = [], []
            for i in range(batch_size):
                cls = int(class_indices[i])
                cls_blocks = cls_to_blocks[cls]
                mask = torch.tensor(
                    [b in cls_blocks for b in selected_blocks],
                    device=input_tensor.device,
                )
                patch_list.append(all_patches[mask, i].mean(dim=0))
                cls_list.append(all_cls[mask, i].mean(dim=0))
            x_prenorm = torch.stack(patch_list, dim=0)
            cls_tok = torch.stack(cls_list, dim=0)
        else:
            # Original mode: single global layer set for all samples
            selected_blocks = resolve_dino_block_indices(feature_extractor, dino_layer_indices)
            selected_layers = feature_extractor.get_intermediate_layers(
                input_tensor, n=selected_blocks, return_class_token=True,
            )
            patch_tokens_list = [layer_patch for layer_patch, _layer_cls in selected_layers]
            cls_tokens_list = [_layer_cls for _layer_patch, _layer_cls in selected_layers]
            x_prenorm = torch.stack(patch_tokens_list, dim=0).mean(dim=0)
            cls_tok = torch.stack(cls_tokens_list, dim=0).mean(dim=0)

        x_norm = cls_tok
        cls_token = cls_tok

    if use_cls_token:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], dim=-1)
    if return_cls_token:
        return x_prenorm, cls_token
    return x_prenorm


def infer_dinov3_feature_dim(feature_extractor, train_loader, dino_layer_indices, use_cls_token, device):
    for batch in train_loader:
        images = batch[0]
        images = images.to(device, non_blocking=True)
        features = extract_dinov3_feature_batch(
            images, feature_extractor, dino_layer_indices, use_cls_token=use_cls_token,
        )
        return int(features.shape[-1])
    raise RuntimeError("训练集为空，无法推断特征维度。")


def infer_dinov3_patch_mask_size(train_dataset, feature_extractor, dino_layer_indices, use_cls_token, device):
    if len(train_dataset.samples) == 0:
        raise RuntimeError("训练集为空，无法推断 patch_mask_size。")
    image_path, _ = train_dataset.samples[0]
    image = Image.open(image_path).convert("RGB")
    image = train_dataset.transform_x(image).unsqueeze(0).to(device, non_blocking=True)
    features = extract_dinov3_feature_batch(
        image, feature_extractor, dino_layer_indices, use_cls_token=use_cls_token,
    )
    num_patches = int(features.shape[1])
    patch_mask_size = int(np.sqrt(num_patches))
    if patch_mask_size * patch_mask_size != num_patches:
        raise RuntimeError(
            f"无法从 patch 数 {num_patches} 推断方形 patch 网格，请检查模型与输入尺寸。"
        )
    return patch_mask_size

if __name__ == "__main__":
    main()
