import os
import random

import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


MVTEC_CLASS_NAMES = [
     "bottle",
     "cable",
     "capsule",
     "carpet",
     "grid",
     "hazelnut",
     "leather",
     "metal_nut",
     "pill",
     "screw",
     "tile",
     "toothbrush",
     "transistor",
     "wood",
     "zipper",
]

VISA_CLASS_NAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


# DINOv3 backbone -> Transformer block 数（不含 patch embed）。
# 层下标表的范围以此为准；新增 backbone 时这里要同步补上。
DINO_BACKBONE_DEPTHS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vitl16plus": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

_ALL_CLASS_NAMES = MVTEC_CLASS_NAMES + VISA_CLASS_NAMES

# 取「层带前段 + 末尾三层」这组的类：screw 与全部 VisA 类，其余类统一用 band。
_TAIL_MIX_CLASSES = ("screw",) + tuple(VISA_CLASS_NAMES)


def _class_layer_table(band, band_tail):
    """把两组层下标展开成 {类名: [层下标]}，每个类都显式列出以便单独微调。

    band      - 主体类使用的层带
    band_tail - _TAIL_MIX_CLASSES 使用的「层带前段 + 末尾三层」
    """
    table = {name: list(band) for name in _ALL_CLASS_NAMES}
    for name in _TAIL_MIX_CLASSES:
        table[name] = list(band_tail)
    return table


# 按 backbone 分档的 {类名: 层下标}（1-based，-1 = 最后一层）。
# 不同尺寸的 block 数不同，同一组下标不能跨尺寸复用：
#   vits16 / vits16plus / vitb16 : 12 blocks
#   vitl16 / vitl16plus          : 24 blocks
#   vith16plus                   : 32 blocks
#   vit7b16                      : 40 blocks
# 注意：只有 vitl16 这套是既有实验调出来的；其余按相对深度位置等比推导，
# 未做消融验证，换 backbone 时请按需自行微调。
DINO_CLASS_LAYER_INDICES = {
    # ── 24 blocks：主力配置，沿用既有实验数值，勿随意改动 ──
    "dinov3_vitl16": _class_layer_table(
        band=[11, 12, 13, 14, 15, 16],
        band_tail=[11, 12, 13, 22, 23, 24],
    ),
    "dinov3_vitl16plus": _class_layer_table(
        band=[11, 12, 13, 14, 15, 16],
        band_tail=[11, 12, 13, 22, 23, 24],
    ),
    # ── 12 blocks（旧 vitb16 实验用的是末尾三层 [10, 11, 12]，如需复现请改这里）──
    "dinov3_vits16": _class_layer_table(
        band=[6, 7, 8],
        band_tail=[6, 7, 8, 10, 11, 12],
    ),
    "dinov3_vits16plus": _class_layer_table(
        band=[6, 7, 8],
        band_tail=[6, 7, 8, 10, 11, 12],
    ),
    "dinov3_vitb16": _class_layer_table(
        band=[6, 7, 8],
        band_tail=[6, 7, 8, 10, 11, 12],
    ),
    # ── 32 blocks ──
    "dinov3_vith16plus": _class_layer_table(
        band=[15, 16, 17, 18, 19, 20, 21],
        band_tail=[15, 16, 17, 30, 31, 32],
    ),
    # ── 40 blocks ──
    "dinov3_vit7b16": _class_layer_table(
        band=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        band_tail=[18, 19, 20, 38, 39, 40],
    ),
}


def get_class_layer_indices(feature_model: str):
    """取某个 DINOv3 backbone 的 {类名: [1-based 层下标]} 表。

    训练（self_train_ad_*_dinov3.py）与推理（infer_*_residual.py）共用同一份表，
    换 backbone 只需改 DINO_CLASS_LAYER_INDICES 一处即可保持两边同步。
    """
    if feature_model not in DINO_CLASS_LAYER_INDICES:
        supported = ", ".join(sorted(DINO_CLASS_LAYER_INDICES))
        raise ValueError(
            f"feature_model={feature_model!r} 没有对应的 DINO_CLASS_LAYER_INDICES 条目。"
            f"已支持: {supported}"
        )
    return DINO_CLASS_LAYER_INDICES[feature_model]


def _validate_layer_tables():
    """导入时自检：每个下标都要落在对应 backbone 的 block 数内（-1 除外）。

    表是硬编码的，写错就是 bug——与其等加载完 backbone 再在训练里炸，
    不如在 import 阶段直接报出来。
    """
    for model, table in DINO_CLASS_LAYER_INDICES.items():
        depth = DINO_BACKBONE_DEPTHS.get(model)
        if depth is None:
            raise ValueError(f"{model} 在 DINO_BACKBONE_DEPTHS 里没有登记 block 数。")
        for class_name, layers in table.items():
            for layer in layers:
                if layer == -1:
                    continue
                if not 1 <= layer <= depth:
                    raise ValueError(
                        f"{model} 的 {class_name} 层下标 {layer} 越界："
                        f"该 backbone 只有 {depth} 个 block（1-based: 1..{depth}）。"
                    )


_validate_layer_tables()


# Mid / tail layer groups for adaptive_rpn_fusionv2_CA (1-based indices, -1 = last block).
# Used when --use_dino_layer_fusion is enabled instead of DINO_CLASS_LAYER_INDICES.
DINO_LAYER_FUSION_INDICES = {
    "dinov3_vitl16": {
        "mid": [9, 10, 11, 12, 13, 14, 15],
        "tail": [22, 23, -1],
    },
    "dinov3_vitl16plus": {
        "mid": [9, 10, 11, 12, 13, 14, 15],
        "tail": [22, 23, -1],
    },
}


def get_dino_layer_fusion_indices(feature_model: str):
    if feature_model not in DINO_LAYER_FUSION_INDICES:
        supported = ", ".join(sorted(DINO_LAYER_FUSION_INDICES))
        raise ValueError(
            f"feature_model={feature_model!r} has no DINO_LAYER_FUSION_INDICES entry. "
            f"Supported: {supported}"
        )
    cfg = DINO_LAYER_FUSION_INDICES[feature_model]
    return list(cfg["mid"]), list(cfg["tail"])


def get_all_class_names(dataset_name: str):
    if dataset_name == "mvtec":
        return MVTEC_CLASS_NAMES
    return VISA_CLASS_NAMES


class MultiClassFeatureDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        image_size: int = 256,
        crop_size: int = 224,
        patch_mask_size: int = 28,
        seed: int = 0,
        shuffle: bool = True,
        class_names_override: list = None,
    ):
        super().__init__()

        if class_names_override is not None:
            self.class_names = list(class_names_override)
        else:
            self.class_names = get_all_class_names(dataset_name)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.data_path = data_path

        self.transform_x = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.patch_mask_size = int(patch_mask_size)
        if self.patch_mask_size <= 0:
            raise ValueError("patch_mask_size 必须为正整数。")

        self.samples = []

        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            image_paths = self._load_train_image_paths(class_name)

            for path in image_paths:
                self.samples.append((path, class_idx))

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.samples)

        self.classwise_global_indices = {idx: [] for idx in range(len(self.class_names))}
        for idx, (_, class_idx) in enumerate(self.samples):
            self.classwise_global_indices[class_idx].append(idx)

    def _load_train_image_paths(self, class_name: str):
        train_dir = os.path.join(self.data_path, class_name, "train", "good")
        if not os.path.isdir(train_dir):
            return []

        image_paths = []
        for filename in sorted(os.listdir(train_dir)):
            file_path = os.path.join(train_dir, filename)
            if not os.path.isfile(file_path):
                continue
            lower = filename.lower()
            if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
                image_paths.append(file_path)

        return image_paths

    def __len__(self):
        return len(self.samples)

    def _resolve_train_mask_path(self, image_path: str):
        class_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
        defect_type = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)

        # In train/good, only noisy* images require masks.
        if defect_type.lower() == "good" and (not filename.lower().startswith("noisy")):
            return None

        mask_root = os.path.join(class_dir, "train","mask")
        candidates = [
            os.path.join(mask_root, f"{stem}_mask.png"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Mask not found for training image: {image_path}. "
            f"Expected under: {os.path.join(mask_root)}"
        )

    def _load_patch_mask(self, image_path: str):
        mask = np.zeros(
            (self.patch_mask_size, self.patch_mask_size),
            dtype=np.float32,
        )
        mask_path = self._resolve_train_mask_path(image_path)
        if mask_path is None:
            return torch.from_numpy(mask.reshape(-1))

        mask_img = Image.open(mask_path).convert("L")
        mask_img = mask_img.resize(
            (self.patch_mask_size, self.patch_mask_size),
            Image.NEAREST,
        )
        mask_np = np.array(mask_img, dtype=np.uint8)
        mask = (mask_np > 0).astype(np.float32)
        return torch.from_numpy(mask.reshape(-1))

    def __getitem__(self, idx):
        image_path, class_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_x(image)
        patch_mask = self._load_patch_mask(image_path)
        class_idx = torch.tensor(class_idx, dtype=torch.long)
        return image, class_idx, idx, patch_mask
