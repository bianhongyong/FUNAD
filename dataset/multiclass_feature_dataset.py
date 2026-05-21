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


# Per-class DINOv3 layer indices (1-based). -1 means last layer.
# Shared by training (self_train_ad_multiclass_dinov3.py) and inference
# (infer_multiclass_residual.py). Modify here to keep both in sync.
DINO_CLASS_LAYER_INDICES = {
    # MVTec
    "bottle": [11,12,13,14,15,16], "cable": [11,12,13,14,15,16], "capsule": [11,12,13,14,15,16], "carpet": [11,12,13,14,15,16],
    "grid": [11,12,13,14,15,16], "hazelnut": [11,12,13,14,15,16], "leather": [11,12,13,14,15,16], "metal_nut": [11,12,13,14,15,16],
    "pill": [11,12,13,14,15,16], "screw": [11,12,13,22,23,24], "tile": [11,12,13,14,15,16], "toothbrush": [11,12,13,14,15,16],
    "transistor": [11,12,13,14,15,16], "wood": [11,12,13,14,15,16], "zipper": [11,12,13,14,15,16],
    # VISA
    "candle": [11,12,13,22,23,24],
    "capsules": [11,12,13,22,23,24],
    "cashew": [11,12,13,22,23,24],
    "chewinggum": [11,12,13,22,23,24],
    "fryum": [11,12,13,22,23,24],
    "macaroni1": [11,12,13,22,23,24],
    "macaroni2": [11,12,13,22,23,24],
    "pipe_fryum": [11,12,13,22,23,24],
    "pcb1": [11,12,13,22,23,24],
    "pcb2": [11,12,13,22,23,24],
    "pcb3": [11,12,13,22,23,24],
    "pcb4": [11,12,13,22,23,24]
}

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
