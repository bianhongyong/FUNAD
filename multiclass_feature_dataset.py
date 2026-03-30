import os
import random

import torch
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
        seed: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()

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
        train_dir = os.path.join(self.data_path, class_name, "train")
        if not os.path.isdir(train_dir):
            return []

        image_paths = []
        for defect_type in sorted(os.listdir(train_dir)):
            defect_dir = os.path.join(train_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue

            for filename in sorted(os.listdir(defect_dir)):
                lower = filename.lower()
                if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
                    image_paths.append(os.path.join(defect_dir, filename))

        return image_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_x(image)
        class_idx = torch.tensor(class_idx, dtype=torch.long)
        return image, class_idx, idx
