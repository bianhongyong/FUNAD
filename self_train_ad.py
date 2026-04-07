import torch
import argparse
import numpy as np
import random
import torch.backends.cudnn as cudnn
import os
import wandb
import sys
from src.model import model
from dataset import dataset
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd
import datetime
from src.inference import inference
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
from dataset import dataload
import math

# from info_nce import InfoNCE, info_nce
from skimage import morphology
import imgaug.augmenters as iaa
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F
import time
import warnings
from src.train.epoch_precompute import precompute_pseudo_labels_feature
from utils import evaluate as eval_utils
from utils.loss import compute_balanced_bce_loss, compute_oto_loss_single
from utils.print import print_epoch_losses, print_epoch_times
import utils.train_utils as common_utils

warnings.filterwarnings("ignore")

import sys

print(sys.executable)

matplotlib.use("Agg")
# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OnDemandFeatureDataset:
    def __init__(self, feature_array):
        self.feature_array = feature_array
        self.extra_samples = []

    def __len__(self):
        return int(self.feature_array.shape[0]) + len(self.extra_samples)

    def __getitem__(self, idx):
        base_len = int(self.feature_array.shape[0])
        if idx < base_len:
            return self.feature_array[idx]
        return self.extra_samples[idx - base_len]

    def append(self, sample):
        self.extra_samples.append(sample)


class IndexedOnDemandFeatureDataset(Dataset):
    def __init__(self, feature_dataset):
        self.feature_dataset = feature_dataset

    def __len__(self):
        return len(self.feature_dataset)

    def __getitem__(self, idx):
        feature = self.feature_dataset[idx]
        feature = np.asarray(feature, dtype=np.float32)
        return torch.from_numpy(feature), idx


def parse_args():
    parser = argparse.ArgumentParser("self-train_ad")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument(
        "--save_path", type=str, default="/media/honeywell/E/bhy/FUNAD/save_results"
    )
    parser.add_argument(
        "--feature_path", type=str, default="/media/honeywell/E/bhy/FUNAD/feature/MVTec"
    )
    parser.add_argument(
        "--synthetic_path", type=str, default="/media/honeywell/E/bhy/FUNAD/synthetic"
    )
    parser.add_argument("--kl", action="store_false")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--gaussian", action="store_false")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="mvtec", choices=["mvtec", "visa"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-r", "--random", type=float, default=0.1)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.9)
    parser.add_argument("-d", "--subdataset", type=str, default="screw")
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "10%", "20%"],
    )
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--k_number", type=int, default=2)
    parser.add_argument("--llambda", type=float, default=1)
    parser.add_argument("--weight", type=float, default=2.5)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--beta_number", type=int, default=15)
    parser.add_argument("--alternative", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--balancing", action="store_false")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument(
        "--oto_loss", type=str, choices=["kl", "mae", "mse"], default="mae"
    )
    parser.add_argument("--perlin", action="store_true")
    parser.add_argument(
        "--perlin_ratio",
        type=str,
        default="10%",
        choices=["1%", "5%", "10%", "25%", "50%", "100%"],
    )
    parser.add_argument("--eval_interval", type=float, default=1)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--backbone_type", type=str, default="vit")

    return parser.parse_args()


def find_matching(id):
    return common_utils.find_matching(id)


def compute_distance(feature):
    return common_utils.compute_distance(feature, use_cuda=use_cuda)


def fix_seed(number):
    common_utils.fix_seed(number)


def extract_feature(input, feature_extractor, concat):
    # 预训练特征提取器 E: 取 ViT patch token，并可拼接 cls token（1536 维）。
    with torch.no_grad():
        feature_extractor.eval()
        feature = feature_extractor.get_intermediate_layers(input)[0]

    x_prenorm = feature[:, 1:, :]  # patch tokens
    x_prenorm = x_prenorm.squeeze()

    if concat:
        x_norm = feature[:, 0, :]  # [cls] tokens
        x_norm = x_norm.squeeze()

    if x_prenorm.shape[1] != 784:
        x_prenorm = x_prenorm.unsqueeze(0)
        x_norm = x_norm.unsqueeze(0)

    if concat:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], axis=-1)

    return x_prenorm


def save_model(model, saved_dir, weight_name):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {"net": model.state_dict()}
    torch.save(check_point, os.path.join(saved_dir, weight_name))


def set_wandb(args):
    wandb.init(
        project=args.dataset + "_" + args.subdataset,
        config={
            "learning_rate": args.lr,
            "num_epochs": args.epoch,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "random": args.random,
            "noise": args.noise,
            "overlap": args.overlap,
            "balancing": args.balancing,
            "oto": args.kl,
            "oto_loss": args.oto_loss,
            "gaussian": args.gaussian,
            "weight": args.weight,
            "synthetic": args.synthetic,
        },
        name="gaussian_"
        + str(args.gaussian)
        + "_noise_"
        + args.noise
        + "_balancing_"
        + str(args.balancing)
        + "_oto_"
        + str(args.kl)
        + "_weight_"
        + str(args.weight)
        + "_synthetic_"
        + str(args.synthetic),
    )


def build_backbone(args):
    backbone = None
    resnet_channel = None
    resnet_idx = None

    if args.synthetic or args.perlin:
        if args.backbone_type == "vit":
            backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(
                device
            )
        elif args.backbone_type == "wideresnet":
            outlayers = ["layer1", "layer2", "layer3", "layer4"]
            layers_idx = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
            backbone = timm.create_model(
                "wide_resnet50_2",
                features_only=True,
                pretrained=True,
                out_indices=[layers_idx[outlayer] for outlayer in outlayers],
            )
            resnet_channel = 1024
            resnet_idx = 2

    return backbone, resnet_channel, resnet_idx


def get_class_names(args):
    if args.subdataset is not None:
        return [args.subdataset]

    if args.dataset == "mvtec":
        return [
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

    return [
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


def load_train_features(args, class_name):
    if args.noise != "10%":
        feature_file = os.path.join(args.feature_path, args.noise, class_name + ".npy")
    else:
        if args.patch:
            feature_file = os.path.join(args.feature_path, class_name + "_train_patch.npy")
        else:
            feature_file = os.path.join(args.feature_path, class_name + ".npy")

    train_features = np.load(feature_file, mmap_mode="r")
    train_features = train_features.reshape(-1, 784, train_features.shape[-1])
    return OnDemandFeatureDataset(train_features)


def maybe_append_perlin_samples(
    args, class_name, train_dataset, backbone, resnet_channel, resnet_idx
):
    perlin_length = 0
    if not args.perlin:
        return train_dataset, perlin_length

    synthetic_dataset = dataload.ImageDataset(
        dataset_path=args.synthetic_path, class_name=class_name, synthetic=True
    )
    ratio_map = {
        "1%": 0.01,
        "3%": 0.03,
        "5%": 0.05,
        "10%": 0.10,
        "25%": 0.25,
        "50%": 0.50,
        "100%": 1.00,
    }
    perlin_length = int(len(train_dataset) * ratio_map.get(args.perlin_ratio, 0.10))

    synthetic_features = []
    for i in range(perlin_length):
        image, mask = synthetic_dataset[i]
        image = image.unsqueeze(0).cuda()
        if args.backbone_type == "vit":
            _syn_feature = extract_feature(image, backbone, True)
        else:
            image = image.cpu()
            with torch.no_grad():
                _syn_feature = (
                    backbone(image)[resnet_idx]
                    .reshape(1, resnet_channel, -1)
                    .permute(0, 2, 1)
                )
                _syn_feature = (
                    F.interpolate(
                        _syn_feature.reshape(1, 14, 14, 1024).permute(0, 3, 1, 2),
                        size=(28, 28),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .reshape(1, 1024, -1)
                    .permute(0, 2, 1)
                )

        dim = _syn_feature.shape[-1]
        _syn_feature = _syn_feature.cpu().reshape(-1, dim)
        synthetic_features.append(_syn_feature.cpu().detach().numpy())

    for i in range(perlin_length):
        train_dataset.append(synthetic_features[i])

    return train_dataset, perlin_length


def build_data_loaders(train_dataset, args):
    indexed_dataset = IndexedOnDemandFeatureDataset(train_dataset)

    if args.perlin:
        local_loader = DataLoader(
            indexed_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            num_workers=16,
        )
    else:
        local_loader = DataLoader(
            indexed_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=16,
        )

    mini_loader = DataLoader(
        indexed_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False
    )
    return local_loader, mini_loader


def build_test_loader(args, class_name):
    if args.overlap:
        test_features = np.load(
            os.path.join(args.feature_path, class_name + "_test.npy")
        ).squeeze()
        mask_gt = np.load(
            os.path.join(args.feature_path, class_name + "_mask.npy")
        ).squeeze()
        mask_gt = np.ceil(mask_gt)
        label_gt = np.load(os.path.join(args.feature_path, class_name + "_gt.npy"))
    else:
        if args.patch:
            test_features = np.load(
                os.path.join(args.feature_path, class_name + "_test_patch.npy")
            ).squeeze()
            mask_gt = np.load(
                os.path.join(args.feature_path, class_name + "_test_patch_masks.npy")
            ).squeeze()
            mask_gt = np.ceil(mask_gt)
            label_gt = np.zeros(mask_gt.shape[0])
            label_gt[np.unique(np.where(mask_gt == 1)[0])] = 1
        else:
            test_features = np.load(
                os.path.join(args.feature_path, class_name + "_test.npy")
            ).squeeze()
            mask_gt = np.load(
                os.path.join(args.feature_path, class_name + "_mask.npy")
            ).squeeze()
            mask_gt = np.ceil(mask_gt)
            label_gt = np.load(os.path.join(args.feature_path, class_name + "_gt.npy"))

    test_dataset = [
        [test_features[i], label_gt[i], mask_gt[i]]
        for i in range(test_features.shape[0])
    ]
    return DataLoader(test_dataset, batch_size=16, pin_memory=True)


def get_oto_loss(args):
    return common_utils.get_oto_loss(args.oto_loss, device)


def evaluate_epoch(localnet, test_loader):
    return eval_utils.evaluate_feature_epoch(localnet, test_loader, device)


def _update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k):
    return common_utils.update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k)


def train_one_epoch(
    args,
    class_name,
    epoch,
    localnet,
    localnet_optimizer,
    onetoone_optimizer,
    localnet_criterion,
    l_loss,
    local_loader,
    mini_loader,
    train_dataset,
    synthetic_dataset,
    backbone,
    perlin_length,
    iteration,
):
    total_batch = len(local_loader)
    threshold = args.threshold
    noise_threshold = args.noise_threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
    memory_bank_time = 0.0
    pseudo_label_time = 0.0
    kl_loss_time = 0.0

    if args.perlin:
        original_indices = list(range(len(train_dataset)))
        shuffled_indices = np.random.permutation(original_indices)
        index_mapping = dict(zip(original_indices, shuffled_indices))
        perlin_indices = original_indices[-perlin_length:]

    if threshold <= 1:
        distance_map, confident_features, dim, mb_time, pl_time = (
            precompute_pseudo_labels_feature(
                args=args,
                localnet=localnet,
                mini_loader=mini_loader,
                device=device,
                update_topk_features_fn=_update_topk_features,
            )
        )
        memory_bank_time += mb_time
        pseudo_label_time += pl_time

    for data_counter, batch_data in enumerate(
        tqdm.tqdm(local_loader, "| run | train | " + str(epoch + 1) + " |")
    ):
        x, sample_idx = batch_data
        sample_idx = sample_idx.detach().cpu().numpy()
        batch = x.shape[0]
        if args.perlin:
            shuffled_dataloader = []
            original_indices_batch = range(
                data_counter * local_loader.batch_size,
                (data_counter + 1) * local_loader.batch_size,
            )
            shuffled_indices_batch = [
                index_mapping[original_index]
                for original_index in original_indices_batch
            ]
            shuffled_batch = [
                torch.as_tensor(train_dataset[shuffled_index], dtype=torch.float32)
                for shuffled_index in shuffled_indices_batch
            ]
            shuffled_dataloader.extend(shuffled_batch)
            x = torch.stack(shuffled_dataloader)
            sample_idx = np.asarray(shuffled_indices_batch)

        distance = np.zeros((batch, 784), dtype=np.float32)
        if threshold <= 1:
            with torch.no_grad():
                if args.beta:
                    first_vector = []
                    second_vector = []

                    for i in range(784):
                        first = random.randint(0, args.beta_number - 1)
                        second = random.randint(0, args.beta_number - 1)
                        while second == first:
                            second = random.randint(0, args.beta_number - 1)
                        first_vector.append(confident_features[first])
                        second_vector.append(confident_features[second])

                    first_vector = torch.stack(first_vector)
                    second_vector = torch.stack(second_vector)

                    mix_ratio = random.random()
                    syn_anomaly = (
                        mix_ratio * first_vector + (1 - mix_ratio) * second_vector
                    )
                    syn_anomaly = syn_anomaly.unsqueeze(0)
                distance = distance_map[sample_idx]

                if (args.threshold <= 1) and args.gaussian:
                    _idx = (distance > threshold) & (distance < noise_threshold)
                    _idx = _idx.reshape(-1)
                    n = np.where(_idx)[0].shape[0]
                    _copy = x.detach()

                    if n > 0:
                        if args.std == None:
                            std = _copy.std(axis=0).max(axis=0)[0]
                            std = std.repeat_interleave(n).reshape(-1, n)
                            std = std.T
                            _copy = _copy.reshape(-1, dim)
                            _copy[_idx] += torch.normal(mean=0, std=std)
                        else:
                            _copy = _copy.reshape(-1, dim)
                            _copy[_idx] += torch.normal(
                                mean=0, std=args.std, size=_copy[_idx].shape
                            )

                    _copy = _copy.reshape(-1, 784, dim)
        else:
            dim = x.shape[-1]

        if args.beta:
            pseudo_label_assign_start = time.perf_counter()
            local_label = torch.zeros((args.batch_size + 1, 784))
            distance[distance > threshold] = 1
            distance[distance <= threshold] = 0
            local_label[:-1] = torch.tensor(distance)
            if (args.threshold <= 1) and args.gaussian:
                _copy = torch.concat([_copy, syn_anomaly])
            else:
                x = torch.concat([x, syn_anomaly])
            pseudo_label_time += time.perf_counter() - pseudo_label_assign_start

        else:
            pseudo_label_assign_start = time.perf_counter()
            local_label = torch.zeros((args.batch_size, 784))
            if args.threshold <= 1:
                distance_mask = torch.as_tensor(distance > threshold, dtype=torch.bool)
                local_label[distance_mask] = 1
            pseudo_label_time += time.perf_counter() - pseudo_label_assign_start

            if args.synthetic:
                synthetic_ids = []
                synthetic_features = []

                perlin_length = int(len(train_dataset) * 0.1)
                synthetic_loop = int(
                    perlin_length / (len(train_dataset) / args.batch_size)
                )

                for i in range(synthetic_loop):
                    a = np.random.randint(len(synthetic_dataset))
                    while a in synthetic_ids:
                        a = np.random.randint(len(synthetic_dataset))
                    synthetic_ids.append(a)
                    image, mask = synthetic_dataset[a]
                    image = image.unsqueeze(0).to(device)
                    _syn_feature = extract_feature(image, backbone, True)
                    _syn_feature = _syn_feature.cpu()
                    mask = mask.reshape(-1)
                    _syn_feature = _syn_feature.reshape(-1, dim)
                    synthetic_features.append(_syn_feature[mask == 1])

                synthetic_features = torch.cat(synthetic_features, dim=0)
                local_label = local_label.reshape(-1)
                local_label = torch.cat(
                    [local_label, torch.ones(synthetic_features.shape[0])]
                )

                x = x.reshape(-1, dim)
                x = torch.cat([x, synthetic_features])

                if (args.threshold <= 1) and args.gaussian:
                    _copy = _copy.reshape(-1, dim)
                    _copy = torch.cat([_copy, synthetic_features])

            if (args.threshold <= 1) and args.hist:
                fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
                axes.hist(distance.reshape(-1), density=True, bins=100, alpha=1)
                axes.set_title(class_name)
                axes.set_xlabel("local feature distance")
                axes.set_ylabel("density")
                plot_dir = os.path.join("plot", class_name)
                os.makedirs(plot_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(plot_dir, "distance_" + str(iteration) + ".png"),
                    dpi=300,
                    format="png",
                    bbox_inches="tight",
                )
                plt.close()

        localnet.train()
        localnet_optimizer.zero_grad()
        if args.alternative:
            onetoone_optimizer.zero_grad()

        x = x.to(device)
        local_label = local_label.to(device)

        batch_feature, local_pred = localnet(x)

        if (args.threshold <= 1) and args.gaussian:
            _copy = _copy.to(device)
            _, gaussian_pred = localnet(_copy)
            if args.balancing:
                _loss = compute_balanced_bce_loss(
                    localnet_criterion=localnet_criterion,
                    pred=gaussian_pred,
                    target=local_label,
                )
            else:
                _loss = localnet_criterion(gaussian_pred, local_label)
        else:
            if args.balancing:
                _loss = compute_balanced_bce_loss(
                    localnet_criterion=localnet_criterion,
                    pred=local_pred,
                    target=local_label,
                )
            else:
                _loss = localnet_criterion(local_pred, local_label)

        if (iteration >= args.iter) and args.kl:
            kl_start = time.perf_counter()
            if args.synthetic:
                real = args.batch_size * 784
            else:
                real = args.batch_size

            _l_loss = compute_oto_loss_single(
                args=args,
                batch_feature=batch_feature,
                local_pred=local_pred,
                real_count=real,
                l_loss=l_loss,
                compute_distance_fn=compute_distance,
                find_matching_fn=find_matching,
            )

            if args.perlin:
                feature_np = (
                    batch_feature.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1, batch_feature.shape[-1])
                )
                _, id = compute_distance(feature_np)
                matched_id = find_matching(id)
                target = local_pred[:real].reshape(-1)[matched_id[0]]
                input = local_pred[:real].reshape(-1)[matched_id[1]]
                for idx, shuf_idx in enumerate(shuffled_indices_batch):
                    if shuf_idx in perlin_indices:
                        target[idx] = target[idx].detach()
                        input[idx] = input[idx].detach()
                if args.oto_loss == "kl":
                    _l_loss = 0.5 * (
                        l_loss(input.log(), (input + target) / 2)
                        + l_loss(target.log(), (input + target) / 2)
                    )
                else:
                    _l_loss = 0.5 * (
                        l_loss(input, (input + target) / 2)
                        + l_loss(target, (input + target) / 2)
                    )
            kl_loss_time += time.perf_counter() - kl_start
        else:
            _l_loss = 0

        if args.alternative:
            _local_loss = _loss
        else:
            _local_loss = _loss + args.weight * _l_loss

        try:
            _local_loss.backward()
            localnet_optimizer.step()

            if args.alternative:
                _l_loss.backward()
                onetoone_optimizer.step()

        except Exception as err:
            pdb.set_trace()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
        oto_loss += _l_loss / total_batch
        iteration += 1

    local_loss_value = (
        local_loss.item() if torch.is_tensor(local_loss) else float(local_loss)
    )
    bce_loss_value = bce_loss.item() if torch.is_tensor(bce_loss) else float(bce_loss)
    oto_loss_value = oto_loss.item() if torch.is_tensor(oto_loss) else float(oto_loss)

    return (
        local_loss_value,
        bce_loss_value,
        oto_loss_value,
        iteration,
        memory_bank_time,
        pseudo_label_time,
        kl_loss_time,
    )


def main():
    # 主训练流程对应论文 Algorithm 1:
    # 1) IRMB 构建 2) patch 伪标签 3) mutual smoothness 4) 联合优化。
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    tsne = TSNE(n_components=2, random_state=args.seed)
    backbone, resnet_channel, resnet_idx = build_backbone(args)
    # torch.cuda.set_device(args.gpu)

    CLASS_NAMES = get_class_names(args)

    localnet = model.localnet(len_feature=1536)
    localnet = localnet.to(device)

    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    if args.alternative:
        onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr)
    localnet_criterion = nn.BCELoss().to(device)

    results = []

    for class_name in CLASS_NAMES:
        # 每个类别独立训练与评估。
        fix_seed(args.seed)
        if args.wandb:
            set_wandb(args)

        saved_dir = os.path.join(args.save_path, "results", class_name)
        os.makedirs(saved_dir, exist_ok=True)

        train_dataset = load_train_features(args, class_name)

        train_dataset, perlin_length = maybe_append_perlin_samples(
            args,
            class_name,
            train_dataset,
            backbone,
            resnet_channel,
            resnet_idx,
        ) 

        if args.synthetic:
            synthetic_dataset = dataload.ImageDataset(
                dataset_path=args.synthetic_path, class_name=class_name, synthetic=True
            )
        else:
            synthetic_dataset = None

        local_loader, mini_loader = build_data_loaders(train_dataset, args)
        test_loader = build_test_loader(args, class_name)
        l_loss = get_oto_loss(args)

        iteration = 0
        if args.beta:
            beta = torch.distributions.beta.Beta(
                0.5 * torch.ones(784), 0.5 * torch.ones(784)
            )

        for epoch in range(args.epoch):
            (
                local_loss_value,
                bce_loss_value,
                oto_loss_value,
                iteration,
                memory_bank_time,
                pseudo_label_time,
                kl_loss_time,
            ) = train_one_epoch(
                args=args,
                class_name=class_name,
                epoch=epoch,
                localnet=localnet,
                localnet_optimizer=localnet_optimizer,
                onetoone_optimizer=onetoone_optimizer if args.alternative else None,
                localnet_criterion=localnet_criterion,
                l_loss=l_loss,
                local_loader=local_loader,
                mini_loader=mini_loader,
                train_dataset=train_dataset,
                synthetic_dataset=synthetic_dataset,
                backbone=backbone,
                perlin_length=perlin_length,
                iteration=iteration,
            )
            print_epoch_losses(epoch, local_loss_value, bce_loss_value, oto_loss_value)
            print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time)

            if (epoch) % args.eval_interval == 0:
                (
                    auroc,
                    ap_sp,
                    f1_sp,
                    pixel_auroc,
                    ap_px,
                    f1_px,
                    aupro_px,
                ) = evaluate_epoch(localnet, test_loader)
                num_epoch = epoch + 1

                print(
                    "epoch %d |" % num_epoch,
                    (
                        f"auroc: {auroc:.5f}, ap_sp: {ap_sp:.5f}, f1_sp: {f1_sp:.5f}, "
                        f"pixel auroc: {pixel_auroc:.5f}, ap_px: {ap_px:.5f}, "
                        f"f1_px: {f1_px:.5f}, aupro_px: {aupro_px:.5f}"
                    ),
                )

                if args.wandb:
                    wandb.log(
                        {
                            "total loss": local_loss_value,
                            "one-to-one loss": oto_loss_value,
                            "bce loss": bce_loss_value,
                            "image AUC": auroc,
                            "image AP": ap_sp,
                            "image F1-max": f1_sp,
                            "pixel AUC": pixel_auroc,
                            "pixel AP": ap_px,
                            "pixel F1-max": f1_px,
                            "pixel AUPRO": aupro_px,
                        }
                    )

                mean = (auroc + pixel_auroc) / 2

                if epoch == 0:
                    best = mean
                    fix_auroc = auroc
                    fix_ap_sp = ap_sp
                    fix_f1_sp = f1_sp
                    fix_pauroc = pixel_auroc
                    fix_ap_px = ap_px
                    fix_f1_px = f1_px
                    fix_aupro_px = aupro_px
                else:
                    if mean > best:
                        best = mean
                        fix_auroc = auroc
                        fix_ap_sp = ap_sp
                        fix_f1_sp = f1_sp
                        fix_pauroc = pixel_auroc
                        fix_ap_px = ap_px
                        fix_f1_px = f1_px
                        fix_aupro_px = aupro_px
                        print(f"class: {class_name}, data_noise: {args.noise}")
                        print("curr_best_auc: ", fix_auroc)
                        print("curr_best_pauc: ", fix_pauroc)
                        save_model(
                            localnet,
                            saved_dir,
                            "gaussian_"
                            + str(args.gaussian)
                            + "_noise_"
                            + args.noise
                            + "_balancing_"
                            + str(args.balancing)
                            + "_oto_"
                            + str(args.kl)
                            + "_weight_"
                            + str(args.weight)
                            + "_synthetic_"
                            + str(args.synthetic)
                            + "_localnet.pt",
                        )
                if epoch % 100 == 0:
                    print("class: ", class_name)

                # Log
                if args.save_log:
                    directory = f"./log_5/{args.epoch}_epoch/{class_name}"
                    file_name = f"metrics.txt"
                    file_path = os.path.join(directory, file_name)
                    os.makedirs(directory, exist_ok=True)

                    with open(file_path, "a") as file:
                        file.write(
                            (
                                f"epoch {num_epoch} | auroc: {auroc:.5f}, ap_sp: {ap_sp:.5f}, "
                                f"f1_sp: {f1_sp:.5f}, pixel auroc: {pixel_auroc:.5f}, "
                                f"ap_px: {ap_px:.5f}, f1_px: {f1_px:.5f}, aupro_px: {aupro_px:.5f}\n"
                            )
                        )
                ##

    print("class:", class_name)
    print("fix_img_auroc:", fix_auroc)
    print("fix_img_ap:", fix_ap_sp)
    print("fix_img_f1:", fix_f1_sp)
    print("fix_pauroc:", fix_pauroc)
    print("fix_px_ap:", fix_ap_px)
    print("fix_px_f1:", fix_f1_px)
    print("fix_px_aupro:", fix_aupro_px)

    results.append(
        [
            class_name,
            fix_auroc,
            fix_ap_sp,
            fix_f1_sp,
            fix_pauroc,
            fix_ap_px,
            fix_f1_px,
            fix_aupro_px,
        ]
    )
    df = pd.DataFrame(
        results,
        columns=[
            "class",
            "auroc_sp",
            "ap_sp",
            "f1_sp",
            "auroc_px",
            "ap_px",
            "f1_px",
            "aupro_px",
        ],
    )
    result_path = os.path.join(args.save_path, "results", args.subdataset)
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(
        result_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(result_path, exist_ok=True)

    dir = os.path.join(
        result_path,
        "gaussian_"
        + str(args.gaussian)
        + "_noise_"
        + args.noise
        + "_balancing_"
        + str(args.balancing)
        + "_oto_"
        + str(args.kl)
        + "_weight_"
        + str(args.weight)
        + "_synthetic_"
        + str(args.synthetic)
        + "_result.xlsx",
    )
    df.to_excel(dir, index=False)


if __name__ == "__main__":
    main()
