import torch
import argparse
import numpy as np
import random
import torch.backends.cudnn as cudnn
import faiss
import os
import wandb
import sys
import model
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd
import datetime
import inference
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
import dataload
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

warnings.filterwarnings("ignore")

import sys

print(sys.executable)

matplotlib.use("Agg")
# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    parser.add_argument("-r", "--random", type=float, default=0.5)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.9)
    parser.add_argument("-d", "--subdataset", type=str, default="bottle")
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "10%", "20%"],
    )
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--k_number", type=int, default=2)
    parser.add_argument("--llambda", type=float, default=1)
    parser.add_argument("--weight", type=float, default=0)
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
    # Mutual closest pair matching (论文 4.2):
    # 如果 i 的最近邻是 j，且 j 的最近邻也是 i，则记为一对互近邻。
    matching = [[], []]
    for i in range(id.shape[0]):
        if i in matching[1]:
            continue
        if i == id[id[i]]:
            input = i
            target = id[i]

            matching[0].append(input)
            matching[1].append(target)

    return matching


def compute_distance(feature):
    # 在当前 batch 内做 FAISS 最近邻检索，用于 mutual smoothness 配对。
    # k=2 是为了跳过自身匹配（第一个近邻通常是自己）。
    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(), feature.shape[-1], faiss.GpuIndexFlatConfig()
    )
    index.add(feature)

    embedding = np.ascontiguousarray(feature)
    distance, id = index.search(embedding, k=2)

    distance = distance.T
    distance = distance[-1]

    id = id.T[-1]
    distance = np.expand_dims(distance, axis=-1)
    return distance, id


def fix_seed(number):
    np.random.seed(number)
    random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    cudnn.benchmark = False
    cudnn.deterministic = True


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
        train_features = np.load(
            os.path.join(args.feature_path, args.noise, class_name + ".npy")
        )
    else:
        if args.patch:
            train_features = np.load(
                os.path.join(args.feature_path, class_name + "_train_patch.npy")
            )
        else:
            train_features = np.load(
                os.path.join(args.feature_path, class_name + ".npy")
            )

    train_features = train_features.reshape(-1, 784, train_features.shape[-1])
    train_dataset = [train_features[i] for i in range(train_features.shape[0])]
    return train_dataset


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
    indexed_dataset = [
        (torch.as_tensor(train_dataset[i], dtype=torch.float32), i)
        for i in range(len(train_dataset))
    ]

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
    if args.oto_loss == "mae":
        return nn.L1Loss().to(device)
    if args.oto_loss == "mse":
        return nn.MSELoss().to(device)
    return nn.KLDivLoss(reduction="batchmean").to(device)


def evaluate_epoch(localnet, test_loader):
    seg_map = []
    img_map = []
    label_gt = []
    mask_gt = []

    for x, y, mask in test_loader:
        with torch.no_grad():
            localnet.eval()
            x = x.to(device)
            y = y.detach().numpy()
            mask = mask.detach().numpy()
            _, score = localnet(x)
            score = score.detach().cpu().numpy()

            img_score = score.max(axis=1)
            img_map.append(img_score)

            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], (224, 224))
                _map = gaussian_filter(_map, sigma=4)
                seg_map.append(_map)
                label_gt.append(y[i])
                mask_gt.append(mask[i])

    img_map = np.concatenate(img_map, axis=0)
    label_gt = np.array(label_gt)
    seg_map = np.stack(seg_map, axis=0)
    mask_gt = np.stack(mask_gt, axis=0)

    auroc = roc_auc_score(label_gt, img_map)
    pixel_auroc = roc_auc_score(mask_gt.ravel(), seg_map.ravel())
    return auroc, pixel_auroc


def precompute_pseudo_labels(args, localnet, mini_loader):
    memory_bank_start = time.perf_counter()

    feature_stack = []
    score_stack = []
    x_stack = [] if args.beta else None

    with torch.no_grad():
        localnet.eval()
        for mini_x, _ in mini_loader:
            features, score = localnet(mini_x.to(device))
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            features = features.detach().cpu().numpy()
            score = score.detach().cpu().numpy()
            score = score.max(axis=-1)

            if score.shape == ():
                score = np.array([score], dtype=np.float32)

            feature_stack.append(features)
            score_stack.append(score)
            if args.beta:
                x_stack.append(mini_x)

    feature_stack = np.concatenate(feature_stack)
    score_stack = np.concatenate(score_stack)

    score_min = score_stack.min()
    score_max = score_stack.max()
    if score_max == score_min:
        normalized_score = np.zeros_like(score_stack)
    else:
        normalized_score = (score_stack - score_min) / (score_max - score_min)

    normal_indices = np.where(normalized_score < 0.5)[0]
    if args.random < 1:
        sample_num = int(normal_indices.shape[0] * args.random)
        sampled = []
        for _ in range(sample_num):
            selected = np.random.randint(normal_indices.shape[0])
            while normal_indices[selected] in sampled:
                selected = np.random.randint(normal_indices.shape[0])
            sampled.append(normal_indices[selected])
        selected_normal_indices = sampled
    else:
        selected_normal_indices = normal_indices

    normal_features = feature_stack[selected_normal_indices]
    normal_features = normal_features.reshape(-1, normal_features.shape[-1])
    dim = int(normal_features.shape[-1])

    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(),
        dim,
        faiss.GpuIndexFlatConfig(),
    )
    index.add(normal_features)
    memory_bank_time = time.perf_counter() - memory_bank_start

    pseudo_start = time.perf_counter()
    all_features = feature_stack.reshape(-1, dim)
    distance, _ = index.search(np.ascontiguousarray(all_features), k=args.k_number)

    distance[distance < 1e-2] = 0
    if args.k_number == 2:
        same_feature = distance[:, 0] == 0
        distance[same_feature, 0] = distance[same_feature, 1]

    distance = distance[:, 0]
    if distance.max() == distance.min():
        distance = np.zeros(distance.shape)
    else:
        distance = (distance - distance.min()) / (distance.max() - distance.min())

    distance_map = distance.reshape(-1, 784)
    pseudo_label_time = time.perf_counter() - pseudo_start

    confident_features = None
    if args.beta:
        x_stack = torch.concat(x_stack)
        x_stack = x_stack[normalized_score > 0.5]
        x_stack = x_stack.reshape(-1, dim)

        high_score_features = feature_stack[normalized_score > 0.5]
        high_score_features = high_score_features.reshape(-1, dim)

        high_score_distance, _ = index.search(
            np.ascontiguousarray(high_score_features), k=1
        )
        high_score_distance = high_score_distance.T.squeeze()

        confident_anomaly = np.argsort(high_score_distance)[::-1][
            : args.beta_number
        ].tolist()
        confident_features = x_stack[confident_anomaly]

    return distance_map, confident_features, dim, memory_bank_time, pseudo_label_time


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
            precompute_pseudo_labels(args, localnet, mini_loader)
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
                pos_mask = local_label == 1
                neg_mask = local_label == 0

                if pos_mask.any().item():
                    _a_loss = localnet_criterion(
                        gaussian_pred[pos_mask], local_label[pos_mask]
                    )
                else:
                    _a_loss = torch.tensor(0.0, device=local_label.device)

                if neg_mask.any().item():
                    _n_loss = localnet_criterion(
                        gaussian_pred[neg_mask], local_label[neg_mask]
                    )
                else:
                    _n_loss = torch.tensor(0.0, device=local_label.device)
                _loss = _a_loss + _n_loss
            else:
                _loss = localnet_criterion(gaussian_pred, local_label)
        else:
            if args.balancing:
                pos_mask = local_label == 1
                neg_mask = local_label == 0

                if pos_mask.any().item():
                    _a_loss = localnet_criterion(
                        local_pred[pos_mask], local_label[pos_mask]
                    )
                else:
                    _a_loss = torch.tensor(0.0, device=local_label.device)

                if neg_mask.any().item():
                    _n_loss = localnet_criterion(
                        local_pred[neg_mask], local_label[neg_mask]
                    )
                else:
                    _n_loss = torch.tensor(0.0, device=local_label.device)
                _loss = _a_loss + _n_loss
            else:
                _loss = localnet_criterion(local_pred, local_label)

        if (iteration >= args.iter) and args.kl:
            kl_start = time.perf_counter()
            feature_np = (
                batch_feature.detach()
                .cpu()
                .numpy()
                .reshape(-1, batch_feature.shape[-1])
            )
            _, id = compute_distance(feature_np)
            matched_id = find_matching(id)
            if args.synthetic:
                real = args.batch_size * 784
            else:
                real = args.batch_size
            target = local_pred[:real].reshape(-1)[matched_id[0]]
            input = local_pred[:real].reshape(-1)[matched_id[1]]

            if args.perlin:
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
            print(
                "epoch %d | loss: %.6f, bce loss: %.6f, one-to-one loss: %.6f"
                % (epoch + 1, local_loss_value, bce_loss_value, oto_loss_value)
            )
            print(
                "epoch %d | memory bank: %.4fs, pseudo label: %.4fs, kl: %.4fs"
                % (epoch + 1, memory_bank_time, pseudo_label_time, kl_loss_time)
            )

            if (epoch) % args.eval_interval == 0:
                auroc, pixel_auroc = evaluate_epoch(localnet, test_loader)
                num_epoch = epoch + 1

                print(
                    "epoch %d |" % num_epoch,
                    f"auroc: {auroc:.5f}, pxiel auroc: {pixel_auroc:.5f}",
                )

                if args.wandb:
                    wandb.log(
                        {
                            "total loss": local_loss_value,
                            "one-to-one loss": oto_loss_value,
                            "bce loss": bce_loss_value,
                            "image AUC": auroc,
                            "pixel AUC": pixel_auroc,
                        }
                    )

                mean = (auroc + pixel_auroc) / 2

                if epoch == 0:
                    best = mean
                    fix_auroc = auroc
                    fix_pauroc = pixel_auroc
                else:
                    if mean > best:
                        best = mean
                        fix_auroc = auroc
                        fix_pauroc = pixel_auroc
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
                            f"epoch {num_epoch} | auroc: {auroc:.5f}, pixel auroc: {pixel_auroc:.5f}\n"
                        )
                ##

    print("class:", class_name)
    print("fix_img_auroc:", fix_auroc)
    print("fix_pauroc:", fix_pauroc)

    results.append([class_name, fix_auroc, fix_pauroc])
    df = pd.DataFrame(results, columns=["class", "auroc", "pixel_auroc"])
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
