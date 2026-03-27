"""
可视化异常检测结果
布局: 左边原图 | 中间真实mask | 右边预测mask + 异常分数
"""

import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

import model
import dataset_extract


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser("visualize anomaly detection")
    parser.add_argument("--data_path", type=str, 
                        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10")
    parser.add_argument("--save_path", type=str, 
                        default="./visualization_results")
    parser.add_argument("--feature_path", type=str, 
                        default="/media/honeywell/E/bhy/FUNAD/feature/MVTec")
    parser.add_argument("--model_path", type=str, default="/media/honeywell/E/bhy/FUNAD/save_results/muti_class/noisy_20/mvtec/20%/gaussian_True_noise_20%_balancing_True_oto_True_weight_0_multiclass_localnet.pt",
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--class_name", type=str, default="bottle",
                        help="Class name to visualize")
    parser.add_argument("--dataset", type=str, default="mvtec", 
                        choices=["mvtec", "visa"])
    parser.add_argument("--max_images", type=int, default=20,
                        help="Maximum number of images to visualize")
    parser.add_argument("--feature_dim", type=int, default=1536,
                        help="Feature dimension for localnet")
    parser.add_argument("--resize_h", type=int, default=224,
                        help="Resize height for visualization")
    parser.add_argument("--resize_w", type=int, default=224,
                        help="Resize width for visualization")
    parser.add_argument("--gaussian_sigma", type=float, default=4.0,
                        help="Gaussian filter sigma for score map")
    return parser.parse_args()


def denormalize_image(tensor_img):
    """反归一化图像用于显示"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor_img * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def get_test_dataset(args):
    """获取测试数据集"""
    test_set = dataset_extract.MyDataset(
        dataset_path=args.data_path,
        dataset=args.dataset,
        class_name=args.class_name,
        is_train=False,
    )
    return test_set


def load_model(args):
    """加载训练好的模型"""
    localnet = model.localnet(args.feature_dim)
    localnet.load_state_dict(torch.load(args.model_path)['net'])
    localnet = localnet.cuda()
    localnet.eval()
    return localnet


def inference(localnet, test_loader, device='cuda'):
    """推理获取预测分数"""
    seg_maps = []
    img_scores = []
    labels = []
    
    with torch.no_grad():
        for images, y, mask in test_loader:
            images = images.to(device)
            localnet.eval()
            
            features, score = localnet(images)
            score = score.detach().cpu().numpy()
            
            img_score = score.max(axis=-1)
            if images.shape[0] == 1:
                img_score = np.array([img_score])
            img_scores.append(img_score)
            
            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], (args.resize_h, args.resize_w))
                _map = gaussian_filter(_map, sigma=args.gaussian_sigma)
                seg_maps.append(_map)
            
            labels.append(y.numpy())
    
    img_scores = np.concatenate(img_scores, axis=0)
    seg_maps = np.stack(seg_maps, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return seg_maps, img_scores, labels


def visualize_results(args, test_set, seg_maps, img_scores, labels):
    """可视化结果"""
    os.makedirs(args.save_path, exist_ok=True)
    
    good_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    n_good = min(len(good_indices), args.max_images // 2)
    n_anomaly = min(len(anomaly_indices), args.max_images // 2)
    
    selected_indices = np.concatenate([
        good_indices[:n_good] if n_good > 0 else [],
        anomaly_indices[:n_anomaly] if n_anomaly > 0 else []
    ])
    
    if len(selected_indices) == 0:
        print(f"No images found for class {args.class_name}")
        return
    
    n_images = len(selected_indices)
    n_cols = 4
    
    fig, axes = plt.subplots(n_images, n_cols, figsize=(n_cols * 3, n_images * 3.5))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        img, label, mask = test_set[idx]
        
        pred_map = seg_maps[idx]
        img_score = img_scores[idx]
        
        img_denorm = denormalize_image(img)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        mask_np = mask.squeeze().numpy() if mask.dim() > 2 else mask.numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        
        mask_resized = cv2.resize(mask_np, (args.resize_h, args.resize_w))
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original Image\nLabel: {'Normal' if label == 0 else 'Anomaly'}", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_resized, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_map, cmap='jet')
        axes[i, 2].set_title("Predicted Score Map", fontsize=10)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(img_np)
        axes[i, 3].imshow(pred_map, cmap='jet', alpha=0.5)
        axes[i, 3].set_title(f"Overlay\nScore: {img_score:.4f}", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(args.save_path, f"{args.class_name}_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_single_image(args, test_set, seg_maps, img_scores, labels, idx):
    """可视化单张图像的详细结果"""
    img, label, mask = test_set[idx]
    
    pred_map = seg_maps[idx]
    img_score = img_scores[idx]
    
    img_denorm = denormalize_image(img)
    img_np = img_denorm.permute(1, 2, 0).numpy()
    
    mask_np = mask.squeeze().numpy() if mask.dim() > 2 else mask.numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    
    mask_resized = cv2.resize(mask_np, (args.resize_h, args.resize_w))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\nLabel: {'Normal' if label == 0 else 'Anomaly'}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(mask_resized, cmap='gray')
    axes[1].set_title("Ground Truth Mask", fontsize=12)
    axes[1].axis('off')
    
    im = axes[2].imshow(pred_map, cmap='jet')
    axes[2].set_title("Predicted Score Map", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    axes[3].imshow(img_np)
    axes[3].imshow(pred_map, cmap='jet', alpha=0.5)
    axes[3].set_title(f"Overlay\nAnomaly Score: {img_score:.4f}", fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(args.save_path, f"{args.class_name}_idx{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    localnet = load_model(args)
    
    print(f"Loading test dataset: {args.class_name}")
    test_set = get_test_dataset(args)
    
    test_loader = DataLoader(
        test_set, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print("Running inference...")
    seg_maps, img_scores, labels = inference(localnet, test_loader)
    
    print(f"Visualizing results...")
    visualize_results(args, test_set, seg_maps, img_scores, labels)
    
    print("\nAnomaly score statistics:")
    good_scores = img_scores[labels == 0]
    anomaly_scores = img_scores[labels == 1]
    print(f"  Normal samples:   mean={good_scores.mean():.4f}, std={good_scores.std():.4f}")
    print(f"  Anomaly samples:  mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")


if __name__ == "__main__":
    main()
