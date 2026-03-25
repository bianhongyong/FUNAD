import dataset_extract
import AnomalyCLIP_lib
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os


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

if __name__ == "__main__":
    main()
