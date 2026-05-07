import argparse

import utils.train_utils as common_utils


def str2bool(value):
    return common_utils.str2bool(value)


def add_io_args(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument("--save_path", type=str, default="/media/honeywell/E/bhy/test")


def add_dataset_args(parser):
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "15%", "10%", "20%"],
    )


def add_training_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--gaussian", action="store_false")
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--alternative", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true",
                        help="Enable autograd anomaly detection (slows training, use for debugging only)")
def add_threshold_args(parser):
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.995)
    parser.add_argument("--beta_number", type=int, default=15)
def add_plot_args(parser):
    parser.add_argument("--hist", action="store_true")

def add_loss_args(parser):
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--weight", type=float, default=0)
    parser.add_argument("--balancing", action="store_false")
    parser.add_argument("--oto_loss", type=str, choices=["kl", "mae", "mse"], default="mae")


def add_feature_extractor_args(parser, feature_model_choices):
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument(
        "--feature_model",
        type=str,
        choices=tuple(feature_model_choices),
        default="dinov3_vitb16",
        help="DINOv3 variant (torch.hub entry); see DINOV3_FEATURE_MODEL_REGISTRY for hub_entry / hub_repo_dir.",
    )
    parser.add_argument(
        "--dino_layer_indices",
        type=int,
        nargs="+",
        default=[22, 23, 24, 25, 26, 27, 28],
        help="DINOv3 layer ids (1-based, as in paper) to aggregate by mean pooling.",
    )


def add_reference_args(parser):
    parser.add_argument("--num_reference_images_per_class", type=int, default=4)
    parser.add_argument("--strict_clean_reference", action="store_true")


def add_faiss_args(parser):
    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)


def add_moe_args(parser):
    parser.add_argument(
        "--gate_aux_weight",
        type=float,
        default=0.05,
        help="Weight for FMoE gate auxiliary loss from discriminator.",
    )
    parser.add_argument(
        "--use_moe_discriminator",
        action="store_true",
        help="Whether to use MoE discriminator for localnet.",
    )
    parser.add_argument("--use_cls_token", type=str2bool, default="False")
    parser.add_argument(
        "--moe_num_expert",
        type=int,
        default=4,
        help="Number of experts used when MoE discriminator is enabled.",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=2,
        help="Top-k experts per token when MoE discriminator is enabled.",
    )
    parser.add_argument(
        "--moe_use_cls_token",
        action="store_true",
        help="Whether MoE discriminator gate uses cls_token as routing input.",
    )
    parser.add_argument(
        "--moe_hard_class_gate",
        action="store_true",
        help="Use fixed class->expert routing gate (expert count follows class count).",
    )
    parser.add_argument(
        "--moe_expert_vis_enable",
        action="store_true",
        help="Enable MoE class-to-expert routing visualization.",
    )
    parser.add_argument(
        "--moe_expert_vis_interval",
        type=int,
        default=1,
        help="Save MoE class-to-expert visualization every N epochs.",
    )
    parser.add_argument(
        "--moe_expert_vis_dirname",
        type=str,
        default="moe_expert_vis",
        help="Sub-directory name for MoE class-to-expert visualization outputs.",
    )


def add_pseudo_label_args(parser):
    parser.add_argument(
        "--memory_bank_score_quantile",
        type=float,
        default=0.3,
        help=(
            "Per-class (multiclass) or global (legacy feature precompute): fraction of images "
            "with lowest normalized image-level scores used as memory-bank normal candidates."
        ),
    )
    parser.add_argument("--k_number", type=int, default=2)
    parser.add_argument(
        "--pseudo_label_scoring",
        type=str,
        choices=["nn", "mahalanobis", "blend", "pca"],
        default="nn",
        help=(
            "Patch vs. memory-bank score: k-NN, Mahalanobis, PCA reconstruction residual, "
            "or per-class min-max normalized blend of k-NN and Mahalanobis."
        ),
    )
    parser.add_argument(
        "--pseudo_label_blend_nn_weight",
        type=float,
        default=0.5,
        help=(
            "blend mode: weight for k-NN branch (Mahalanobis weight defaults complement; "
            "both renormalized to sum to 1)."
        ),
    )
    parser.add_argument(
        "--pseudo_label_blend_maha_weight",
        type=float,
        default=0.5,
        help="blend mode: weight for Mahalanobis branch.",
    )
    parser.add_argument(
        "--pseudo_label_mahalanobis_dim",
        type=int,
        default=128,
        help=(
            "Mahalanobis mode: project patch features to this dim (bias-free Linear) before "
            "Gaussian fit and scoring when feature dim is larger; set 0 to disable projection."
        ),
    )
    parser.add_argument(
        "--pseudo_label_pca_dim",
        type=int,
        default=0,
        help="PCA mode: retained principal components; set 0 to auto-select by explained variance.",
    )
    parser.add_argument(
        "--pseudo_label_pca_ev",
        type=float,
        default=0.99,
        help="PCA mode: explained variance target used when --pseudo_label_pca_dim=0.",
    )
    parser.add_argument(
        "--pseudo_label_pca_eps",
        type=float,
        default=1e-6,
        help="PCA mode: numerical stability epsilon for eigenvalue clamping and ratio computation.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm",
        type=str,
        choices=["minmax", "robust_mad"],
        default="minmax",
        help="Per-class patch-distance normalization: min-max or robust median+MAD.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm_eps",
        type=float,
        default=1e-6,
        help="Numerical stability epsilon for patch-distance normalization.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm_robust_scale",
        type=float,
        default=1.4826,
        help="Robust MAD scale factor used in robust_mad normalization.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm_robust_clip",
        type=float,
        default=3.0,
        help="Clip range for robust z-score before mapping to [0, 1].",
    )
    parser.add_argument(
        "--greedy_keep_images",
        type=int,
        default=2,
        help="Greedy coreset keeps feature points equivalent to this many images per class.",
    )
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
        help="Image-level score uses mean of top-k patch scores, k=ceil(num_patches*ratio).",
    )


def parse_args(feature_model_choices):
    parser = argparse.ArgumentParser("self-train_ad_multiclass")
    add_io_args(parser)
    add_dataset_args(parser)
    add_training_args(parser)
    add_plot_args(parser)
    add_loss_args(parser)
    add_threshold_args(parser)
    add_feature_extractor_args(parser, feature_model_choices)
    add_reference_args(parser)
    add_faiss_args(parser)
    add_moe_args(parser)
    add_pseudo_label_args(parser)
    return parser.parse_args()
