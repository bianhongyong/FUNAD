import argparse
import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Patch: use NaiveGate instead of SwitchGate as the default MoE gate.
# SwitchGate asserts top_k == 1, which blocks param counting for top_k > 1.
# Since we're only counting parameters (not running forward), NaiveGate works
# for any top_k.
# ---------------------------------------------------------------------------
from src.model.moeblock import layers as _moe_layers
from src.model.moeblock.gates import NaiveGate, SwitchGate

_defaults = list(_moe_layers.FMoE.__init__.__defaults__)
for i, d in enumerate(_defaults):
    if d is SwitchGate:
        _defaults[i] = NaiveGate
        break
_moe_layers.FMoE.__init__.__defaults__ = tuple(_defaults)

from src.model import model

# ---------------------------------------------------------------------------
# DINOv3 embed dims (from torch.hub definitions)
# ---------------------------------------------------------------------------
DINOV3_EMBED_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl14": 1024,
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
    "dinov3_vith16plus": 1536,
    "dinov3_vit7b16": 8192,
}

# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt(n):
    return f"{n / 1e6:<8.2f}M"


# ---------------------------------------------------------------------------
# Model building (mirrors self_train_ad_multiclass_dinov3.py)
# ---------------------------------------------------------------------------
def build_feature_extractor(feature_model: str, device: torch.device):
    entry = feature_model
    local_repo = "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"
    print(f"[backbone] loading {entry} from local repo ({local_repo}) …")
    feat = torch.hub.load(local_repo, entry, source="local", pretrained=True)
    feat = feat.to(device)
    feat.eval()
    return feat


def count_backbone_params(feature_model: str):
    device = torch.device("cpu")
    feat = build_feature_extractor(feature_model, device)
    feat_params = count_params(feat)
    num_blocks = len(getattr(feat, "blocks", []))
    embed_dim = DINOV3_EMBED_DIM.get(feature_model, "?")
    # Clean up to save memory
    del feat
    return feat_params, num_blocks, embed_dim


def count_localnet_params(feature_dim, use_moe, num_expert, top_k, use_cls_token, hard_class_gate):
    net = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=use_moe,
        moe_num_expert=num_expert,
        moe_top_k=top_k,
        moe_use_cls_token=use_cls_token,
        moe_hard_class_gate=hard_class_gate,
    )
    total = count_params(net)
    adaptor = count_params(net.adaptor)

    if use_moe:
        disc = net.discriminator
        disc_total = count_params(disc)
        # MoE layer internals
        moe_layer = disc.moe_layer
        expert = count_params(moe_layer.experts) if hasattr(moe_layer, "experts") else 0
        gate = count_params(moe_layer.gate) if hasattr(moe_layer, "gate") else 0
        shared = disc_total - expert - gate
        return {"total": total, "adaptor": adaptor, "disc_total": disc_total,
                "expert": expert, "gate": gate, "shared_remains": shared}
    else:
        disc = net.discriminator
        disc_total = count_params(disc)
        return {"total": total, "adaptor": adaptor, "disc_total": disc_total,
                "expert": 0, "gate": 0, "shared_remains": disc_total}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("count_dino_localnet_params")
    p.add_argument("--feature_model", type=str, default="dinov3_vitb16",
                   choices=list(DINOV3_EMBED_DIM.keys()),
                   help="DINOv3 variant")
    p.add_argument("--use_moe_discriminator", action="store_true")
    p.add_argument("--moe_num_expert", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--moe_use_cls_token", action="store_true",
                   help="MoE gate uses cls_token as routing input.")
    p.add_argument("--moe_hard_class_gate", action="store_true",
                   help="Fixed class->expert routing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    embed_dim = DINOV3_EMBED_DIM.get(args.feature_model)
    feature_dim = embed_dim  # without cls_token concat
    if args.moe_use_cls_token:
        feature_dim = embed_dim  # cls_token not concatenated to feature in current usage

    print("=" * 70)
    print(f"   Feature model    : {args.feature_model}  (embed_dim={embed_dim})")
    print(f"   localnet feat_dim: {feature_dim}")
    if args.use_moe_discriminator:
        print(f"   MoE discriminator: E={args.moe_num_expert}, top-{args.moe_top_k}, "
              f"cls_token_gate={args.moe_use_cls_token}, hard_class_gate={args.moe_hard_class_gate}")
    else:
        print("   Discriminator    : 3-layer MLP")
    print("=" * 70)

    # ---- backbone ----
    print("\n[Counting backbone parameters …]")
    feat_params, num_blocks, _dim = count_backbone_params(args.feature_model)
    print(f"  Backbone (DINOv3)         : {fmt(feat_params)}  ({num_blocks} blocks)")

    # ---- localnet ----
    local = count_localnet_params(
        feature_dim=feature_dim,
        use_moe=args.use_moe_discriminator,
        num_expert=args.moe_num_expert,
        top_k=args.moe_top_k,
        use_cls_token=args.moe_use_cls_token,
        hard_class_gate=args.moe_hard_class_gate,
    )

    print(f"  localnet (total)          : {fmt(local['total'])}")
    print(f"    ├─ adaptor (1×Linear)    : {fmt(local['adaptor'])}")
    if args.use_moe_discriminator:
        print(f"    └─ MoEDiscriminator     : {fmt(local['disc_total'])}")
        print(f"        ├─ experts          : {fmt(local['expert'])}")
        print(f"        ├─ gate             : {fmt(local['gate'])}")
        print(f"        └─ shared/other     : {fmt(local['shared_remains'])}")
    else:
        print(f"    └─ MLP discriminator    : {fmt(local['disc_total'])}")
        print(f"        (3×Linear: {feature_dim}→1024→128→1)")

    total = feat_params + local["total"]
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL                     : {fmt(total)}")
    print(f"  {'─' * 50}")


if __name__ == "__main__":
    main()
