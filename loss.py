import numpy as np
import torch


def compute_balanced_bce_loss(localnet_criterion, pred, target):
    pos_mask = target == 1
    neg_mask = target == 0

    if pos_mask.any().item():
        pos_loss = localnet_criterion(pred[pos_mask], target[pos_mask])
    else:
        pos_loss = torch.tensor(0.0, device=target.device)

    if neg_mask.any().item():
        neg_loss = localnet_criterion(pred[neg_mask], target[neg_mask])
    else:
        neg_loss = torch.tensor(0.0, device=target.device)

    return pos_loss + neg_loss


def compute_oto_loss_multiclass(
    args,
    batch_feature,
    local_pred,
    class_idx,
    l_loss,
    compute_distance_fn,
    find_matching_fn,
):
    if args.oto_loss == "kl":
        transform_fn = lambda a, b: 0.5 * (
            l_loss(a.log(), (a + b) / 2) + l_loss(b.log(), (a + b) / 2)
        )
    else:
        transform_fn = lambda a, b: 0.5 * (
            l_loss(a, (a + b) / 2) + l_loss(b, (a + b) / 2)
        )

    unique_cls = torch.unique(class_idx)
    loss_list = []

    for cls in unique_cls:
        cls_mask = class_idx == cls
        if cls_mask.sum().item() < 2:
            continue

        cls_feature = batch_feature[cls_mask]
        cls_score = local_pred[cls_mask]
        feature_np = cls_feature.detach().cpu().numpy().reshape(-1, cls_feature.shape[-1])
        _, id_array = compute_distance_fn(feature_np)
        matched = find_matching_fn(id_array)
        if len(matched[0]) == 0:
            continue

        target = cls_score.reshape(-1)[matched[0]]
        input_score = cls_score.reshape(-1)[matched[1]]
        loss_list.append(transform_fn(input_score, target))

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=local_pred.device)

    return torch.stack(loss_list).mean()


def compute_oto_loss_single(
    args,
    batch_feature,
    local_pred,
    real_count,
    l_loss,
    compute_distance_fn,
    find_matching_fn,
):
    feature_np = batch_feature.detach().cpu().numpy().reshape(-1, batch_feature.shape[-1])
    _, id_array = compute_distance_fn(feature_np)
    matched_id = find_matching_fn(id_array)
    target = local_pred[:real_count].reshape(-1)[matched_id[0]]
    input_score = local_pred[:real_count].reshape(-1)[matched_id[1]]

    if args.oto_loss == "kl":
        return 0.5 * (
            l_loss(input_score.log(), (input_score + target) / 2)
            + l_loss(target.log(), (input_score + target) / 2)
        )
    return 0.5 * (
        l_loss(input_score, (input_score + target) / 2)
        + l_loss(target, (input_score + target) / 2)
    )


def compute_origin_regularizer(
    args,
    input_feature,
    output_feature,
    normal_mask,
    anomaly_mask,
):
    input_2d = input_feature.reshape(-1, input_feature.shape[-1])
    output_2d = output_feature.reshape(-1, output_feature.shape[-1])

    normal_loss = torch.tensor(0.0, device=output_feature.device)
    if normal_mask.any().item():
        normal_feat = output_2d[normal_mask]
        normal_loss = (normal_feat.pow(2).sum(dim=-1)).mean()

    anomaly_loss = torch.tensor(0.0, device=output_feature.device)
    if anomaly_mask.any().item():
        anomaly_in = input_2d[anomaly_mask]
        anomaly_out = output_2d[anomaly_mask]
        anomaly_loss = torch.mean((anomaly_out - anomaly_in) ** 2)

    return (
        args.origin_normal_weight * normal_loss
        + args.origin_anomaly_weight * anomaly_loss
    )


def build_adaptive_threshold_map(distance, class_idx_np, default_threshold, quantile):
    threshold_map = np.full_like(distance, fill_value=default_threshold, dtype=np.float32)
    safe_q = min(max(float(quantile), 0.0), 1.0)

    for cls in np.unique(class_idx_np).tolist():
        cls_mask = class_idx_np == int(cls)
        if not np.any(cls_mask):
            continue
        cls_values = distance[cls_mask].reshape(-1)
        if cls_values.size == 0:
            continue
        cls_thr = float(np.quantile(cls_values, safe_q))
        threshold_map[cls_mask] = cls_thr

    return threshold_map
