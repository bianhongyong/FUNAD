import numpy as np
import torch
import torch.nn.functional as F


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
def calculate_log_barrier_bi_occ_loss(features, mask, target=None):
    """
    Calculate Abnormal Invariant OCC loss.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    Aa = A[mask == 1]
    if torch.sum(mask == 1) != 0:  # second stage, and exist anomalies
        r_max = min(0.9 * Aa.min().item(), 0.4)  # get the minimum abnormal radii as r_max
        r_min = r_max * 0.99  
    else:  # first stage, or no anomalies in second stage
        r_max = 0.4
        r_min = 0.99 * 0.4
    
    loss, loss_n, loss_a = 0, 0, 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        An_larger = An[An > r_max]  # larger than r_max
        An_lower = An[An < r_min]  # lower than r_min
        if An_larger.shape[0] != 0:
            weights = torch.exp(An_larger - r_max).detach()
            loss_larger = torch.mean(-F.logsigmoid(-(An_larger - r_max)) * weights)
        else:
            loss_larger = 0
        if An_lower.shape[0] != 0:
            weights = torch.exp(r_min - An_lower).detach()
            loss_lower = torch.mean(-F.logsigmoid(-(r_min - An_lower)) * weights)
        else:
            loss_lower = 0
        
        # another implementation
        # loss_larger = torch.mean(log_sigmoid(-(An - r_max)))  # cooresponding to r_max, pull into r_max
        # loss_lower = torch.mean(log_sigmoid(-(r_min - An)))  # cooresponding to r_min, pull into r_min
        
        loss_n = loss_larger + loss_lower
        loss += loss_n

    # for anomalies, we keep the mapped features as the original features
    if torch.sum(mask == 1) != 0 and target is not None:
        ano_features = features[mask == 1]
        target_features = target[mask == 1]
        loss_mse = F.mse_loss(ano_features, target_features)
        loss_cos = torch.mean(1 - F.cosine_similarity(ano_features, target_features))
        loss_inv = loss_mse + loss_cos  # anomaly invariant loss
        
        boundary = r_max + 0.1
        # using log barrier loss to push ano features out the boundary
        Aa_lower = Aa[Aa < boundary]  # lower than r_min
        if Aa_lower.shape[0] != 0:
            weights = torch.exp(boundary - Aa_lower).detach()
            loss_lower = torch.mean(-F.logsigmoid(-(boundary - Aa_lower)) * weights)
        else:
            loss_lower = 0
        loss_a = loss_inv + loss_lower
        loss += loss_a

    return loss, loss_n.item() if torch.is_tensor(loss_n) else 0, loss_a.item() if torch.is_tensor(loss_a) else 0
