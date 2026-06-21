from abc import ABC, abstractmethod

import numpy as np


class PseudoLabelNormalizer(ABC):
    """归一化抽象基类：将每类的 distance 值映射到 [0, 1] 区间。"""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    @abstractmethod
    def normalize(self, values: np.ndarray) -> np.ndarray:
        ...


class MinMaxNormalizer(PseudoLabelNormalizer):
    """标准 min-max 归一化。"""

    def normalize(self, values: np.ndarray) -> np.ndarray:
        vmin = float(values.min())
        vmax = float(values.max())
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return np.zeros_like(values, dtype=np.float32)
        return ((values - vmin) / (vmax - vmin)).astype(np.float32)


class PercentileNormalizer(PseudoLabelNormalizer):
    """Percentile clipping + min-max 归一化。

    先用高分位数（如 p99）和低分位数（如 p1）裁剪极端离群值，
    再对裁剪后的值做 min-max 映射到 [0, 1]。
    """

    def __init__(self, percentile_range: float = 99.0, eps: float = 1e-6):
        super().__init__(eps=eps)
        self.percentile_range = percentile_range

    def normalize(self, values: np.ndarray) -> np.ndarray:
        p_low = 100.0 - self.percentile_range
        p_high = self.percentile_range
        vmin = float(np.percentile(values, p_low))
        vmax = float(np.percentile(values, p_high))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return np.zeros_like(values, dtype=np.float32)
        clipped = np.clip(values, vmin, vmax)
        return ((clipped - vmin) / (vmax - vmin)).astype(np.float32)


class TopPercentNormalizer(PseudoLabelNormalizer):
    """取最高的 top_ratio 比例的分数做 min-max 归一化。

    1. 从 values 中取出最高的 top_ratio 个分数作为子集
    2. 用这个子集的最小值和最大值对所有 values 做 min-max
    3. 低于子集最小值的 clamp 到 0
    """

    def __init__(self, top_ratio: float = 0.01, eps: float = 1e-6):
        super().__init__(eps=eps)
        self.top_ratio = top_ratio

    def normalize(self, values: np.ndarray) -> np.ndarray:
        n_top = max(1, int(values.size * self.top_ratio))
        top_subset = np.sort(values)[-n_top:]
        vmin = float(top_subset.min())
        vmax = float(top_subset.max())
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            return np.zeros_like(values, dtype=np.float32)
        normalized = (values - vmin) / (vmax - vmin)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)


class PseudoLabelNormalizerFactory:
    """根据配置创建对应的归一化器。"""

    @staticmethod
    def create(args) -> PseudoLabelNormalizer:
        mode = str(getattr(args, "pseudo_label_distance_norm", "minmax")).strip().lower()
        eps = float(getattr(args, "pseudo_label_distance_norm_eps", 1e-6))

        if mode == "minmax":
            return MinMaxNormalizer(eps=eps)
        elif mode == "percentile":
            percentile_range = float(
                getattr(args, "pseudo_label_distance_norm_percentile", 99.0)
            )
            return PercentileNormalizer(percentile_range=percentile_range, eps=eps)
        elif mode == "top_percent":
            top_ratio = float(
                getattr(args, "pseudo_label_distance_norm_top_ratio", 0.01)
            )
            return TopPercentNormalizer(top_ratio=top_ratio, eps=eps)
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")
