
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from fedrsam_baseline.core import ConeState


TensorMap = Mapping[str, Tensor]

HISTORY_METRIC_NAMES = (
    "historical_update_norm",
    "applied_update_norm",
    "raw_aggregate_norm",
    "axis_cosine",
    "layer_macro_axis_cosine",
    "raw_axis_cosine",
    "angle_degrees",
    "forward_energy",
    "reversal",
    "axis_ncd",
    "axis_capture",
    "full_ncd",
    "full_capture",
    "full_hit_001",
    "full_hit_005",
    "full_hit_010",
    "shuffled_ncd",
    "shuffled_capture",
    "isotropic_l2_ncd",
    "isotropic_l2_capture",
    "isotropic_volume_ncd",
    "isotropic_volume_capture",
    "raw_full_ncd",
    "raw_full_capture",
    "variance_residual_spearman",
    "top10_residual_energy",
    "variance_temporal_spearman",
)


@dataclass(frozen=True)
class HistoryMetricConfig:

    rho: float = 0.1
    projection_steps: int = 5
    projection_step_size: float = 0.8
    max_rank_samples_per_tensor: int = 4096
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.projection_steps < 1:
            raise ValueError("projection_steps must be positive")
        if not 0.0 < self.projection_step_size <= 1.0:
            raise ValueError("projection_step_size must be in (0, 1]")
        if self.max_rank_samples_per_tensor < 32:
            raise ValueError("max_rank_samples_per_tensor must be at least 32")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def _common_names(
    first: TensorMap,
    second: TensorMap,
    names: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    candidates = tuple(first) if names is None else tuple(names)
    return tuple(
        name
        for name in candidates
        if (
            name in first
            and name in second
            and torch.is_floating_point(first[name])
            and first[name].shape == second[name].shape
        )
    )


def _global_dot(first: TensorMap, second: TensorMap, names: Sequence[str]) -> Tensor:
    value = torch.zeros((), dtype=torch.float64)
    for name in names:
        value += torch.sum(
            first[name].detach().cpu().double()
            * second[name].detach().cpu().double()
        )
    return value


def _global_norm(values: TensorMap, names: Sequence[str]) -> Tensor:
    return torch.sqrt(
        torch.clamp(_global_dot(values, values, names), min=0.0)
    )


def _cosine(
    first: TensorMap,
    second: TensorMap,
    names: Sequence[str],
    eps: float,
) -> float:
    first_norm = _global_norm(first, names)
    second_norm = _global_norm(second, names)
    denominator = first_norm * second_norm
    if float(denominator.item()) <= eps:
        return float("nan")
    value = _global_dot(first, second, names) / denominator
    return float(torch.clamp(value, min=-1.0, max=1.0).item())


def _layer_macro_cosine(
    first: TensorMap,
    second: TensorMap,
    names: Sequence[str],
    eps: float,
) -> float:
    values = []
    for name in names:
        first_flat = first[name].detach().cpu().reshape(-1).double()
        second_flat = second[name].detach().cpu().reshape(-1).double()
        denominator = first_flat.norm() * second_flat.norm()
        if float(denominator.item()) <= eps:
            continue
        values.append(
            float(
                torch.clamp(
                    torch.dot(first_flat, second_flat) / denominator,
                    min=-1.0,
                    max=1.0,
                ).item()
            )
        )
    return sum(values) / len(values) if values else float("nan")


def _scaled_direction(
    update: TensorMap,
    names: Sequence[str],
    radius: float,
    eps: float,
) -> Dict[str, Tensor]:
    norm = _global_norm(update, names)
    if float(norm.item()) <= eps:
        return {name: torch.zeros_like(update[name], device="cpu") for name in names}
    scale = radius / float(norm.item())
    return {
        name: update[name].detach().cpu() * scale
        for name in names
    }


def _orthogonal(value: Tensor, direction: Tensor) -> Tensor:
    value_flat = value.reshape(-1)
    direction_flat = direction.reshape(-1)
    return (
        value_flat
        - torch.dot(value_flat, direction_flat) * direction_flat
    ).view_as(value)


def _bounds(
    variance: Tensor,
    variant: str,
    eps: float,
) -> Tensor:
    bound = torch.sqrt(torch.clamp(variance.detach().cpu(), min=0.0) + eps)
    if variant == "full":
        return bound
    if variant == "shuffled":
        flat = bound.reshape(-1)
        if flat.numel() <= 1:
            return bound
        shift = max(1, int(flat.numel() * 0.38196601125))
        return torch.roll(flat, shifts=shift).view_as(bound)
    if variant == "isotropic_l2":
        matched = float(bound.norm().item()) / max(bound.numel() ** 0.5, 1.0)
        return torch.full_like(bound, matched)
    if variant == "isotropic_volume":
        matched = float(
            torch.exp(torch.mean(torch.log(bound.double() + eps))).item()
        )
        return torch.full_like(bound, matched)
    raise ValueError("unknown bound variant: {}".format(variant))


@torch.no_grad()
def _project_target(
    target: TensorMap,
    cone: ConeState,
    config: HistoryMetricConfig,
    variant: str,
) -> Tuple[float, float]:

    names = tuple(name for name in cone.names if name in target)
    coefficients: Dict[str, Tensor] = {}
    residuals: Dict[str, Tensor] = {}
    bounds: Dict[str, Tensor] = {}

    for name in names:
        value = target[name].detach().cpu()
        direction = cone.directions[name].detach().cpu().to(value.dtype)
        coefficient = torch.clamp(
            torch.dot(value.reshape(-1), direction.reshape(-1)),
            min=0.0,
        )
        coefficients[name] = coefficient
        if variant != "axis":
            bound = _bounds(cone.variances[name], variant, config.eps)
            residual = _orthogonal(
                value - coefficient * direction,
                direction,
            )
            residuals[name] = torch.clamp(residual, -bound, bound)
            bounds[name] = bound

    if variant != "axis":
        for _ in range(config.projection_steps):
            for name in names:
                value = target[name].detach().cpu()
                direction = cone.directions[name].detach().cpu().to(value.dtype)
                residual_perp = _orthogonal(residuals[name], direction)
                candidate = coefficients[name] * direction + residual_perp
                error = candidate - value
                coefficients[name] = torch.clamp(
                    coefficients[name]
                    - config.projection_step_size
                    * torch.dot(error.reshape(-1), direction.reshape(-1)),
                    min=0.0,
                )
                residuals[name] = torch.clamp(
                    residuals[name]
                    - config.projection_step_size
                    * _orthogonal(error, direction),
                    -bounds[name],
                    bounds[name],
                )

    candidate: Dict[str, Tensor] = {}
    for name in names:
        direction = cone.directions[name].detach().cpu().to(target[name].dtype)
        if variant == "axis":
            candidate[name] = coefficients[name] * direction
        else:
            candidate[name] = (
                coefficients[name] * direction
                + _orthogonal(residuals[name], direction)
            )

    candidate_norm = _global_norm(candidate, names)
    if float(candidate_norm.item()) > config.rho:
        scale = config.rho / float(candidate_norm.item())
        candidate = {name: value * scale for name, value in candidate.items()}

    squared_distance = torch.zeros((), dtype=torch.float64)
    for name in names:
        difference = (
            target[name].detach().cpu().double()
            - candidate[name].detach().cpu().double()
        )
        squared_distance += torch.sum(difference * difference)
    distance = float(torch.sqrt(torch.clamp(squared_distance, min=0.0)).item())
    ncd = distance / config.rho
    capture = max(0.0, 1.0 - ncd * ncd)
    return ncd, capture


def _sample_flat(value: Tensor, limit: int) -> Tensor:
    flat = value.detach().cpu().reshape(-1).double()
    if flat.numel() <= limit:
        return flat
    stride = max(1, flat.numel() // limit)
    return flat[::stride][:limit]


def _rankdata(value: Tensor) -> Tensor:
    order = torch.argsort(value, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float64)
    return ranks


def _spearman(first: Tensor, second: Tensor, eps: float) -> float:
    if first.numel() < 3 or second.numel() != first.numel():
        return float("nan")
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = first_rank.norm() * second_rank.norm()
    if float(denominator.item()) <= eps:
        return float("nan")
    return float(
        torch.clamp(
            torch.dot(first_rank, second_rank) / denominator,
            min=-1.0,
            max=1.0,
        ).item()
    )


@torch.no_grad()
def _variance_residual_metrics(
    target: TensorMap,
    cone: ConeState,
    config: HistoryMetricConfig,
) -> Tuple[float, float]:
    correlations = []
    top_energy = []
    for name in cone.names:
        if name not in target:
            continue
        value = target[name].detach().cpu()
        direction = cone.directions[name].detach().cpu().to(value.dtype)
        coefficient = torch.clamp(
            torch.dot(value.reshape(-1), direction.reshape(-1)),
            min=0.0,
        )
        residual = value - coefficient * direction
        variance_sample = _sample_flat(
            cone.variances[name],
            config.max_rank_samples_per_tensor,
        )
        residual_sample = _sample_flat(
            residual * residual,
            config.max_rank_samples_per_tensor,
        )
        correlation = _spearman(
            torch.log(variance_sample + config.eps),
            torch.log(residual_sample + config.eps),
            config.eps,
        )
        if correlation == correlation:
            correlations.append(correlation)

        count = residual_sample.numel()
        if count:
            top_count = max(1, int(math_ceil(0.1 * count)))
            top_indices = torch.topk(
                variance_sample,
                k=top_count,
                largest=True,
            ).indices
            total = float(residual_sample.sum().item())
            if total > config.eps:
                top_energy.append(
                    float(residual_sample[top_indices].sum().item()) / total
                )

    return (
        sum(correlations) / len(correlations)
        if correlations
        else float("nan"),
        sum(top_energy) / len(top_energy)
        if top_energy
        else float("nan"),
    )


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1


@torch.no_grad()
def _variance_stability(
    previous: ConeState,
    current: Optional[ConeState],
    config: HistoryMetricConfig,
) -> float:
    if current is None:
        return float("nan")
    values = []
    for name in previous.names:
        if name not in current.variances:
            continue
        first = _sample_flat(
            previous.variances[name],
            config.max_rank_samples_per_tensor,
        )
        second = _sample_flat(
            current.variances[name],
            config.max_rank_samples_per_tensor,
        )
        correlation = _spearman(
            torch.log(first + config.eps),
            torch.log(second + config.eps),
            config.eps,
        )
        if correlation == correlation:
            values.append(correlation)
    return sum(values) / len(values) if values else float("nan")


@torch.no_grad()
def compute_history_metrics(
    previous_cone: ConeState,
    applied_update: TensorMap,
    raw_aggregate_update: TensorMap,
    current_cone: Optional[ConeState],
    config: HistoryMetricConfig,
) -> Dict[str, float]:

    previous_cone.validate()
    names = _common_names(
        previous_cone.raw_global_update,
        applied_update,
        previous_cone.names,
    )
    if not names:
        raise ValueError("no common parameters for historical metrics")

    applied_norm = float(_global_norm(applied_update, names).item())
    raw_norm = float(_global_norm(raw_aggregate_update, names).item())
    historical_norm = float(
        _global_norm(previous_cone.raw_global_update, names).item()
    )
    axis_cosine = _cosine(
        previous_cone.raw_global_update,
        applied_update,
        names,
        config.eps,
    )
    raw_axis_cosine = _cosine(
        previous_cone.raw_global_update,
        raw_aggregate_update,
        names,
        config.eps,
    )
    layer_axis_cosine = _layer_macro_cosine(
        previous_cone.raw_global_update,
        applied_update,
        names,
        config.eps,
    )
    target = _scaled_direction(
        applied_update,
        names,
        config.rho,
        config.eps,
    )
    raw_target = _scaled_direction(
        raw_aggregate_update,
        names,
        config.rho,
        config.eps,
    )

    axis_ncd, axis_capture = _project_target(
        target,
        previous_cone,
        config,
        "axis",
    )
    full_ncd, full_capture = _project_target(
        target,
        previous_cone,
        config,
        "full",
    )
    shuffled_ncd, shuffled_capture = _project_target(
        target,
        previous_cone,
        config,
        "shuffled",
    )
    isotropic_l2_ncd, isotropic_l2_capture = _project_target(
        target,
        previous_cone,
        config,
        "isotropic_l2",
    )
    isotropic_volume_ncd, isotropic_volume_capture = _project_target(
        target,
        previous_cone,
        config,
        "isotropic_volume",
    )
    full_ncd = min(full_ncd, axis_ncd)
    shuffled_ncd = min(shuffled_ncd, axis_ncd)
    isotropic_l2_ncd = min(isotropic_l2_ncd, axis_ncd)
    isotropic_volume_ncd = min(isotropic_volume_ncd, axis_ncd)
    full_capture = max(0.0, 1.0 - full_ncd * full_ncd)
    shuffled_capture = max(0.0, 1.0 - shuffled_ncd * shuffled_ncd)
    isotropic_l2_capture = max(
        0.0,
        1.0 - isotropic_l2_ncd * isotropic_l2_ncd,
    )
    isotropic_volume_capture = max(
        0.0,
        1.0 - isotropic_volume_ncd * isotropic_volume_ncd,
    )
    raw_full_ncd, raw_full_capture = _project_target(
        raw_target,
        previous_cone,
        config,
        "full",
    )
    variance_residual_correlation, top10_residual_energy = (
        _variance_residual_metrics(target, previous_cone, config)
    )

    if axis_cosine == axis_cosine:
        angle_degrees = float(
            torch.rad2deg(
                torch.acos(
                    torch.tensor(axis_cosine, dtype=torch.float64)
                )
            ).item()
        )
        forward_energy = max(axis_cosine, 0.0) ** 2
        reversal = float(axis_cosine < 0.0)
    else:
        angle_degrees = float("nan")
        forward_energy = float("nan")
        reversal = float("nan")

    result = {
        "historical_update_norm": historical_norm,
        "applied_update_norm": applied_norm,
        "raw_aggregate_norm": raw_norm,
        "axis_cosine": axis_cosine,
        "layer_macro_axis_cosine": layer_axis_cosine,
        "raw_axis_cosine": raw_axis_cosine,
        "angle_degrees": angle_degrees,
        "forward_energy": forward_energy,
        "reversal": reversal,
        "axis_ncd": axis_ncd,
        "axis_capture": axis_capture,
        "full_ncd": full_ncd,
        "full_capture": full_capture,
        "full_hit_001": float(full_ncd <= 0.01),
        "full_hit_005": float(full_ncd <= 0.05),
        "full_hit_010": float(full_ncd <= 0.10),
        "shuffled_ncd": shuffled_ncd,
        "shuffled_capture": shuffled_capture,
        "isotropic_l2_ncd": isotropic_l2_ncd,
        "isotropic_l2_capture": isotropic_l2_capture,
        "isotropic_volume_ncd": isotropic_volume_ncd,
        "isotropic_volume_capture": isotropic_volume_capture,
        "raw_full_ncd": raw_full_ncd,
        "raw_full_capture": raw_full_capture,
        "variance_residual_spearman": variance_residual_correlation,
        "top10_residual_energy": top10_residual_energy,
        "variance_temporal_spearman": _variance_stability(
            previous_cone,
            current_cone,
            config,
        ),
    }
    if tuple(result) != HISTORY_METRIC_NAMES:
        raise RuntimeError("history metric schema drifted")
    return result
