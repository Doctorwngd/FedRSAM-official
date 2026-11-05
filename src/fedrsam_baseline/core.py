
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.func import functional_call


TensorMap = Mapping[str, Tensor]
MutableTensorMap = Dict[str, Tensor]
LossFunction = Callable[[object], Tensor]


@dataclass(frozen=True)
class FedRSAMConfig:

    rho: float = 0.1
    alpha_lr: float = 1.0
    z_lr: float = 1e-3
    lambda_z: float = 1.5
    inner_steps: int = 1
    alpha_init: float = 0.1
    eps: float = 1e-12
    radius_mode: str = "boundary"
    project_z_box: bool = True
    regularizer_solver: str = "gradient"

    def __post_init__(self) -> None:
        if self.rho < 0:
            raise ValueError("rho must be non-negative")
        if self.alpha_lr < 0 or self.z_lr < 0:
            raise ValueError("inner learning rates must be non-negative")
        if self.lambda_z < 0:
            raise ValueError("lambda_z must be non-negative")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be at least one")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.radius_mode not in {"boundary", "clip"}:
            raise ValueError("radius_mode must be 'boundary' or 'clip'")
        if self.regularizer_solver not in {"gradient", "proximal"}:
            raise ValueError(
                "regularizer_solver must be 'gradient' or 'proximal'"
            )


@dataclass(frozen=True)
class ConeState:

    names: Tuple[str, ...]
    directions: Mapping[str, Tensor]
    variances: Mapping[str, Tensor]
    raw_global_update: Mapping[str, Tensor]
    participant_count: int
    eps: float = 1e-12

    def validate(self) -> None:
        if not self.names:
            raise ValueError("cone state contains no parameters")
        if self.participant_count < 1:
            raise ValueError("participant_count must be positive")
        for name in self.names:
            if name not in self.directions:
                raise KeyError("missing direction for parameter: {}".format(name))
            if name not in self.variances:
                raise KeyError("missing variance for parameter: {}".format(name))
            if name not in self.raw_global_update:
                raise KeyError("missing raw update for parameter: {}".format(name))
            direction = self.directions[name]
            variance = self.variances[name]
            raw_update = self.raw_global_update[name]
            if direction.shape != variance.shape or direction.shape != raw_update.shape:
                raise ValueError("shape mismatch for parameter: {}".format(name))
            if torch.any(variance < 0):
                raise ValueError("negative variance for parameter: {}".format(name))


@dataclass(frozen=True)
class TrainingConeState:

    names: Tuple[str, ...]
    directions: Mapping[str, Tensor]
    variances: Mapping[str, Tensor]
    eps: float = 1e-12

    def validate(self) -> None:
        if not self.names:
            raise ValueError("training cone contains no parameters")
        for name in self.names:
            if name not in self.directions:
                raise KeyError("missing direction for parameter: {}".format(name))
            if name not in self.variances:
                raise KeyError("missing variance for parameter: {}".format(name))
            direction = self.directions[name]
            variance = self.variances[name]
            if direction.shape != variance.shape:
                raise ValueError("shape mismatch for parameter: {}".format(name))
            if torch.any(variance < 0):
                raise ValueError("negative variance for parameter: {}".format(name))


ConeLike = Union[ConeState, TrainingConeState]


def make_training_cone(cone: ConeState) -> TrainingConeState:

    cone.validate()
    training_cone = TrainingConeState(
        names=cone.names,
        directions=cone.directions,
        variances=cone.variances,
        eps=cone.eps,
    )
    training_cone.validate()
    return training_cone


def prepare_training_cone(
    model: nn.Module,
    cone: ConeLike,
) -> TrainingConeState:

    cone.validate()
    all_named_parameters = dict(model.named_parameters())
    names = []
    directions: MutableTensorMap = {}
    variances: MutableTensorMap = {}
    for name in cone.names:
        parameter = all_named_parameters.get(name)
        if parameter is None or not parameter.requires_grad:
            continue
        if parameter.shape != cone.directions[name].shape:
            continue
        names.append(name)
        directions[name] = cone.directions[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=True,
        )
        variances[name] = cone.variances[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=True,
        )
    if not names:
        raise ValueError("the model has no trainable parameters covered by the cone")
    prepared = TrainingConeState(
        names=tuple(names),
        directions=directions,
        variances=variances,
        eps=cone.eps,
    )
    prepared.validate()
    return prepared


def _eligible_names(
    previous_global: TensorMap,
    current_global: TensorMap,
    client_updates: Sequence[TensorMap],
    parameter_names: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if not client_updates:
        raise ValueError("client_updates must contain at least one client")

    candidate_names = (
        tuple(parameter_names)
        if parameter_names is not None
        else tuple(current_global.keys())
    )
    names = []
    for name in candidate_names:
        if name not in previous_global or name not in current_global:
            continue
        current = current_global[name]
        previous = previous_global[name]
        if not torch.is_floating_point(current):
            continue
        if current.shape != previous.shape:
            continue
        if any(name not in update or update[name].shape != current.shape for update in client_updates):
            continue
        names.append(name)
    return tuple(names)


@torch.no_grad()
def build_cone_state(
    previous_global: TensorMap,
    current_global: TensorMap,
    client_updates: Sequence[TensorMap],
    parameter_names: Optional[Sequence[str]] = None,
    *,
    variance_unbiased: bool = False,
    eps: float = 1e-12,
) -> ConeState:

    if eps <= 0:
        raise ValueError("eps must be positive")
    names = _eligible_names(
        previous_global,
        current_global,
        client_updates,
        parameter_names,
    )
    if not names:
        raise ValueError("no common floating-point parameters were found")

    directions: MutableTensorMap = {}
    variances: MutableTensorMap = {}
    raw_updates: MutableTensorMap = {}

    for name in names:
        raw_update = (
            current_global[name].detach().cpu()
            - previous_global[name].detach().cpu()
        )
        update_norm = raw_update.norm(p=2)
        if float(update_norm.item()) <= eps:
            direction = torch.zeros_like(raw_update)
        else:
            direction = raw_update / update_norm

        stacked_updates = torch.stack(
            [client[name].detach().cpu() for client in client_updates],
            dim=0,
        )
        variance = torch.var(
            stacked_updates,
            dim=0,
            unbiased=variance_unbiased,
        )

        raw_updates[name] = raw_update.contiguous()
        directions[name] = direction.contiguous()
        variances[name] = variance.contiguous()

    state = ConeState(
        names=names,
        directions=directions,
        variances=variances,
        raw_global_update=raw_updates,
        participant_count=len(client_updates),
        eps=eps,
    )
    state.validate()
    return state


def _project_orthogonal(z: Tensor, direction: Tensor) -> Tensor:
    z_flat = z.reshape(-1)
    direction_flat = direction.reshape(-1)
    return (
        z_flat - torch.dot(z_flat, direction_flat) * direction_flat
    ).view_as(z)


def _global_l2_norm(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("cannot compute a norm over an empty tensor sequence")
    return torch.norm(
        torch.stack([tensor.norm(p=2) for tensor in tensors]),
        p=2,
    )


@torch.no_grad()
def aggregate_client_updates(
    current_global: TensorMap,
    client_updates: Sequence[TensorMap],
    client_weights: Optional[Sequence[float]] = None,
    momentum_buffer: Optional[TensorMap] = None,
    server_momentum: float = 0.85,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:

    if not client_updates:
        raise ValueError("client_updates must contain at least one client")
    if not 0.0 <= server_momentum < 1.0:
        raise ValueError("server_momentum must be in [0, 1)")
    if client_weights is None:
        weights = torch.full(
            (len(client_updates),),
            1.0 / float(len(client_updates)),
            dtype=torch.float64,
        )
    else:
        if len(client_weights) != len(client_updates):
            raise ValueError("client_weights length must match client_updates")
        weights = torch.tensor(client_weights, dtype=torch.float64)
        if torch.any(weights < 0):
            raise ValueError("client_weights must be non-negative")
        weight_sum = float(weights.sum().item())
        if weight_sum <= 0:
            raise ValueError("client_weights must have a positive sum")
        weights = weights / weight_sum

    next_global: Dict[str, Tensor] = {}
    aggregate_delta: Dict[str, Tensor] = {}
    next_momentum: Dict[str, Tensor] = {}
    for name, value in current_global.items():
        current = value.detach().cpu()
        if (
            not torch.is_floating_point(current)
            or any(name not in update for update in client_updates)
        ):
            next_global[name] = current.clone()
            continue
        delta = torch.zeros_like(current)
        for weight, update in zip(weights, client_updates):
            client_delta = update[name].detach().cpu()
            if client_delta.shape != current.shape:
                raise ValueError(
                    "client update shape mismatch for: {}".format(name)
                )
            delta.add_(
                client_delta,
                alpha=float(weight.item()),
            )
        aggregate_delta[name] = delta.contiguous()
        previous_velocity = None
        if momentum_buffer is not None and name in momentum_buffer:
            candidate = momentum_buffer[name].detach().cpu()
            if candidate.shape != current.shape:
                raise ValueError(
                    "momentum shape mismatch for: {}".format(name)
                )
            previous_velocity = candidate
        if previous_velocity is None:
            velocity = delta.clone()
        else:
            velocity = (
                server_momentum * previous_velocity + delta
            )
        next_momentum[name] = velocity.contiguous()
        next_global[name] = (current + velocity).contiguous()
    return next_global, aggregate_delta, next_momentum


def solve_restricted_perturbation(
    model: nn.Module,
    loss_function: LossFunction,
    model_args: Sequence[object],
    cone: ConeLike,
    config: FedRSAMConfig,
    model_kwargs: Optional[Mapping[str, object]] = None,
    prepared_cone: Optional[TrainingConeState] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:

    kwargs = dict(model_kwargs or {})
    all_named_parameters = dict(model.named_parameters())
    device_cone = (
        prepared_cone
        if prepared_cone is not None
        else prepare_training_cone(model, cone)
    )
    device_cone.validate()
    constrained = [
        (name, all_named_parameters[name])
        for name in device_cone.names
    ]
    directions = device_cone.directions
    variances = device_cone.variances

    alpha_raw = {
        name: torch.tensor(
            [config.alpha_init],
            device=parameter.device,
            dtype=parameter.dtype,
            requires_grad=True,
        )
        for name, parameter in constrained
    }
    residual_z = {
        name: torch.zeros_like(
            parameter,
            device=parameter.device,
            requires_grad=True,
        )
        for name, parameter in constrained
    }

    last_regularizer = None
    regularizer_path = []
    violation_rate_path = []
    excess_ratio_path = []
    max_ratio_path = []
    correction_ratio_path = []
    proposal_violation_rate_path = []
    proximal_shrinkage_ratio_path = []
    for _ in range(config.inner_steps):
        deltas: Dict[str, Tensor] = {}
        regularizers = []
        for name, _ in constrained:
            std = torch.sqrt(variances[name] + config.eps)
            if config.project_z_box:
                with torch.no_grad():
                    residual_z[name].clamp_(-std, std)
            z_perp = _project_orthogonal(residual_z[name], directions[name])
            deltas[name] = (
                alpha_raw[name] * alpha_raw[name]
            ) * directions[name] + z_perp
            regularizers.append(
                ((residual_z[name] * residual_z[name]) /
                 (variances[name] + config.eps)).mean()
            )

        last_regularizer = torch.stack(regularizers).mean()
        regularizer_path.append(float(last_regularizer.detach().item()))
        parameter_and_buffer = {}
        for name, parameter in model.named_parameters():
            if name in deltas:
                parameter_and_buffer[name] = parameter.detach() + deltas[name]
            else:
                parameter_and_buffer[name] = parameter
        for name, buffer in model.named_buffers():
            parameter_and_buffer[name] = buffer

        output = functional_call(
            model,
            parameter_and_buffer,
            tuple(model_args),
            kwargs,
        )
        inner_objective = loss_function(output)
        if config.regularizer_solver == "gradient":
            inner_objective = (
                inner_objective
                - config.lambda_z * last_regularizer
            )
        variables = (
            [alpha_raw[name] for name, _ in constrained]
            + [residual_z[name] for name, _ in constrained]
        )
        gradients = torch.autograd.grad(
            inner_objective,
            variables,
            create_graph=False,
            allow_unused=False,
        )
        split = len(constrained)
        with torch.no_grad():
            for (name, _), gradient in zip(constrained, gradients[:split]):
                alpha_raw[name].add_(config.alpha_lr * gradient)
            violation_count = 0
            coordinate_count = 0
            excess_ratio_sum = torch.zeros(
                (),
                device=constrained[0][1].device,
                dtype=constrained[0][1].dtype,
            )
            maximum_ratio = torch.zeros_like(excess_ratio_sum)
            correction_squared = torch.zeros_like(excess_ratio_sum)
            residual_squared = torch.zeros_like(excess_ratio_sum)
            proposal_violation_count = 0
            proposal_coordinate_count = 0
            proximal_correction_squared = torch.zeros_like(
                excess_ratio_sum
            )
            proposal_squared = torch.zeros_like(excess_ratio_sum)
            for (name, _), gradient in zip(constrained, gradients[split:]):
                residual_z[name].add_(config.z_lr * gradient)
                std = torch.sqrt(variances[name] + config.eps)
                proposal_ratio = (
                    residual_z[name].abs() / (std + config.eps)
                )
                proposal_violation_count += int(
                    (proposal_ratio > 1.0).sum().item()
                )
                proposal_coordinate_count += proposal_ratio.numel()
                if (
                    config.regularizer_solver == "proximal"
                    and config.lambda_z > 0.0
                ):
                    proposal = residual_z[name].clone()
                    layer_coefficient = (
                        1.0
                        / float(len(constrained))
                        / float(max(residual_z[name].numel(), 1))
                    )
                    denominator = (
                        1.0
                        + (
                            2.0
                            * config.z_lr
                            * config.lambda_z
                            * layer_coefficient
                        )
                        / (variances[name] + config.eps)
                    )
                    residual_z[name].div_(denominator)
                    proximal_correction_squared += torch.sum(
                        (proposal - residual_z[name]) ** 2
                    )
                    proposal_squared += torch.sum(proposal ** 2)
                ratio = residual_z[name].abs() / (std + config.eps)
                excess_ratio = torch.clamp(ratio - 1.0, min=0.0)
                violation_count += int((ratio > 1.0).sum().item())
                coordinate_count += ratio.numel()
                excess_ratio_sum += excess_ratio.sum()
                maximum_ratio = torch.maximum(maximum_ratio, ratio.max())
                clipped = torch.clamp(residual_z[name], -std, std)
                correction_squared += torch.sum(
                    (residual_z[name] - clipped) ** 2
                )
                residual_squared += torch.sum(residual_z[name] ** 2)
                if config.project_z_box:
                    residual_z[name].copy_(clipped)
            proposal_violation_rate_path.append(
                float(proposal_violation_count)
                / float(max(proposal_coordinate_count, 1))
            )
            proximal_shrinkage_ratio_path.append(
                float(
                    torch.sqrt(proximal_correction_squared)
                    / (torch.sqrt(proposal_squared) + config.eps)
                )
                if config.regularizer_solver == "proximal"
                and config.lambda_z > 0.0
                else 0.0
            )
            violation_rate_path.append(
                float(violation_count) / float(max(coordinate_count, 1))
            )
            excess_ratio_path.append(
                float(excess_ratio_sum.item())
                / float(max(coordinate_count, 1))
            )
            max_ratio_path.append(float(maximum_ratio.item()))
            correction_ratio_path.append(
                float(
                    torch.sqrt(correction_squared)
                    / (torch.sqrt(residual_squared) + config.eps)
                )
            )

        alpha_raw = {
            name: value.detach().requires_grad_(True)
            for name, value in alpha_raw.items()
        }
        residual_z = {
            name: value.detach().requires_grad_(True)
            for name, value in residual_z.items()
        }

    final_delta: Dict[str, Tensor] = {}
    main_components = []
    residual_components = []
    saturation_numerator = 0
    saturation_denominator = 0
    final_regularizers = []
    final_violation_count = 0
    final_coordinate_count = 0
    final_excess_ratio_sum = 0.0
    final_max_ratio = 0.0
    for name, _ in constrained:
        std = torch.sqrt(variances[name] + config.eps)
        ratio = residual_z[name].detach().abs() / (std + config.eps)
        excess_ratio = torch.clamp(ratio - 1.0, min=0.0)
        final_violation_count += int((ratio > 1.0).sum().item())
        final_coordinate_count += ratio.numel()
        final_excess_ratio_sum += float(excess_ratio.sum().item())
        final_max_ratio = max(final_max_ratio, float(ratio.max().item()))
        if config.project_z_box:
            with torch.no_grad():
                residual_z[name].clamp_(-std, std)
        z_perp = _project_orthogonal(residual_z[name], directions[name])
        main = (alpha_raw[name] * alpha_raw[name]) * directions[name]
        final_delta[name] = main + z_perp
        main_components.append(main)
        residual_components.append(z_perp)
        final_regularizers.append(
            ((residual_z[name] * residual_z[name]) /
             (variances[name] + config.eps)).mean()
        )
        if config.project_z_box:
            saturation_numerator += int(
                torch.isclose(
                    residual_z[name].abs(),
                    std,
                    rtol=1e-5,
                    atol=1e-8,
                ).sum().item()
            )
            saturation_denominator += residual_z[name].numel()

    raw_norm = _global_l2_norm(list(final_delta.values()))
    if float(raw_norm.item()) <= config.eps or config.rho == 0:
        scale = torch.zeros_like(raw_norm)
    else:
        scale = config.rho / (raw_norm + config.eps)
        if config.radius_mode == "clip":
            scale = torch.clamp(scale, max=1.0)
    final_delta = {
        name: (delta * scale).detach()
        for name, delta in final_delta.items()
    }

    main_norm = _global_l2_norm(main_components)
    residual_norm = _global_l2_norm(residual_components)
    total_component_norm = main_norm + residual_norm
    regularizer_value = (
        float(last_regularizer.detach().item())
        if last_regularizer is not None
        else 0.0
    )
    final_regularizer_value = (
        float(torch.stack(final_regularizers).mean().detach().item())
        if final_regularizers
        else 0.0
    )
    diagnostics = {
        "regularizer": regularizer_value,
        "weighted_regularizer": config.lambda_z * regularizer_value,
        "regularizer_path_mean": (
            sum(regularizer_path) / len(regularizer_path)
            if regularizer_path
            else 0.0
        ),
        "final_regularizer": final_regularizer_value,
        "weighted_final_regularizer": (
            config.lambda_z * final_regularizer_value
        ),
        "preclip_violation_rate_mean": (
            sum(violation_rate_path) / len(violation_rate_path)
            if violation_rate_path
            else 0.0
        ),
        "preclip_violation_rate_max": (
            max(violation_rate_path) if violation_rate_path else 0.0
        ),
        "preclip_excess_ratio_mean": (
            sum(excess_ratio_path) / len(excess_ratio_path)
            if excess_ratio_path
            else 0.0
        ),
        "preclip_max_ratio": (
            max(max_ratio_path) if max_ratio_path else 0.0
        ),
        "virtual_clip_correction_ratio_mean": (
            sum(correction_ratio_path) / len(correction_ratio_path)
            if correction_ratio_path
            else 0.0
        ),
        "loss_proposal_violation_rate_mean": (
            sum(proposal_violation_rate_path)
            / len(proposal_violation_rate_path)
            if proposal_violation_rate_path
            else 0.0
        ),
        "proximal_shrinkage_ratio_mean": (
            sum(proximal_shrinkage_ratio_path)
            / len(proximal_shrinkage_ratio_path)
            if proximal_shrinkage_ratio_path
            else 0.0
        ),
        "final_z_box_violation_rate": (
            float(final_violation_count)
            / float(max(final_coordinate_count, 1))
        ),
        "final_z_box_excess_ratio_mean": (
            final_excess_ratio_sum
            / float(max(final_coordinate_count, 1))
        ),
        "final_z_box_max_ratio": final_max_ratio,
        "raw_perturbation_norm": float(raw_norm.detach().item()),
        "final_perturbation_norm": float(
            _global_l2_norm(list(final_delta.values())).item()
        ),
        "scale": float(scale.detach().item()),
        "main_component_norm": float(main_norm.detach().item()),
        "residual_component_norm": float(residual_norm.detach().item()),
        "residual_component_ratio": float(
            (residual_norm / (total_component_norm + config.eps))
            .detach()
            .item()
        ),
        "z_box_saturation_rate": (
            float(saturation_numerator) / float(saturation_denominator)
            if saturation_denominator
            else 0.0
        ),
    }
    for step_index in range(min(config.inner_steps, 5)):
        diagnostics[
            "inner_regularizer_step_{}".format(step_index)
        ] = regularizer_path[step_index]
        diagnostics[
            "inner_violation_rate_step_{}".format(step_index)
        ] = violation_rate_path[step_index]
        diagnostics[
            "inner_correction_ratio_step_{}".format(step_index)
        ] = correction_ratio_path[step_index]
        diagnostics[
            "inner_loss_proposal_violation_rate_step_{}".format(
                step_index
            )
        ] = proposal_violation_rate_path[step_index]
        diagnostics[
            "inner_proximal_shrinkage_ratio_step_{}".format(step_index)
        ] = proximal_shrinkage_ratio_path[step_index]
    return final_delta, diagnostics


@torch.no_grad()
def apply_perturbations_(
    model: nn.Module,
    perturbations: Mapping[str, Tensor],
) -> Dict[str, Tensor]:

    saved: Dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in perturbations:
            continue
        delta = perturbations[name]
        if delta.shape != parameter.shape:
            raise ValueError("perturbation shape mismatch for: {}".format(name))
        saved[name] = parameter.detach().clone()
        parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
    return saved


@torch.no_grad()
def restore_parameters_(
    model: nn.Module,
    saved_parameters: Mapping[str, Tensor],
) -> None:

    for name, parameter in model.named_parameters():
        if name in saved_parameters:
            parameter.copy_(
                saved_parameters[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


class FedRSAMLocalOptimizer:

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        config: FedRSAMConfig,
        grad_clip_norm: Optional[float] = 10.0,
    ) -> None:
        self.model = model
        self.base_optimizer = base_optimizer
        self.config = config
        self.grad_clip_norm = grad_clip_norm
        self._prepared_cone: Optional[TrainingConeState] = None

    def prepare_cone(self, cone: ConeLike) -> None:

        self._prepared_cone = prepare_training_cone(self.model, cone)

    def step(
        self,
        model_args: Sequence[object],
        loss_function: LossFunction,
        cone: ConeState,
        model_kwargs: Optional[Mapping[str, object]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if self._prepared_cone is None:
            self.prepare_cone(cone)
        perturbations, diagnostics = solve_restricted_perturbation(
            self.model,
            loss_function,
            model_args,
            cone,
            self.config,
            model_kwargs=model_kwargs,
            prepared_cone=self._prepared_cone,
        )
        saved = apply_perturbations_(self.model, perturbations)
        self.base_optimizer.zero_grad()
        try:
            output = self.model(
                *tuple(model_args),
                **dict(model_kwargs or {})
            )
            outer_loss = loss_function(output)
            outer_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )
        finally:
            restore_parameters_(self.model, saved)
        self.base_optimizer.step()
        return outer_loss.detach(), diagnostics
