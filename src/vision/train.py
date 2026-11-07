
import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from pathlib import Path


SEARCH_DIAGNOSTIC_NAMES = (
    "regularizer",
    "weighted_regularizer",
    "regularizer_path_mean",
    "final_regularizer",
    "weighted_final_regularizer",
    "preclip_violation_rate_mean",
    "preclip_violation_rate_max",
    "preclip_excess_ratio_mean",
    "preclip_max_ratio",
    "virtual_clip_correction_ratio_mean",
    "loss_proposal_violation_rate_mean",
    "proximal_shrinkage_ratio_mean",
    "final_z_box_violation_rate",
    "final_z_box_excess_ratio_mean",
    "final_z_box_max_ratio",
    "raw_perturbation_norm",
    "scale",
    "main_component_norm",
    "residual_component_norm",
    "z_box_saturation_rate",
    "inner_regularizer_step_0",
    "inner_regularizer_step_1",
    "inner_regularizer_step_2",
    "inner_correction_ratio_step_0",
    "inner_correction_ratio_step_1",
    "inner_correction_ratio_step_2",
    "inner_violation_rate_step_0",
    "inner_violation_rate_step_1",
    "inner_violation_rate_step_2",
    "inner_loss_proposal_violation_rate_step_0",
    "inner_loss_proposal_violation_rate_step_1",
    "inner_loss_proposal_violation_rate_step_2",
    "inner_proximal_shrinkage_ratio_step_0",
    "inner_proximal_shrinkage_ratio_step_1",
    "inner_proximal_shrinkage_ratio_step_2",
)

SEARCH_DISTRIBUTION_BASES = (
    "final_regularizer",
    "preclip_violation_rate_mean",
    "preclip_max_ratio",
    "virtual_clip_correction_ratio_mean",
    "loss_proposal_violation_rate_mean",
    "proximal_shrinkage_ratio_mean",
    "final_z_box_violation_rate",
    "final_z_box_max_ratio",
    "residual_component_ratio",
)

SEARCH_DISTRIBUTION_NAMES = tuple(
    "{}_{}".format(name, suffix)
    for name in SEARCH_DISTRIBUTION_BASES
    for suffix in ("q25", "median", "q75")
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="FedRSAM vision history experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alg",
        choices=("FedRSAM", "NSAMFed", "FedSAM"),
        default="FedRSAM",
        help="NSAMFed is accepted as the original draw3 name for FedRSAM.",
    )
    parser.add_argument(
        "--protocol",
        choices=("clean", "original_draw3"),
        default="clean",
        help="Training-system compatibility preset.",
    )
    parser.add_argument("--data_name", choices=("CIFAR100", "CIFAR10"), default="CIFAR100")
    parser.add_argument(
        "--model",
        "--CNN",
        dest="model",
        choices=("resnet18", "mobilenetv2", "lenet5"),
        default="resnet18",
    )
    parser.add_argument("--normalization", choices=("GN", "BN"), default="GN")
    parser.add_argument("--data_root", "--datapath", dest="data_root", default="./data")
    parser.add_argument(
        "--preprocessing_profile",
        choices=("fedrsam_legacy", "fedgloss_cifar100"),
        default="fedrsam_legacy",
        help=(
            "fedgloss_cifar100 uses FedGloSS CIFAR-100 normalization and "
            "deterministic test preprocessing."
        ),
    )
    parser.add_argument("--rounds", "--epoch", dest="rounds", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=100)
    parser.add_argument("--selection", type=float, default=0.1)
    parser.add_argument("--local_epochs", "--E", dest="local_epochs", type=int, default=5)
    parser.add_argument("--max_steps", "--K", dest="max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.998)
    parser.add_argument(
        "--lr_schedule",
        choices=("exponential", "cosine"),
        default=None,
        help="Defaults to cosine for original_draw3 and exponential otherwise.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument(
        "--rho_schedule",
        choices=("constant", "linear"),
        default="constant",
    )
    parser.add_argument("--rho_start", type=float, default=None)
    parser.add_argument("--rho_schedule_rounds", type=int, default=0)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--alpha_lr", type=float, default=0.5)
    parser.add_argument(
        "--alpha_scope",
        choices=("tensor", "global"),
        default="tensor",
        help=(
            "Use one learned alpha per parameter tensor (submitted method) "
            "or one scalar shared by the full model axis."
        ),
    )
    parser.add_argument(
        "--cone_scope",
        choices=("tensor", "global"),
        default="tensor",
        help=(
            "Normalize the historical axis and impose residual "
            "orthogonality independently per parameter tensor, or over "
            "the concatenated full-model parameter vector."
        ),
    )
    parser.add_argument("--z_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_z", type=float, default=0.8)
    parser.add_argument(
        "--disable_residual",
        action="store_true",
        help=(
            "Set the orthogonal cone residual z exactly to zero while "
            "retaining the learned per-tensor alpha update."
        ),
    )
    parser.add_argument(
        "--fixed_axis_until",
        type=int,
        default=0,
        help=(
            "Use the direct fixed-axis perturbation for round indices below "
            "this value, then switch to the configured closed-form cone "
            "solver without resetting the model, server momentum, cone, or "
            "learning-rate schedule."
        ),
    )
    parser.add_argument(
        "--regularizer_solver",
        choices=("gradient", "proximal", "closed_form"),
        default="gradient",
    )
    parser.add_argument(
        "--skip_zero_init_single_step_regularizer",
        action="store_true",
        help=(
            "Skip the mathematically inactive z regularizer when inner_steps=1, "
            "z starts at zero, and the gradient solver is used."
        ),
    )
    parser.add_argument(
        "--restricted_impl",
        choices=("reference", "fast"),
        default="reference",
        help=(
            "reference retains the fully instrumented functional-call solver; "
            "fast uses the mathematically equivalent closed-form update for "
            "the one-step, zero-initialized setting."
        ),
    )
    parser.add_argument(
        "--search_diagnostics",
        choices=("full", "minimal"),
        default="full",
        help=(
            "Per-mini-batch cone-search diagnostics. The fast implementation "
            "requires minimal to avoid GPU synchronization overhead."
        ),
    )
    parser.add_argument("--alpha_init", type=float, default=0.1)
    parser.add_argument(
        "--client_lookahead_gamma",
        "--gamma",
        dest="client_lookahead_gamma",
        type=float,
        default=0.0,
        help=(
            "Previous server velocity added to the client starting model. "
            "This is distinct from server_momentum."
        ),
    )
    parser.add_argument(
        "--warmup_rho",
        type=float,
        default=None,
        help="SAM radius before the first historical cone is available.",
    )
    parser.add_argument(
        "--project_z_box",
        dest="project_z_box",
        action="store_true",
    )
    parser.add_argument(
        "--no_project_z_box",
        dest="project_z_box",
        action="store_false",
    )
    parser.set_defaults(project_z_box=None)
    parser.add_argument("--dirichlet_alpha", "--alpha_value", dest="dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_gpus_per_client", "--num_gpus_per", dest="num_gpus_per_client", type=float, default=0.5)
    parser.add_argument(
        "--ray_temp_dir",
        default=None,
        help="Short isolated Ray session directory for concurrent runs.",
    )
    parser.add_argument(
        "--serial_clients",
        action="store_true",
        help=(
            "Run selected clients sequentially in the driver process. This "
            "avoids starting a second Ray runtime for concurrent GPU runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_every",
        "--preprint",
        dest="eval_every",
        type=int,
        default=10,
    )
    parser.add_argument("--eval_batch_size", type=int, default=200)
    parser.add_argument("--num_eval_workers", type=int, default=4)
    parser.add_argument(
        "--deterministic_test",
        action="store_true",
        help=(
            "Disable the legacy RandomCrop on the test set while retaining "
            "the original_draw3 training protocol."
        ),
    )
    parser.add_argument("--geometry_every", type=int, default=1)
    parser.add_argument("--geometry_projection_steps", type=int, default=5)
    parser.add_argument("--geometry_rank_samples", type=int, default=4096)
    parser.add_argument(
        "--history_profile",
        choices=("off", "full"),
        default="full",
        help="Disable passive metrics for a timing/accuracy-only baseline.",
    )
    parser.add_argument("--output_dir", default="./results/fedrsam_vision_history")
    parser.add_argument("--run_name", "--extname", dest="run_name", default="baseline")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--print", dest="legacy_print", type=int, default=0)
    parser.add_argument("--pix", type=float, default=32.0)
    parser.add_argument("--lora", type=int, default=0)
    parser.add_argument("--save_best", dest="save_best", action="store_true")
    parser.add_argument("--no_save_best", dest="save_best", action="store_false")
    parser.set_defaults(
        protocol="original_draw3",
        alpha_scope="global",
        cone_scope="global",
        regularizer_solver="closed_form",
        restricted_impl="fast",
        search_diagnostics="minimal",
        deterministic_test=True,
        history_profile="off",
        project_z_box=True,
        save_best=True,
        server_momentum=0.85,
        normalized_server_momentum=False,
    )
    return parser


def apply_protocol_defaults(args):
    if args.alg == "NSAMFed":
        args.alg = "FedRSAM"
    if args.lr_schedule is None:
        args.lr_schedule = (
            "cosine"
            if args.protocol == "original_draw3"
            else "exponential"
        )
    if args.warmup_rho is None:
        args.warmup_rho = (
            0.05
            if args.protocol == "original_draw3"
            else args.rho
        )
    if args.project_z_box is None:
        args.project_z_box = args.protocol != "original_draw3"
    return args


def validate_args(args):
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.num_workers < 1:
        raise ValueError("num_workers must be positive")
    if not 0.0 < args.selection <= 1.0:
        raise ValueError("selection must be in (0, 1]")
    if int(args.num_workers * args.selection) < 1:
        raise ValueError("num_workers * selection selects no clients")
    if args.local_epochs < 1 or args.max_steps < 1:
        raise ValueError("local_epochs and max_steps must be positive")
    if args.rho <= 0:
        raise ValueError("rho must be positive")
    if args.rho_schedule == "linear":
        if args.rho_start is None or args.rho_start <= 0:
            raise ValueError("linear rho schedule requires rho_start > 0")
        if args.rho_schedule_rounds < 1:
            raise ValueError(
                "linear rho schedule requires rho_schedule_rounds >= 1"
            )
    if args.alg in {"FedRSAM", "NSAMFed"} and not 0.0 <= args.server_momentum < 1.0:
        raise ValueError("server_momentum must be in [0, 1)")
    if args.client_lookahead_gamma < 0.0:
        raise ValueError("client_lookahead_gamma must be non-negative")
    if args.warmup_rho <= 0.0:
        raise ValueError("warmup_rho must be positive")
    if args.eval_every < 1 or args.geometry_every < 1:
        raise ValueError("logging intervals must be positive")
    if not 0 <= args.fixed_axis_until <= args.rounds:
        raise ValueError("fixed_axis_until must be between 0 and rounds")
    if args.restricted_impl == "fast":
        if args.inner_steps != 1:
            raise ValueError("fast Restricted-SAM requires inner_steps=1")
        if args.regularizer_solver not in {"gradient", "closed_form"}:
            raise ValueError(
                "fast Restricted-SAM requires the gradient or "
                "closed-form solver"
            )
        if (
            args.regularizer_solver == "gradient"
            and not args.skip_zero_init_single_step_regularizer
        ):
            raise ValueError(
                "fast gradient Restricted-SAM requires "
                "--skip_zero_init_single_step_regularizer"
            )
        if args.search_diagnostics != "minimal":
            raise ValueError(
                "fast Restricted-SAM requires --search_diagnostics minimal"
            )


def set_random_seed(seed, torch_module=None):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


def scheduled_rho(args, round_index):
    if args.rho_schedule == "constant":
        return float(args.rho)
    progress = min(
        max(int(round_index), 0),
        int(args.rho_schedule_rounds),
    )
    fraction = progress / float(args.rho_schedule_rounds)
    return float(args.rho_start + fraction * (args.rho - args.rho_start))


def cifar_transforms(
    data_name,
    train,
    preprocessing_profile="fedrsam_legacy",
    legacy_test_augmentation=False,
):
    from torchvision import transforms

    if (
        preprocessing_profile == "fedgloss_cifar100"
        and data_name != "CIFAR100"
    ):
        raise ValueError(
            "fedgloss_cifar100 preprocessing requires CIFAR100"
        )
    if preprocessing_profile == "fedgloss_cifar100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        legacy_test_augmentation = False
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    operations = []
    if train:
        operations.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif legacy_test_augmentation:
        operations.append(transforms.RandomCrop(32, padding=4))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(operations)


def build_dataset(
    data_name,
    data_root,
    train,
    download=False,
    preprocessing_profile="fedrsam_legacy",
    legacy_test_augmentation=False,
):
    from torchvision import datasets

    transform = cifar_transforms(
        data_name,
        train,
        preprocessing_profile=preprocessing_profile,
        legacy_test_augmentation=legacy_test_augmentation,
    )
    if data_name == "CIFAR100":
        return datasets.CIFAR100(
            data_root,
            train=train,
            download=download,
            transform=transform,
        )
    return datasets.CIFAR10(
        data_root,
        train=train,
        download=download,
        transform=transform,
    )


def build_model(data_name, model_name, normalization, device):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    for import_root in (module_dir, project_root):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

    def load_model_symbol(filename, symbol):
        candidates = (
            os.path.join(module_dir, "models", filename),
            os.path.join(project_root, "models", filename),
        )
        model_path = next((path for path in candidates if os.path.isfile(path)), None)
        if model_path is None:
            raise ModuleNotFoundError(
                "Cannot locate models/{} from {}".format(filename, module_dir)
            )
        module_name = "_fedrsam_local_{}".format(os.path.splitext(filename)[0])
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, symbol)

    num_classes = 100 if data_name == "CIFAR100" else 10
    if model_name == "mobilenetv2":
        from models.mobilenetv2 import MobileNetV2

        model = MobileNetV2(
            num_classes=num_classes,
            normalization=normalization,
        )
    elif model_name == "lenet5":
        from lenet5_fast_exact import LeNet5

        model = LeNet5(num_classes=num_classes)
    elif normalization == "GN":
        ResNet18 = load_model_symbol("resnet.py", "ResNet18")

        model = ResNet18(num_classes=num_classes)
    else:
        ResNet18BN = load_model_symbol("resnet_bn.py", "ResNet18BN")

        model = ResNet18BN(num_classes=num_classes)
    return model.to(device)


def communicated_parameter_names(model):
    state = model.state_dict()
    names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in state and parameter.is_floating_point()
    )
    if not names:
        raise RuntimeError("model has no communicated parameters")
    return names


def cpu_parameter_state(model, names):
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in names
    }


def tensor_map_nbytes(tensors):

    return sum(
        int(value.numel()) * int(value.element_size())
        for value in tensors.values()
    )


def load_or_create_partition(args, train_dataset):
    path = Path(
        "num_workers_{}-alpha_value_{}-data_{}".format(
            args.num_workers,
            args.dirichlet_alpha,
            args.data_name,
        )
    )
    if path.exists():
        with path.open("rb") as handle:
            indices = pickle.load(handle)
        print("Loaded partition:", path)
        return indices

    from dirichlet_data import data_from_dirichlet

    num_classes = 100 if args.data_name == "CIFAR100" else 10
    samples_per_client = max(1, len(train_dataset) // args.num_workers)
    indices, _ = data_from_dirichlet(
        args.data_name,
        args.dirichlet_alpha,
        num_classes,
        args.num_workers,
        samples_per_client,
        seed=args.seed,
    )
    with path.open("wb") as handle:
        pickle.dump(indices, handle)
    print("Created partition:", path)
    return indices


def standard_sam_step(
    model,
    optimizer,
    images,
    labels,
    rho,
    grad_clip_norm=10.0,
    eps=1e-12,
):
    import torch

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad()
    clean_loss = torch.nn.functional.cross_entropy(model(images), labels)
    clean_loss.backward()
    gradients = [
        parameter.grad.detach().norm(p=2)
        for parameter in trainable
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("SAM received no gradients")
    grad_norm = torch.norm(torch.stack(gradients), p=2)
    scale = rho / (grad_norm + eps)

    saved = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            saved[name] = parameter.detach().clone()
            parameter.add_(parameter.grad, alpha=float(scale.item()))

    optimizer.zero_grad()
    try:
        perturbed_loss = torch.nn.functional.cross_entropy(
            model(images),
            labels,
        )
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in saved:
                    parameter.copy_(saved[name])
    torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    optimizer.step()
    return perturbed_loss.detach(), {
        "final_perturbation_norm": float(rho),
        "sam_grad_norm": float(grad_norm.detach().item()),
    }


def average_diagnostics(diagnostics):
    if not diagnostics:
        return {}
    keys = sorted(set().union(*(item.keys() for item in diagnostics)))
    result = {}
    for key in keys:
        values = [float(item[key]) for item in diagnostics if key in item]
        if values:
            result[key] = sum(values) / len(values)
    return result


def evaluate(model, dataset, batch_size, num_workers, device):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return {
        "test_loss": loss_sum / max(total, 1),
        "test_accuracy": correct / max(total, 1),
    }


def write_row(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, args, round_index, row, state):
    import torch

    torch.save(
        {
            "round": round_index,
            "args": vars(args),
            "metric": row,
            "parameter_state": state,
        },
        path,
    )


def summarize_csv(csv_path, summary_path, history_names):
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {"rounds": len(rows), "all": {}, "stages": {}}

    def numeric_values(selected_rows, name):
        values = []
        for row in selected_rows:
            text = row.get(name, "")
            if text in ("", "nan", "NaN"):
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    for name in history_names:
        values = numeric_values(rows, name)
        if values:
            ordered = sorted(values)
            result["all"][name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "q25": ordered[int(0.25 * (len(ordered) - 1))],
                "q75": ordered[int(0.75 * (len(ordered) - 1))],
                "count": len(values),
            }

    boundaries = (
        ("early", 0, max(1, len(rows) // 3)),
        ("middle", len(rows) // 3, max(2, 2 * len(rows) // 3)),
        ("late", 2 * len(rows) // 3, len(rows)),
    )
    for label, start, end in boundaries:
        selected = rows[start:end]
        result["stages"][label] = {}
        for name in (
            "axis_cosine",
            "raw_axis_cosine",
            "full_ncd",
            "full_capture",
            "full_hit_005",
        ):
            values = numeric_values(selected, name)
            if values:
                result["stages"][label][name] = statistics.fmean(values)

    result["runtime"] = {}
    for name in (
        "client_seconds",
        "history_seconds",
        "eval_seconds",
        "round_seconds",
    ):
        values = numeric_values(rows, name)
        if values:
            result["runtime"][name] = {
                "total": sum(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
            }
    elapsed = numeric_values(rows, "elapsed_seconds")
    if elapsed:
        result["runtime"]["wall_clock_seconds"] = elapsed[-1]

    result["effective_communication"] = {}
    for output_name, column_name in (
        ("server_to_client_bytes", "effective_s2c_bytes_round"),
        ("client_to_server_bytes", "effective_c2s_bytes_round"),
        ("total_bytes", "effective_total_bytes_round"),
    ):
        values = numeric_values(rows, column_name)
        if values:
            total = sum(values)
            result["effective_communication"][output_name] = total
            result["effective_communication"][
                output_name.replace("_bytes", "_gib")
            ] = total / float(1024 ** 3)
    cumulative = numeric_values(
        rows,
        "effective_total_bytes_cumulative",
    )
    if cumulative:
        result["effective_communication"][
            "total_bytes_cumulative"
        ] = cumulative[-1]
        result["effective_communication"][
            "total_gib_cumulative"
        ] = cumulative[-1] / float(1024 ** 3)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


def run(args):
    apply_protocol_defaults(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import ray
    import torch
    from torch.utils.data import DataLoader, SubsetRandomSampler

    from fedrsam_closed_form_hybrid.core import (
        FedRSAMConfig,
        FedRSAMLocalOptimizer,
        aggregate_client_updates,
        build_cone_state,
        make_training_cone,
    )
    from fedrsam_baseline.history_metrics import (
        HISTORY_METRIC_NAMES,
        HistoryMetricConfig,
        compute_history_metrics,
    )

    validate_args(args)
    set_random_seed(args.seed, torch)
    is_fedrsam = args.alg in {"FedRSAM", "NSAMFed"}
    legacy_draw3 = args.protocol == "original_draw3"
    driver_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = build_dataset(
        args.data_name,
        args.data_root,
        train=True,
        download=True,
        preprocessing_profile=args.preprocessing_profile,
        legacy_test_augmentation=legacy_draw3,
    )
    test_dataset = build_dataset(
        args.data_name,
        args.data_root,
        train=False,
        download=True,
        preprocessing_profile=args.preprocessing_profile,
        legacy_test_augmentation=(
            legacy_draw3 and not args.deterministic_test
        ),
    )
    partition = load_or_create_partition(args, train_dataset)

    server_model = build_model(
        args.data_name,
        args.model,
        args.normalization,
        driver_device,
    )
    parameter_names = communicated_parameter_names(server_model)
    global_state = cpu_parameter_state(server_model, parameter_names)
    def make_restricted_config(radius, round_index):
        hybrid_fixed_axis = (
            args.fixed_axis_until > 0
            and round_index < args.fixed_axis_until
        )
        return FedRSAMConfig(
            rho=radius,
            alpha_lr=0.0 if hybrid_fixed_axis else args.alpha_lr,
            z_lr=0.0 if hybrid_fixed_axis else args.z_lr,
            lambda_z=args.lambda_z,
            inner_steps=args.inner_steps,
            alpha_init=args.alpha_init,
            radius_mode="boundary",
            project_z_box=args.project_z_box,
            regularizer_solver=args.regularizer_solver,
            skip_zero_init_single_step_regularizer=(
                args.skip_zero_init_single_step_regularizer
            ),
            inner_implementation=args.restricted_impl,
            diagnostics_mode=args.search_diagnostics,
            disable_residual=args.disable_residual,
            alpha_scope=args.alpha_scope,
            cone_scope=args.cone_scope,
            fixed_axis_single_pass=hybrid_fixed_axis,
        )
    geometry_config = HistoryMetricConfig(
        rho=args.rho,
        projection_steps=args.geometry_projection_steps,
        max_rank_samples_per_tensor=args.geometry_rank_samples,
    )

    class VisionClient:
        def __init__(
            self,
            worker_id,
            data_name,
            data_root,
            model_name,
            normalization,
            client_partition,
            names,
            seed,
            legacy_protocol,
            preprocessing_profile,
        ):
            import torch as actor_torch

            self.worker_id = worker_id
            self.data_name = data_name
            self.data_root = data_root
            self.model_name = model_name
            self.normalization = normalization
            self.partition = client_partition
            self.parameter_names = tuple(names)
            self.seed = seed
            self.legacy_protocol = legacy_protocol
            self.preprocessing_profile = preprocessing_profile
            actor_torch.manual_seed(seed + int(worker_id))
            if actor_torch.cuda.is_available():
                actor_torch.cuda.manual_seed_all(seed + int(worker_id))
            self.device = actor_torch.device(
                "cuda" if actor_torch.cuda.is_available() else "cpu"
            )
            self.dataset = build_dataset(
                data_name,
                data_root,
                train=True,
                download=False,
                preprocessing_profile=preprocessing_profile,
                legacy_test_augmentation=legacy_protocol,
            )
            self.model = build_model(
                data_name,
                model_name,
                normalization,
                self.device,
            )
            if communicated_parameter_names(self.model) != self.parameter_names:
                raise RuntimeError("server/client parameter names differ")
            self.cpu_rng_state = actor_torch.get_rng_state()
            self.cuda_rng_state = (
                actor_torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            )

        def update(
            self,
            algorithm,
            state,
            client_id,
            round_index,
            learning_rate,
            local_epochs,
            max_steps,
            batch_size,
            weight_decay,
            rho,
            cone,
            config,
            previous_server_velocity,
            client_lookahead_gamma,
            warmup_rho,
        ):
            import torch as actor_torch

            actor_torch.set_rng_state(self.cpu_rng_state)
            if self.cuda_rng_state is not None:
                actor_torch.cuda.set_rng_state(
                    self.cuda_rng_state,
                    self.device,
                )
            local_seed = self.seed + 1009 * round_index + int(client_id)
            if not self.legacy_protocol:
                actor_torch.manual_seed(local_seed)
                if actor_torch.cuda.is_available():
                    actor_torch.cuda.manual_seed_all(local_seed)
            local_start_state = {}
            for name in self.parameter_names:
                value = state[name].detach().cpu()
                if (
                    client_lookahead_gamma != 0.0
                    and name in previous_server_velocity
                ):
                    value = (
                        value
                        + client_lookahead_gamma
                        * previous_server_velocity[name].detach().cpu()
                    )
                local_start_state[name] = value.clone()
            self.model.load_state_dict(local_start_state, strict=False)
            self.model.to(self.device)
            self.model.train()

            generator = actor_torch.Generator()
            generator.manual_seed(
                self.seed if self.legacy_protocol else local_seed
            )
            sampler = SubsetRandomSampler(
                self.partition[client_id],
                generator=generator,
            )
            loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
            )
            optimizer = actor_torch.optim.SGD(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            restricted_optimizer = None
            if algorithm in {"FedRSAM", "NSAMFed"} and cone is not None:
                restricted_optimizer = FedRSAMLocalOptimizer(
                    self.model,
                    optimizer,
                    config,
                    grad_clip_norm=10.0,
                )
                restricted_optimizer.prepare_cone(cone)

            step_count = 0
            loss_sum = 0.0
            diagnostics = []
            for _ in range(local_epochs):
                for images, labels in loader:
                    if step_count >= max_steps:
                        break
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if restricted_optimizer is None:
                        sam_radius = (
                            warmup_rho
                            if (
                                algorithm in {"FedRSAM", "NSAMFed"}
                                and cone is None
                            )
                            else rho
                        )
                        loss, diagnostic = standard_sam_step(
                            self.model,
                            optimizer,
                            images,
                            labels,
                            sam_radius,
                        )
                    else:
                        def loss_function(output):
                            return actor_torch.nn.functional.cross_entropy(
                                output,
                                labels,
                            )

                        loss, diagnostic = restricted_optimizer.step(
                            (images,),
                            loss_function,
                            cone,
                        )
                    loss_sum += float(loss.item())
                    diagnostics.append(diagnostic)
                    step_count += 1
                if step_count >= max_steps:
                    break

            current = self.model.state_dict()
            delta = {
                name: (
                    current[name].detach().cpu()
                    - local_start_state[name]
                )
                for name in self.parameter_names
            }
            self.cpu_rng_state = actor_torch.get_rng_state()
            if self.cuda_rng_state is not None:
                self.cuda_rng_state = actor_torch.cuda.get_rng_state(
                    self.device
                )
            return {
                "client_id": int(client_id),
                "delta": delta,
                "loss": loss_sum / max(step_count, 1),
                "steps": step_count,
                "update_mode": (
                    "RestrictedSAM"
                    if restricted_optimizer is not None
                    else (
                        "SAM_warmup"
                        if algorithm in {"FedRSAM", "NSAMFed"}
                        else "SAM"
                    )
                ),
                "diagnostics": average_diagnostics(diagnostics),
            }

    selected_count = int(args.num_workers * args.selection)
    visible_gpu_ids = [
        value.strip()
        for value in str(args.gpu).split(",")
        if value.strip()
    ]
    if args.serial_clients:
        actor_count = selected_count
        workers = [
            VisionClient(
                worker_id,
                args.data_name,
                os.path.abspath(args.data_root),
                args.model,
                args.normalization,
                partition,
                parameter_names,
                args.seed,
                legacy_draw3,
                args.preprocessing_profile,
            )
            for worker_id in range(actor_count)
        ]
    else:
        ray_address = os.environ.get("FEDRSAM_RAY_ADDRESS")
        ray_init_kwargs = {
            "ignore_reinit_error": True,
        }
        if ray_address:
            ray_init_kwargs["address"] = ray_address
        else:
            ray_init_kwargs["num_gpus"] = len(visible_gpu_ids)
            ray_init_kwargs.update(num_cpus=10, object_store_memory=536870912, include_dashboard=False, _node_ip_address="127.0.0.1")
        if args.ray_temp_dir:
            ray_temp_dir = os.path.abspath(args.ray_temp_dir)
            os.makedirs(ray_temp_dir, exist_ok=True)
            ray_init_kwargs["_temp_dir"] = ray_temp_dir
        ray.init(**ray_init_kwargs)
        Actor = ray.remote(num_gpus=args.num_gpus_per_client)(VisionClient)
        if args.num_gpus_per_client > 0.0:
            actor_capacity = int(
                len(visible_gpu_ids) / args.num_gpus_per_client
            )
            if actor_capacity < 1:
                raise ValueError(
                    "num_gpus_per_client exceeds the available GPU capacity"
                )
            actor_count = min(selected_count, actor_capacity)
        else:
            actor_count = selected_count
        workers = [
            Actor.remote(
                worker_id,
                args.data_name,
                os.path.abspath(args.data_root),
                args.model,
                args.normalization,
                partition,
                parameter_names,
                args.seed,
                legacy_draw3,
                args.preprocessing_profile,
            )
            for worker_id in range(actor_count)
        ]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_{}_{}_p{}_seed{}_{}".format(
        args.alg,
        args.data_name,
        args.model,
        str(args.selection).replace(".", "p"),
        args.seed,
        args.run_name,
    )
    csv_path = output_dir / "{}.csv".format(stem)
    config_path = output_dir / "{}_config.json".format(stem)
    summary_path = output_dir / "{}_summary.json".format(stem)
    checkpoint_path = output_dir / "{}_best.pt".format(stem)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "selected_clients_per_round": selected_count,
                "client_actor_count": actor_count,
                "parameter_count": sum(
                    global_state[name].numel() for name in parameter_names
                ),
                "history_target": (
                    "server-applied update including server momentum; "
                    "raw aggregate is logged as a secondary diagnostic"
                ),
                "cone_implementation": (
                    "layer-wise direction normalization, unbiased=False "
                    "variance, boundary rho normalization"
                ),
                "effective_communication_definition": (
                    "Logical tensor bytes crossing the server/client boundary "
                    "and consumed by the receiver. Excludes serialization, "
                    "transport overhead, server-only momentum, and any "
                    "client-lookahead velocity when gamma is zero."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    base_fields = (
        "round",
        "hybrid_phase",
        "client_update",
        "lr",
        "rho",
        "client_loss",
        "perturbation_norm",
        "residual_component_ratio",
        "test_loss",
        "test_accuracy",
        "client_seconds",
        "history_seconds",
        "eval_seconds",
        "round_seconds",
        "elapsed_seconds",
        "effective_model_bytes_per_client",
        "effective_auxiliary_bytes_per_client",
        "effective_s2c_bytes_per_client",
        "effective_c2s_bytes_per_client",
        "effective_s2c_bytes_round",
        "effective_c2s_bytes_round",
        "effective_total_bytes_round",
        "effective_total_bytes_cumulative",
    )
    fieldnames = (
        base_fields
        + SEARCH_DIAGNOSTIC_NAMES
        + SEARCH_DISTRIBUTION_NAMES
        + HISTORY_METRIC_NAMES
    )
    rng = np.random.default_rng(args.seed)
    momentum_buffer = {}
    cone = None
    best_accuracy = -1.0
    cumulative_effective_bytes = 0
    start_time = time.time()

    try:
        for round_index in range(args.rounds):
            round_start = time.time()
            round_rho = scheduled_rho(args, round_index)
            restricted_config = make_restricted_config(
                round_rho,
                round_index,
            )
            reported_round = (
                round_index if legacy_draw3 else round_index + 1
            )
            if args.lr_schedule == "cosine":
                learning_rate = (
                    0.5
                    * args.lr
                    * (
                        1.0
                        + math.cos(
                            math.pi * round_index / args.rounds
                        )
                    )
                )
            else:
                learning_rate = args.lr * (
                    args.lr_decay ** round_index
                )
            if legacy_draw3:
                client_order = np.arange(args.num_workers)
                np.random.shuffle(client_order)
                selected = client_order[:selected_count].tolist()
            else:
                selected = np.sort(
                    rng.choice(
                        args.num_workers,
                        size=selected_count,
                        replace=False,
                    )
                ).tolist()
            round_start_state = {
                name: value.detach().cpu().clone()
                for name, value in global_state.items()
            }
            state_ref = (
                global_state
                if args.serial_clients
                else ray.put(global_state)
            )
            cone_ref = None
            if cone is not None:
                training_cone = make_training_cone(cone)
                cone_ref = (
                    training_cone
                    if args.serial_clients
                    else ray.put(training_cone)
                )
            effective_model_bytes = tensor_map_nbytes(global_state)
            effective_auxiliary_bytes = 0
            if cone is not None:
                effective_auxiliary_bytes += tensor_map_nbytes(
                    cone.directions
                )
                effective_auxiliary_bytes += tensor_map_nbytes(
                    cone.variances
                )
            if args.client_lookahead_gamma != 0.0:
                effective_auxiliary_bytes += tensor_map_nbytes(
                    momentum_buffer
                )
                client_velocity = momentum_buffer
            else:
                client_velocity = {}
            velocity_ref = (
                client_velocity
                if args.serial_clients
                else ray.put(client_velocity)
            )
            update_args = [
                (
                    args.alg,
                    state_ref,
                    int(client_id),
                    round_index,
                    learning_rate,
                    args.local_epochs,
                    args.max_steps,
                    args.batch_size,
                    args.weight_decay,
                    round_rho,
                    cone_ref,
                    restricted_config,
                    velocity_ref,
                    args.client_lookahead_gamma,
                    (
                        round_rho
                        if args.rho_schedule != "constant"
                        else args.warmup_rho
                    ),
                )
                for client_id in selected
            ]
            if args.serial_clients:
                results = [
                    workers[task_index % actor_count].update(*client_args)
                    for task_index, client_args in enumerate(update_args)
                ]
            else:
                futures = [
                    workers[task_index % actor_count].update.remote(
                        *client_args
                    )
                    for task_index, client_args in enumerate(update_args)
                ]
                results = ray.get(futures)
            client_seconds = time.time() - round_start
            client_updates = [result["delta"] for result in results]
            effective_s2c_per_client = (
                effective_model_bytes + effective_auxiliary_bytes
            )
            update_payloads = [
                tensor_map_nbytes(update)
                for update in client_updates
            ]
            effective_c2s_per_client = (
                sum(update_payloads) / len(update_payloads)
            )
            effective_s2c_round = (
                effective_s2c_per_client * len(client_updates)
            )
            effective_c2s_round = sum(update_payloads)
            effective_total_round = (
                effective_s2c_round + effective_c2s_round
            )
            cumulative_effective_bytes += effective_total_round

            server_momentum = (
                args.server_momentum if is_fedrsam else 0.0
            )
            global_state, raw_aggregate, momentum_buffer = (
                aggregate_client_updates(
                    global_state,
                    client_updates,
                    momentum_buffer=momentum_buffer,
                    server_momentum=server_momentum,
                    normalized_momentum=(
                        args.normalized_server_momentum
                        if is_fedrsam
                        else False
                    ),
                )
            )
            applied_update = {
                name: global_state[name] - round_start_state[name]
                for name in parameter_names
            }
            server_model.load_state_dict(global_state, strict=False)

            next_cone = None
            if is_fedrsam:
                next_cone = build_cone_state(
                    round_start_state,
                    global_state,
                    client_updates,
                    parameter_names=parameter_names,
                    variance_unbiased=False,
                    direction_scope=args.cone_scope,
                )

            history = {name: float("nan") for name in HISTORY_METRIC_NAMES}
            history_start = time.time()
            if (
                is_fedrsam
                and args.history_profile != "off"
                and cone is not None
                and reported_round % args.geometry_every == 0
            ):
                history = compute_history_metrics(
                    cone,
                    applied_update,
                    raw_aggregate,
                    next_cone,
                    geometry_config,
                )
            history_seconds = time.time() - history_start
            cone = next_cone

            diagnostic_key = "final_perturbation_norm"
            perturbation_norm = sum(
                result["diagnostics"].get(diagnostic_key, 0.0)
                for result in results
            ) / len(results)
            residual_ratio = (
                sum(
                    result["diagnostics"].get(
                        "residual_component_ratio",
                        0.0,
                    )
                    for result in results
                ) / len(results)
                if results[0]["update_mode"] == "RestrictedSAM"
                else float("nan")
            )
            search_diagnostics = {}
            restricted_mode = (
                results[0]["update_mode"] == "RestrictedSAM"
            )
            for diagnostic_name in SEARCH_DIAGNOSTIC_NAMES:
                diagnostic_values = [
                    result["diagnostics"][diagnostic_name]
                    for result in results
                    if diagnostic_name in result["diagnostics"]
                ]
                search_diagnostics[diagnostic_name] = (
                    sum(diagnostic_values) / len(diagnostic_values)
                    if restricted_mode and diagnostic_values
                    else float("nan")
                )
            search_distributions = {}
            for diagnostic_name in SEARCH_DISTRIBUTION_BASES:
                diagnostic_values = [
                    result["diagnostics"][diagnostic_name]
                    for result in results
                    if diagnostic_name in result["diagnostics"]
                ]
                if restricted_mode and diagnostic_values:
                    quantiles = np.quantile(
                        np.asarray(diagnostic_values, dtype=np.float64),
                        (0.25, 0.5, 0.75),
                    )
                    for suffix, value in zip(
                        ("q25", "median", "q75"),
                        quantiles,
                    ):
                        search_distributions[
                            "{}_{}".format(diagnostic_name, suffix)
                        ] = float(value)
                else:
                    for suffix in ("q25", "median", "q75"):
                        search_distributions[
                            "{}_{}".format(diagnostic_name, suffix)
                        ] = float("nan")
            row = {
                "round": reported_round,
                "hybrid_phase": (
                    "fixed_axis"
                    if round_index < args.fixed_axis_until
                    else "closed_form_cone"
                ),
                "client_update": results[0]["update_mode"],
                "lr": learning_rate,
                "rho": round_rho,
                "client_loss": sum(result["loss"] for result in results)
                / len(results),
                "perturbation_norm": perturbation_norm,
                "residual_component_ratio": residual_ratio,
                "test_loss": float("nan"),
                "test_accuracy": float("nan"),
                "client_seconds": client_seconds,
                "history_seconds": history_seconds,
                "eval_seconds": 0.0,
                "round_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "effective_model_bytes_per_client": effective_model_bytes,
                "effective_auxiliary_bytes_per_client": (
                    effective_auxiliary_bytes
                ),
                "effective_s2c_bytes_per_client": (
                    effective_s2c_per_client
                ),
                "effective_c2s_bytes_per_client": (
                    effective_c2s_per_client
                ),
                "effective_s2c_bytes_round": effective_s2c_round,
                "effective_c2s_bytes_round": effective_c2s_round,
                "effective_total_bytes_round": effective_total_round,
                "effective_total_bytes_cumulative": (
                    cumulative_effective_bytes
                ),
                **search_diagnostics,
                **search_distributions,
                **history,
            }

            should_evaluate = (
                reported_round % args.eval_every == 0
                or round_index + 1 == args.rounds
            )
            if should_evaluate:
                eval_start = time.time()
                evaluation = evaluate(
                    server_model,
                    test_dataset,
                    args.eval_batch_size,
                    args.num_eval_workers,
                    driver_device,
                )
                row["eval_seconds"] = time.time() - eval_start
                row.update(evaluation)
                if (
                    args.save_best
                    and evaluation["test_accuracy"] > best_accuracy
                ):
                    best_accuracy = evaluation["test_accuracy"]
                    save_checkpoint(
                        checkpoint_path,
                        args,
                        reported_round,
                        row,
                        global_state,
                    )

            row["round_seconds"] = time.time() - round_start
            row["elapsed_seconds"] = time.time() - start_time
            write_row(csv_path, fieldnames, row)
            print(
                "[{}/{}] {} lr={:.6g} loss={:.4f} "
                "axis_cos={} full_ncd={} hit5={} acc={} time={:.1f}s".format(
                    reported_round,
                    args.rounds - 1 if legacy_draw3 else args.rounds,
                    row["client_update"],
                    learning_rate,
                    row["client_loss"],
                    (
                        "{:.4f}".format(row["axis_cosine"])
                        if math.isfinite(row["axis_cosine"])
                        else "-"
                    ),
                    (
                        "{:.4f}".format(row["full_ncd"])
                        if math.isfinite(row["full_ncd"])
                        else "-"
                    ),
                    (
                        "{:.0f}".format(row["full_hit_005"])
                        if math.isfinite(row["full_hit_005"])
                        else "-"
                    ),
                    (
                        "{:.4f}".format(row["test_accuracy"])
                        if math.isfinite(row["test_accuracy"])
                        else "-"
                    ),
                    row["round_seconds"],
                ),
                flush=True,
            )
    finally:
        if not args.serial_clients:
            ray.shutdown()

    summarize_csv(csv_path, summary_path, HISTORY_METRIC_NAMES)
    print("Metrics:", csv_path)
    print("Summary:", summary_path)
    if args.save_best:
        print("Best checkpoint:", checkpoint_path)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
