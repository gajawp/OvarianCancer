from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "imagenet-100"
DEFAULT_OUTPUT_ROOT = ABLATION_DIR / "runs"
RESULTS_DIR = ABLATION_DIR / "results"
DATASET_ENV_VAR = "MTA_ABLATION_IMAGENET100_PATH"
OUTPUT_ENV_VAR = "MTA_ABLATION_OUTPUT_ROOT"
COMPONENT_GROUP = "component_ablation"
HYPERPARAM_GROUP = "hyperparameter_ablation"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TRAINING_DEPS: dict[str, Any] | None = None


BASE_CONFIG: dict[str, Any] = {
    "seed": 42,
    "output_dir": "",
    "model": {
        "name": "mta_swin_tiny",
        "num_classes": 100,
        "drop_path_rate": 0.2,
        "stage_qk_conv": [True, True, False, False],
        "stage_head_mixing": [False, False, True, True],
        "stage_group_norm": [True, True, False, False],
        "cq": 5,
        "ck": 11,
        "ch": 2,
        "label_smoothing": 0.0,
    },
    "data": {
        "data_dir": str(DEFAULT_DATASET_ROOT),
        "batch_size": 512,
        "num_workers": 8,
        "img_size": 224,
        "pin_memory": True,
    },
    "aug": {
        "mixup": 0.0,
        "cutmix": 0.0,
        "mixup_prob": 0.0,
        "mixup_switch_prob": 0.0,
        "mixup_mode": "batch",
    },
    "epochs": 300,
    "grad_clip": 5.0,
    "print_freq": 10,
    "save_freq": 20,
    "optimizer": {
        "name": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 0.05,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    },
    "scheduler": {
        "name": "cosine",
        "min_lr": 1.0e-5,
        "warmup_epochs": 20,
        "warmup_lr_init": 1.0e-6,
    },
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "min_delta": 0.01,
        "mode": "max",
        "restore_best_weights": True,
    },
    "tensorboard": True,
    "performance": {
        "amp": True,
        "amp_dtype": "bf16",
        "cudnn_benchmark": True,
        "gradient_checkpointing": False,
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    group: str
    experiment_id: str
    stage_qk_conv: tuple[bool, bool, bool, bool]
    stage_head_mixing: tuple[bool, bool, bool, bool]
    stage_group_norm: tuple[bool, bool, bool, bool]
    cq: int
    ck: int
    ch: int = 2
    description: str = ""


def _flags(code: str) -> tuple[bool, bool, bool, bool]:
    if len(code) != 4 or any(char not in {"T", "F"} for char in code):
        raise ValueError(f"Invalid stage code: {code}")
    return tuple(char == "T" for char in code)  # type: ignore[return-value]


def _code(flags: Iterable[bool]) -> str:
    return "".join("T" if value else "F" for value in flags)


def _component_spec(index: int, qk: str, head: str, norm: str) -> ExperimentSpec:
    description = f"QK={qk}, Head-Mixing={head}, GroupNorm={norm}, cq=5, ck=11"
    return ExperimentSpec(
        group=COMPONENT_GROUP,
        experiment_id=f"component_{index:02d}",
        stage_qk_conv=_flags(qk),
        stage_head_mixing=_flags(head),
        stage_group_norm=_flags(norm),
        cq=5,
        ck=11,
        description=description,
    )


COMPONENT_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    _component_spec(1, "TTFF", "FFTT", "FFFF"),
    _component_spec(2, "TTFF", "FFTT", "TTFF"),
    _component_spec(3, "TFFF", "FFTT", "TTFF"),
    _component_spec(4, "TTFF", "FFFT", "TTFF"),
    _component_spec(5, "TTFF", "FFTT", "FFTT"),
    _component_spec(6, "TFFF", "FFFT", "TTFF"),
    _component_spec(7, "FFFF", "FFFF", "TTFF"),
    _component_spec(8, "FFFF", "FFFF", "FFFF"),
    _component_spec(9, "TTFF", "FFTT", "TTTT"),
    _component_spec(10, "FFFF", "FFFF", "FFTT"),
    _component_spec(11, "FFFF", "FFFF", "TTTT"),
)


def _hyperparam_spec(index: int, cq: int, ck: int) -> ExperimentSpec:
    description = f"QK=TTFF, Head-Mixing=FFTT, GroupNorm=TTFF, cq={cq}, ck={ck}"
    return ExperimentSpec(
        group=HYPERPARAM_GROUP,
        experiment_id=f"hyperparam_{index:02d}_cq{cq}_ck{ck}",
        stage_qk_conv=_flags("TTFF"),
        stage_head_mixing=_flags("FFTT"),
        stage_group_norm=_flags("TTFF"),
        cq=cq,
        ck=ck,
        description=description,
    )


HYPERPARAM_EXPERIMENTS: tuple[ExperimentSpec, ...] = tuple(
    _hyperparam_spec(index, cq, ck)
    for index, (cq, ck) in enumerate(
        [(3, 9), (3, 11), (3, 13), (5, 9), (5, 11), (5, 13), (7, 9), (7, 11), (7, 13)],
        start=1,
    )
)


SUMMARY_FIELDNAMES = [
    "group",
    "experiment_id",
    "status",
    "best_acc",
    "best_loss",
    "best_epoch",
    "total_epochs",
    "cq",
    "ck",
    "stage_qk_conv",
    "stage_head_mixing",
    "stage_group_norm",
    "timestamp",
    "run_dir",
]


def get_group_specs(group: str) -> tuple[ExperimentSpec, ...]:
    if group == COMPONENT_GROUP:
        return COMPONENT_EXPERIMENTS
    if group == HYPERPARAM_GROUP:
        return HYPERPARAM_EXPERIMENTS
    raise ValueError(f"Unknown experiment group: {group}")


def get_output_root(cli_output_root: str | None = None) -> Path:
    if cli_output_root:
        return Path(cli_output_root).expanduser()
    env_value = os.environ.get(OUTPUT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_OUTPUT_ROOT


def get_dataset_root() -> Path:
    env_value = os.environ.get(DATASET_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_DATASET_ROOT


def write_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def create_experiment_spec_payload(spec: ExperimentSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["stage_qk_conv"] = list(spec.stage_qk_conv)
    payload["stage_head_mixing"] = list(spec.stage_head_mixing)
    payload["stage_group_norm"] = list(spec.stage_group_norm)
    return payload


def build_config(spec: ExperimentSpec, run_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(BASE_CONFIG)
    config["output_dir"] = str(run_dir)
    config["data"]["data_dir"] = str(get_dataset_root())
    config["model"]["stage_qk_conv"] = list(spec.stage_qk_conv)
    config["model"]["stage_head_mixing"] = list(spec.stage_head_mixing)
    config["model"]["stage_group_norm"] = list(spec.stage_group_norm)
    config["model"]["cq"] = spec.cq
    config["model"]["ck"] = spec.ck
    config["model"]["ch"] = spec.ch
    return config


def _get_training_deps() -> dict[str, Any]:
    global _TRAINING_DEPS
    if _TRAINING_DEPS is None:
        import torch
        import torch.nn as nn
        from torch.utils.tensorboard import SummaryWriter

        from src.data.ablation import create_dataloaders_with_mixup, mixup_criterion, verify_dataset
        from src.models import create_model
        from src.utils.helpers import (
            AverageMeter,
            EarlyStopping,
            TimeEstimator,
            TrainingState,
            accuracy,
            create_optimizer_and_scheduler,
            get_device,
            print_gpu_info,
            print_model_info,
            set_seed,
            setup_logging,
        )

        _TRAINING_DEPS = {
            "torch": torch,
            "nn": nn,
            "SummaryWriter": SummaryWriter,
            "create_dataloaders_with_mixup": create_dataloaders_with_mixup,
            "mixup_criterion": mixup_criterion,
            "verify_dataset": verify_dataset,
            "create_model": create_model,
            "AverageMeter": AverageMeter,
            "EarlyStopping": EarlyStopping,
            "TimeEstimator": TimeEstimator,
            "TrainingState": TrainingState,
            "accuracy": accuracy,
            "create_optimizer_and_scheduler": create_optimizer_and_scheduler,
            "get_device": get_device,
            "print_gpu_info": print_gpu_info,
            "print_model_info": print_model_info,
            "set_seed": set_seed,
            "setup_logging": setup_logging,
        }
    return _TRAINING_DEPS


def get_amp_dtype(config: dict[str, Any]):
    torch = _get_training_deps()["torch"]
    dtype_str = config["performance"].get("amp_dtype", "fp16").lower()
    return torch.bfloat16 if dtype_str == "bf16" else torch.float16


def log_print(message: str, logger=None) -> None:
    print(message)
    if logger:
        logger.info(message)


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    config: dict[str, Any],
    device,
    mixup_fn=None,
    time_estimator=None,
    writer=None,
    logger=None,
) -> tuple[float, float]:
    deps = _get_training_deps()
    torch = deps["torch"]
    AverageMeter = deps["AverageMeter"]
    accuracy = deps["accuracy"]
    mixup_criterion = deps["mixup_criterion"]
    model.train()

    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")

    if time_estimator:
        time_estimator.start_epoch()

    amp_enabled = bool(config["performance"]["amp"] and device.type == "cuda")
    amp_dtype = get_amp_dtype(config)
    print_freq = config.get("print_freq", 10)
    num_batches = len(train_loader)

    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start = datetime.now().timestamp()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model(images)
            if mixup_fn is not None and isinstance(targets, tuple):
                targets_a, targets_b, lam = targets
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if not isinstance(targets, tuple):
            hard_targets = targets.argmax(dim=1) if targets.dim() == 2 else targets
            acc1, = accuracy(outputs, hard_targets, topk=(1,))
            top1.update(acc1.item(), images.size(0))

        losses.update(loss.item(), images.size(0))

        batch_time = datetime.now().timestamp() - batch_start
        if time_estimator:
            time_estimator.update_batch_time(batch_time)

        if batch_idx % print_freq == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            message = (
                f"Epoch [{epoch}][{batch_idx}/{num_batches}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc@1: {top1.val:.2f} ({top1.avg:.2f}) "
                f"LR: {current_lr:.6f}"
            )
            log_print(message, logger)

        if writer and batch_idx % max(1, print_freq * 2) == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("Train/Loss_Batch", losses.val, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)

    if time_estimator:
        time_estimator.end_epoch(epoch)

    return losses.avg, top1.avg


def validate(
    model,
    val_loader,
    criterion,
    epoch: int,
    device,
    writer=None,
    logger=None,
) -> tuple[float, float, float]:
    deps = _get_training_deps()
    torch = deps["torch"]
    AverageMeter = deps["AverageMeter"]
    accuracy = deps["accuracy"]
    model.eval()
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    message = (
        f"Val Epoch [{epoch}] Loss: {losses.avg:.4f} "
        f"Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%"
    )
    log_print(message, logger)

    if writer:
        writer.add_scalar("Val/Loss", losses.avg, epoch)
        writer.add_scalar("Val/Acc1", top1.avg, epoch)
        writer.add_scalar("Val/Acc5", top5.avg, epoch)

    return losses.avg, top1.avg, top5.avg


def ensure_run_dir(output_root: Path, group: str, experiment_id: str) -> tuple[Path, str]:
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    run_dir = output_root / group / experiment_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def select_experiments(group: str, requested_ids: list[str] | None) -> list[ExperimentSpec]:
    specs = list(get_group_specs(group))
    if not requested_ids:
        return specs

    by_id = {spec.experiment_id: spec for spec in specs}
    selected = []
    missing = []
    for experiment_id in requested_ids:
        if experiment_id in by_id:
            selected.append(by_id[experiment_id])
        else:
            missing.append(experiment_id)

    if missing:
        available = ", ".join(by_id.keys())
        raise ValueError(f"Unknown experiment id(s): {', '.join(missing)}. Available: {available}")

    return selected


def flatten_requested_ids(raw_ids: list[str] | None) -> list[str] | None:
    if not raw_ids:
        return None
    return [experiment_id for chunk in raw_ids for experiment_id in chunk.split(",") if experiment_id]


def list_experiments(group: str) -> None:
    for spec in get_group_specs(group):
        print(
            f"{spec.experiment_id}: "
            f"QK={_code(spec.stage_qk_conv)}, "
            f"Head-Mixing={_code(spec.stage_head_mixing)}, "
            f"GroupNorm={_code(spec.stage_group_norm)}, "
            f"cq={spec.cq}, ck={spec.ck}"
        )


def serialize_metrics(
    spec: ExperimentSpec,
    status: str,
    run_dir: Path,
    timestamp: str,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    metrics = {
        "group": spec.group,
        "experiment_id": spec.experiment_id,
        "status": status,
        "best_acc": None,
        "best_loss": None,
        "best_epoch": None,
        "total_epochs": 0,
        "cq": spec.cq,
        "ck": spec.ck,
        "stage_qk_conv": _code(spec.stage_qk_conv),
        "stage_head_mixing": _code(spec.stage_head_mixing),
        "stage_group_norm": _code(spec.stage_group_norm),
        "timestamp": timestamp,
        "run_dir": str(run_dir),
    }
    if summary is not None:
        metrics.update(
            {
                "best_acc": float(summary["best_acc"]),
                "best_loss": float(summary["best_loss"]),
                "best_epoch": int(summary["best_epoch"]) + 1,
                "total_epochs": int(summary["total_epochs"]),
            }
        )
    if error is not None:
        metrics["error"] = error
    return metrics


def train_single_experiment(spec: ExperimentSpec, gpu: int, output_root: Path) -> dict[str, Any]:
    deps = _get_training_deps()
    torch = deps["torch"]
    nn = deps["nn"]
    SummaryWriter = deps["SummaryWriter"]
    create_dataloaders_with_mixup = deps["create_dataloaders_with_mixup"]
    verify_dataset = deps["verify_dataset"]
    create_model = deps["create_model"]
    EarlyStopping = deps["EarlyStopping"]
    TimeEstimator = deps["TimeEstimator"]
    TrainingState = deps["TrainingState"]
    create_optimizer_and_scheduler = deps["create_optimizer_and_scheduler"]
    get_device = deps["get_device"]
    print_gpu_info = deps["print_gpu_info"]
    print_model_info = deps["print_model_info"]
    set_seed = deps["set_seed"]
    setup_logging = deps["setup_logging"]

    run_dir, timestamp = ensure_run_dir(output_root, spec.group, spec.experiment_id)
    config = build_config(spec, run_dir)
    spec_payload = create_experiment_spec_payload(spec)
    write_yaml(config, run_dir / "resolved_config.yaml")
    write_json(spec_payload, run_dir / "experiment_spec.json")

    logger = setup_logging(str(run_dir / "train.log"))
    logger.info("Starting experiment: %s", spec.experiment_id)
    logger.info("Description: %s", spec.description)

    writer = None
    try:
        set_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        device = get_device(gpu)
        print_gpu_info()

        verify_dataset(config["data"]["data_dir"])

        model_config = config["model"].copy()
        model_name = model_config.pop("name")
        model = create_model(model_name=model_name, **model_config)
        model = model.to(device)
        print_model_info(model)

        train_loader, val_loader, mixup_fn = create_dataloaders_with_mixup(config)
        criterion = nn.CrossEntropyLoss(label_smoothing=config["model"].get("label_smoothing", 0.0))
        optimizer, scheduler, use_step_scheduler = create_optimizer_and_scheduler(
            model, config, steps_per_epoch=len(train_loader)
        )

        training_state = TrainingState()
        time_estimator = TimeEstimator(config["epochs"])

        early_stopping = None
        if config.get("early_stopping", {}).get("enabled", False):
            es_config = config["early_stopping"]
            early_stopping = EarlyStopping(
                patience=es_config["patience"],
                min_delta=es_config["min_delta"],
                mode=es_config["mode"],
                restore_best_weights=es_config["restore_best_weights"],
            )

        if config.get("tensorboard", False):
            writer = SummaryWriter(str(run_dir / "tensorboard"))

        scaler = None
        amp_enabled = bool(config["performance"]["amp"] and device.type == "cuda")
        if amp_enabled and get_amp_dtype(config) == torch.float16:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()

        training_state.best_acc = float("-inf")
        best_acc = float("-inf")
        time_estimator.start_training()
        log_print(f"Starting training for {config['epochs']} epochs", logger)

        for epoch in range(config["epochs"]):
            train_loss, train_acc = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler if use_step_scheduler else None,
                scaler=scaler,
                epoch=epoch,
                config=config,
                device=device,
                mixup_fn=mixup_fn,
                time_estimator=time_estimator,
                writer=writer,
                logger=logger,
            )

            if scheduler is not None and not use_step_scheduler:
                scheduler.step()

            val_loss, val_acc1, val_acc5 = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                writer=writer,
                logger=logger,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            prev_best = best_acc
            is_best = training_state.update(epoch, train_loss, val_loss, val_acc1, current_lr)

            if is_best:
                best_acc = val_acc1
                log_print(
                    f"Validation accuracy increased ({prev_best:.6f} --> {best_acc:.6f}). Saving model...",
                    logger,
                )
                checkpoint_state = {
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "training_state": training_state,
                    "config": config,
                    "model": model.state_dict(),
                }
                torch.save(checkpoint_state, run_dir / "best_model.pth")

            if early_stopping:
                should_stop = early_stopping(copy.deepcopy(val_acc1), copy.deepcopy(model.state_dict()), epoch)
                if should_stop:
                    logger.info("Early stopping triggered")
                    if early_stopping.restore_best_weights and early_stopping.get_best_weights() is not None:
                        model.load_state_dict(early_stopping.get_best_weights())
                    break

            epoch_time = time_estimator.epoch_times[-1] if time_estimator.epoch_times else 0.0
            avg_time = (
                sum(time_estimator.epoch_times) / len(time_estimator.epoch_times)
                if time_estimator.epoch_times
                else 0.0
            )
            log_print(f"Epoch [{epoch + 1}/{config['epochs']}] - Time: {epoch_time:.2f}s | Avg: {avg_time:.2f}s", logger)
            log_print(f"Train   Loss: {train_loss:.4f} | Acc: {train_acc:.4f}", logger)
            log_print(f"Val     Loss: {val_loss:.4f} | Acc: {val_acc1:.4f}", logger)
            if early_stopping and early_stopping.counter > 0:
                log_print(
                    f"EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}",
                    logger,
                )

        training_state.save_history(str(run_dir / "training_history.json"))
        summary = training_state.get_summary()
        metrics = serialize_metrics(spec, "success", run_dir, timestamp, summary=summary)
        write_json(metrics, run_dir / "metrics.json")
        logger.info("Experiment completed: %s", spec.experiment_id)
        logger.info("Best validation accuracy: %.2f%%", summary["best_acc"])
        return metrics
    except Exception as exc:  # pragma: no cover - failure path depends on runtime
        traceback.print_exc()
        metrics = serialize_metrics(spec, "failed", run_dir, timestamp, error=str(exc))
        write_json(metrics, run_dir / "metrics.json")
        logger.error("Experiment failed: %s", exc)
        raise
    finally:
        if writer:
            writer.close()


def run_group(group: str, requested_ids: list[str] | None, gpu: int, output_root: Path) -> int:
    specs = select_experiments(group, requested_ids)
    failures: list[str] = []
    print(f"Running {len(specs)} experiment(s) for group: {group}")
    for index, spec in enumerate(specs, start=1):
        print("=" * 80)
        print(f"[{index}/{len(specs)}] {spec.experiment_id}")
        print(spec.description)
        try:
            train_single_experiment(spec=spec, gpu=gpu, output_root=output_root)
        except Exception:
            failures.append(spec.experiment_id)
    print("=" * 80)
    print(f"Finished group: {group}")
    if failures:
        print(f"Failed experiments: {', '.join(failures)}")
        return 1
    print("All experiments completed successfully.")
    return 0


def build_group_arg_parser(group: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {group} experiments")
    parser.add_argument(
        "--experiment-id",
        nargs="+",
        help="Run only specific experiment id(s). Multiple ids can be provided separated by spaces.",
    )
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=f"Override output root. Defaults to ${OUTPUT_ENV_VAR} or {DEFAULT_OUTPUT_ROOT}.",
    )
    return parser


def run_cli(group: str) -> int:
    parser = build_group_arg_parser(group)
    args = parser.parse_args()
    requested_ids = flatten_requested_ids(args.experiment_id)
    if args.list_experiments:
        list_experiments(group)
        return 0
    return run_group(group=group, requested_ids=requested_ids, gpu=args.gpu, output_root=get_output_root(args.output_root))


def collect_latest_successful_metrics(group: str, output_root: Path) -> list[dict[str, Any]]:
    latest_by_id: dict[str, dict[str, Any]] = {}
    for spec in get_group_specs(group):
        metrics_files = sorted((output_root / group / spec.experiment_id).glob("*/metrics.json"))
        for metrics_file in metrics_files:
            with metrics_file.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("status") != "success":
                continue
            current = latest_by_id.get(spec.experiment_id)
            if current is None or metrics["timestamp"] > current["timestamp"]:
                latest_by_id[spec.experiment_id] = metrics

    ordered = []
    for spec in get_group_specs(group):
        if spec.experiment_id in latest_by_id:
            ordered.append(latest_by_id[spec.experiment_id])
    return ordered


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDNAMES})


def aggregate_results(output_root: Path) -> dict[str, Path]:
    component_rows = collect_latest_successful_metrics(COMPONENT_GROUP, output_root)
    hyper_rows = collect_latest_successful_metrics(HYPERPARAM_GROUP, output_root)
    all_rows = component_rows + hyper_rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    component_csv = RESULTS_DIR / "component_ablation_summary.csv"
    hyper_csv = RESULTS_DIR / "hyperparameter_ablation_summary.csv"
    all_csv = RESULTS_DIR / "all_ablation_summary.csv"

    write_summary_csv(component_rows, component_csv)
    write_summary_csv(hyper_rows, hyper_csv)
    write_summary_csv(all_rows, all_csv)

    return {
        "component": component_csv,
        "hyperparameter": hyper_csv,
        "all": all_csv,
    }
