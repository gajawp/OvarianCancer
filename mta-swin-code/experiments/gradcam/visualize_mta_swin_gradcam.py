from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import create_model
from src.utils.gradcam import TokenGradCAM, denormalize_imagenet, overlay_heatmap


DEFAULT_CHECKPOINT = Path(
    "/projects/insightx-lab/mta-swin/finetuned-weights-seed=1/best_MTA-Swin_pretrained_model.pt"
)
DEFAULT_DATASET_ROOT = Path("/projects/insightx-lab/cleaned-brain-tumour-dataset")
DEFAULT_CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_LAYERS = tuple(
    f"stage{stage}_{position}_{module_name}"
    for stage in range(1, 5)
    for position in ("first", "last")
    for module_name in ("block", "norm1", "norm2")
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SelectedImage:
    class_name: str
    class_index: int
    sample_index: int
    path: Path
    predicted_index: int
    confidence: float


class MTASwinFineTuned(nn.Module):
    """Wrapper matching comparison checkpoints saved from MTASwinModel.state_dict()."""

    def __init__(self, num_classes: int, img_size: int):
        super().__init__()
        self.model = create_model(
            model_name="mta_swin_tiny",
            num_classes=1000,
            img_size=img_size,
            stage_qk_conv=[True, True, False, False],
            stage_head_mixing=[False, False, True, True],
            stage_group_norm=[True, True, False, False],
            cq=5,
            ck=11,
            ch=2,
            drop_path_rate=0.1,
        )
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM figures for a fine-tuned MTA-Swin checkpoint. "
            "By default, the script selects correctly classified images from each class in the dataset split."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Fine-tuned MTA-Swin checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Cleaned brain tumor dataset root.")
    parser.add_argument("--split", type=str, default="Testing", help="Dataset split folder to sample from.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/gradcam/outputs"), help="Output folder.")
    parser.add_argument("--class-names", nargs="+", default=list(DEFAULT_CLASS_NAMES), help="Class names in label order.")
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS), help="Target layers to visualize.")
    parser.add_argument(
        "--image-paths",
        nargs="+",
        type=Path,
        help="Explicit image paths to visualize. Class names are inferred from parent folders.",
    )
    parser.add_argument("--samples-per-class", type=int, default=1, help="Number of images to select per class.")
    parser.add_argument(
        "--selection-policy",
        choices=("first_correct", "highest_confidence_correct"),
        default="first_correct",
        help="How to select images within each class.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--alpha", type=float, default=0.40, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        help="Also save raw per-layer heatmap PNGs for manual inspection.",
    )
    parser.add_argument(
        "--explain",
        choices=("true", "predicted"),
        default="true",
        help="Explain the ground-truth class or the predicted class.",
    )
    parser.add_argument(
        "--allow-misclassified",
        action="store_true",
        help="If set, use the first image per class even when it is misclassified.",
    )
    parser.add_argument(
        "--paper-figure",
        action="store_true",
        help="Save a compact two-row paper figure. Requires exactly one Grad-CAM layer.",
    )
    parser.add_argument(
        "--paper-figure-name",
        type=str,
        default="paper_gradcam_2x4.png",
        help="File name for the compact paper figure.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and all(torch.is_tensor(tensor) for tensor in value.values()):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict) or not all(torch.is_tensor(tensor) for tensor in checkpoint.values()):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}
    return state_dict


def load_model(checkpoint_path: Path, num_classes: int, image_size: int, device: torch.device) -> MTASwinFineTuned:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = MTASwinFineTuned(num_classes=num_classes, img_size=image_size)
    state_dict = load_checkpoint_state(checkpoint_path)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        unwrapped = {
            key.replace("model.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if len(unwrapped) != len(state_dict):
            raise
        model.model.load_state_dict(unwrapped, strict=True)

    model.to(device)
    model.eval()
    return model


def list_class_images(dataset_root: Path, split: str, class_name: str) -> list[Path]:
    class_dir = dataset_root / split / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    return sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path, transform, device: torch.device) -> tuple[int, float]:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_index = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())
    return predicted_index, confidence


def select_images(
    model: nn.Module,
    dataset_root: Path,
    split: str,
    class_names: list[str],
    transform,
    device: torch.device,
    require_correct: bool,
    samples_per_class: int,
    selection_policy: str,
) -> list[SelectedImage]:
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")

    selections: list[SelectedImage] = []

    for class_index, class_name in enumerate(class_names):
        candidates = list_class_images(dataset_root, split, class_name)
        if not candidates:
            raise FileNotFoundError(f"No images found for class '{class_name}' under {dataset_root / split}")

        evaluated: list[tuple[Path, int, float]] = []
        correct: list[tuple[Path, int, float]] = []
        for image_path in candidates:
            predicted_index, confidence = predict_image(model, image_path, transform, device)
            record = (image_path, predicted_index, confidence)
            evaluated.append(record)
            if predicted_index == class_index:
                correct.append(record)
                if selection_policy == "first_correct" and len(correct) >= samples_per_class:
                    break

        if selection_policy == "highest_confidence_correct":
            correct.sort(key=lambda record: record[2], reverse=True)

        chosen = correct[:samples_per_class]
        if len(chosen) < samples_per_class:
            if require_correct:
                raise RuntimeError(
                    f"Only found {len(chosen)} correctly classified images for class '{class_name}' "
                    f"in {dataset_root / split / class_name}; need {samples_per_class}. "
                    "Use --allow-misclassified to fill the remainder with misclassified images."
                )
            chosen_paths = {record[0] for record in chosen}
            for record in evaluated:
                if record[0] not in chosen_paths:
                    chosen.append(record)
                if len(chosen) >= samples_per_class:
                    break

        if len(chosen) < samples_per_class:
            raise RuntimeError(
                f"Only found {len(chosen)} usable images for class '{class_name}'; need {samples_per_class}."
            )

        for sample_index, (image_path, predicted_index, confidence) in enumerate(chosen, start=1):
            selections.append(
                SelectedImage(
                    class_name=class_name,
                    class_index=class_index,
                    sample_index=sample_index,
                    path=image_path,
                    predicted_index=predicted_index,
                    confidence=confidence,
                )
            )

    return selections


def infer_class_name(image_path: Path, class_names: list[str]) -> str:
    for parent in image_path.parents:
        if parent.name in class_names:
            return parent.name
    available = ", ".join(class_names)
    raise ValueError(f"Could not infer class for {image_path}; expected one of: {available}")


def select_explicit_images(
    model: nn.Module,
    image_paths: list[Path],
    class_names: list[str],
    transform,
    device: torch.device,
    require_correct: bool,
) -> list[SelectedImage]:
    selections: list[SelectedImage] = []
    counts_by_class = {class_name: 0 for class_name in class_names}

    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        class_name = infer_class_name(image_path, class_names)
        class_index = class_names.index(class_name)
        predicted_index, confidence = predict_image(model, image_path, transform, device)
        if require_correct and predicted_index != class_index:
            predicted_class = class_names[predicted_index]
            raise RuntimeError(
                f"Explicit image is misclassified: {image_path} "
                f"(true={class_name}, predicted={predicted_class}, confidence={confidence:.4f}). "
                "Use --allow-misclassified to include it anyway."
            )

        counts_by_class[class_name] += 1
        selections.append(
            SelectedImage(
                class_name=class_name,
                class_index=class_index,
                sample_index=counts_by_class[class_name],
                path=image_path,
                predicted_index=predicted_index,
                confidence=confidence,
            )
        )

    return selections


def sample_key(selection: SelectedImage) -> str:
    safe_class = selection.class_name.replace("/", "_")
    return f"{safe_class}_{selection.sample_index:02d}"


def available_target_layers() -> list[str]:
    return [
        f"stage{stage}_{position}_{module_name}"
        for stage in range(1, 5)
        for position in ("first", "last")
        for module_name in ("block", "norm1", "norm2")
    ]


def resolve_target_module(model: MTASwinFineTuned, layer_name: str):
    valid_layers = available_target_layers()
    if layer_name not in valid_layers:
        available = ", ".join(valid_layers)
        raise ValueError(f"Unknown target layer '{layer_name}'. Available layers: {available}")

    stage_name, position, module_name = layer_name.split("_")
    layer_idx = int(stage_name.replace("stage", "")) - 1
    block_idx = 0 if position == "first" else -1
    block = model.model.layers[layer_idx].blocks[block_idx]
    target_module = block if module_name == "block" else getattr(block, module_name)
    token_grid_size = block.input_resolution[0]
    return target_module, token_grid_size


def compute_layer_gradcam(
    model: MTASwinFineTuned,
    input_tensor: torch.Tensor,
    layer_name: str,
    target_index: int,
) -> np.ndarray:
    target_module, token_grid_size = resolve_target_module(model, layer_name)
    gradcam = TokenGradCAM(model=model, target_module=target_module, token_grid_size=token_grid_size)
    try:
        with torch.enable_grad():
            cam_maps, _, _, _ = gradcam(input_tensor=input_tensor, target_index=target_index)
    finally:
        gradcam.close()
    return cam_maps[0]


def save_image_figure(
    selection: SelectedImage,
    class_names: list[str],
    original: np.ndarray,
    overlays: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = 1 + len(overlays)
    fig, axes = plt.subplots(1, columns, figsize=(4.0 * columns, 4.2))
    if columns == 1:
        axes = [axes]

    predicted_label = class_names[selection.predicted_index]
    title = (
        f"{selection.class_name} #{selection.sample_index} | predicted={predicted_label} "
        f"({selection.confidence * 100.0:.2f}%)"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{selection.path.name}", fontsize=9)
    axes[0].axis("off")

    for axis, (layer_name, overlay) in zip(axes[1:], overlays.items()):
        axis.imshow(overlay)
        axis.set_title(layer_name, fontsize=9)
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_grid_figure(
    selections: list[SelectedImage],
    class_names: list[str],
    originals: dict[str, np.ndarray],
    all_overlays: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = list(next(iter(all_overlays.values())).keys())
    rows = len(selections)
    columns = 1 + len(layers)
    fig, axes = plt.subplots(rows, columns, figsize=(2.15 * columns, 2.25 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(title, fontsize=12, fontweight="bold")

    for row_idx, selection in enumerate(selections):
        key = sample_key(selection)
        predicted_label = class_names[selection.predicted_index]
        row_label = (
            f"{selection.class_name} #{selection.sample_index}\npred={predicted_label}\n"
            f"{selection.confidence * 100.0:.2f}%\n{selection.path.name}"
        )
        axes[row_idx, 0].imshow(originals[key])
        axes[row_idx, 0].set_ylabel(row_label, rotation=0, labelpad=64, va="center", fontsize=8)
        axes[row_idx, 0].set_title("Original" if row_idx == 0 else "", fontsize=9)
        axes[row_idx, 0].axis("off")

        for column_idx, layer_name in enumerate(layers, start=1):
            axes[row_idx, column_idx].imshow(all_overlays[key][layer_name])
            axes[row_idx, column_idx].set_title(layer_name if row_idx == 0 else "", fontsize=9)
            axes[row_idx, column_idx].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def display_class_name(class_name: str) -> str:
    if class_name == "notumor":
        return "No tumor"
    return class_name.capitalize()


def save_paper_figure(
    selections: list[SelectedImage],
    originals: dict[str, np.ndarray],
    all_overlays: dict[str, dict[str, np.ndarray]],
    layer_name: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = len(selections)
    fig, axes = plt.subplots(2, columns, figsize=(2.15 * columns, 4.45))
    if columns == 1:
        axes = np.expand_dims(axes, axis=1)

    for column_idx, selection in enumerate(selections):
        key = sample_key(selection)
        panel_label = chr(ord("A") + column_idx)
        axes[0, column_idx].imshow(originals[key])
        axes[0, column_idx].set_title(
            f"({panel_label}) {display_class_name(selection.class_name)}",
            fontsize=10,
            fontweight="bold",
        )
        axes[0, column_idx].axis("off")

        axes[1, column_idx].imshow(all_overlays[key][layer_name])
        axes[1, column_idx].axis("off")

    fig.text(0.014, 0.735, "Original", rotation=90, fontsize=10, fontweight="bold", va="center")
    fig.text(0.014, 0.265, "Grad-CAM", rotation=90, fontsize=10, fontweight="bold", va="center")

    plt.tight_layout(rect=(0.035, 0, 1, 1), pad=0.25, w_pad=0.18, h_pad=0.28)
    plt.savefig(output_path, dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.paper_figure and len(args.layers) != 1:
        raise ValueError("--paper-figure requires exactly one layer in --layers.")

    class_names = list(args.class_names)
    device = torch.device(args.device)
    transform = build_eval_transform(args.image_size)

    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(class_names),
        image_size=args.image_size,
        device=device,
    )

    if args.image_paths:
        selections = select_explicit_images(
            model=model,
            image_paths=args.image_paths,
            class_names=class_names,
            transform=transform,
            device=device,
            require_correct=not args.allow_misclassified,
        )
    else:
        selections = select_images(
            model=model,
            dataset_root=args.dataset_root,
            split=args.split,
            class_names=class_names,
            transform=transform,
            device=device,
            require_correct=not args.allow_misclassified,
            samples_per_class=args.samples_per_class,
            selection_policy=args.selection_policy,
        )

    originals: dict[str, np.ndarray] = {}
    all_overlays: dict[str, dict[str, np.ndarray]] = {}
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "layers": list(args.layers),
        "samples_per_class": args.samples_per_class,
        "selection_policy": args.selection_policy,
        "image_paths": [str(path) for path in args.image_paths] if args.image_paths else None,
        "explain": args.explain,
        "allow_misclassified": args.allow_misclassified,
        "selected_images": [],
    }

    for selection in selections:
        pil_image = Image.open(selection.path).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        original = denormalize_imagenet(input_tensor[0])
        target_index = selection.class_index if args.explain == "true" else selection.predicted_index
        key = sample_key(selection)

        overlays: dict[str, np.ndarray] = {}
        for layer_name in args.layers:
            cam_map = compute_layer_gradcam(
                model=model,
                input_tensor=input_tensor,
                layer_name=layer_name,
                target_index=target_index,
            )
            overlays[layer_name] = overlay_heatmap(original, cam_map, alpha=args.alpha)
            if args.save_heatmaps:
                heatmap_path = args.output_dir / "heatmaps" / f"{key}_{layer_name}.png"
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(heatmap_path, cam_map, cmap="jet")

        image_output = args.output_dir / "per_image" / f"{key}_gradcam_layers.png"
        save_image_figure(
            selection=selection,
            class_names=class_names,
            original=original,
            overlays=overlays,
            output_path=image_output,
        )

        originals[key] = original
        all_overlays[key] = overlays
        metadata["selected_images"].append(
            {
                "class_name": selection.class_name,
                "class_index": selection.class_index,
                "sample_index": selection.sample_index,
                "path": str(selection.path),
                "predicted_class": class_names[selection.predicted_index],
                "predicted_index": selection.predicted_index,
                "confidence": selection.confidence,
                "figure": str(image_output),
            }
        )
        print(f"Saved {image_output}")

    class_figures = {}
    for class_name in class_names:
        class_selections = [selection for selection in selections if selection.class_name == class_name]
        if not class_selections:
            continue
        safe_class = class_name.replace("/", "_")
        class_output = args.output_dir / f"{safe_class}_gradcam_layers.png"
        save_grid_figure(
            selections=class_selections,
            class_names=class_names,
            originals=originals,
            all_overlays=all_overlays,
            output_path=class_output,
            title=f"{class_name} Grad-CAM layer comparison",
        )
        class_figures[class_name] = str(class_output)
        print(f"Saved {class_output}")

    overview_path = args.output_dir / "all_classes_gradcam_layers.png"
    if args.paper_figure:
        save_paper_figure(
            selections=selections,
            originals=originals,
            all_overlays=all_overlays,
            layer_name=args.layers[0],
            output_path=overview_path,
        )
        paper_path = args.output_dir / args.paper_figure_name
        save_paper_figure(
            selections=selections,
            originals=originals,
            all_overlays=all_overlays,
            layer_name=args.layers[0],
            output_path=paper_path,
        )
        metadata["paper_figure"] = str(paper_path)
    else:
        save_grid_figure(
            selections=selections,
            class_names=class_names,
            originals=originals,
            all_overlays=all_overlays,
            output_path=overview_path,
            title="All selected images Grad-CAM layer comparison",
        )

    metadata["class_figures"] = class_figures
    metadata["overview_figure"] = str(overview_path)

    if args.paper_figure:
        print(f"Saved {overview_path}")
        print(f"Saved {paper_path}")

    metadata_path = args.output_dir / "gradcam_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {overview_path}")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
