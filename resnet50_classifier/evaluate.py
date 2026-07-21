from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_common.config import get_config
from classification_common.dataset import (
    OvarianClassificationDataset,
    build_eval_transform,
)
from classification_common.metrics import (
    calculate_multiclass_metrics,
    save_confusion_matrix,
    save_metrics,
)
from resnet50_classifier.model import build_resnet50


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in tqdm(
        loader,
        desc="Evaluating",
    ):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.asarray(all_targets),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def main() -> None:
    config = get_config()

    # Must match the mode used during training.
    config.input_mode = "ground_truth_roi"
    # config.input_mode = "full_image"
    # config.input_mode = "predicted_roi"
    # config.predicted_mask_dir = (
    #     config.project_root / "classification_results" / "predicted_masks"
    # )

    config.create_output_directories()
    config.validate_paths(require_masks=True)

    checkpoint_path = config.best_checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the classifier before evaluation."
        )

    dataset = OvarianClassificationDataset(
        image_dir=config.image_dir,
        list_path=config.val_list,
        num_classes=config.num_classes,
        transform=build_eval_transform(config.image_size),
        input_mode=config.input_mode,
        mask_dir=config.mask_dir,
        predicted_mask_dir=config.predicted_mask_dir,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
        return_metadata=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )

    model = build_resnet50(
        num_classes=checkpoint.get(
            "num_classes",
            config.num_classes,
        ),
        pretrained=False,
        dropout=config.dropout,
    ).to(config.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 72)
    print("ResNet50 Official Validation/Test Evaluation")
    print("=" * 72)
    print(f"Device      : {config.device}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Input mode  : {config.input_mode}")
    print(f"Samples     : {len(dataset)}")

    targets, predictions, probabilities = evaluate_model(
        model=model,
        loader=loader,
        device=config.device,
    )

    metrics = calculate_multiclass_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_names=config.class_names,
        top_k_values=config.top_k,
    )

    metrics_path = (
        config.metrics_dir / "resnet50_evaluation_metrics.json"
    )
    save_metrics(metrics, metrics_path)

    confusion = np.asarray(metrics["confusion_matrix"])

    save_confusion_matrix(
        confusion=confusion,
        class_names=config.class_names,
        output_path=(
            config.confusion_matrix_dir
            / "resnet50_confusion_matrix.png"
        ),
        normalize=False,
    )

    save_confusion_matrix(
        confusion=confusion,
        class_names=config.class_names,
        output_path=(
            config.confusion_matrix_dir
            / "resnet50_confusion_matrix_normalized.png"
        ),
        normalize=True,
    )

    print("\nEvaluation Results")
    print("-" * 72)
    print(f"Accuracy              : {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy     : "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro precision       : {metrics['macro_precision']:.4f}")
    print(f"Macro recall          : {metrics['macro_recall']:.4f}")
    print(f"Macro F1              : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1           : {metrics['weighted_f1']:.4f}")
    print(
        f"Top-3 accuracy        : "
        f"{metrics.get('top_3_accuracy', float('nan')):.4f}"
    )

    macro_auc = metrics.get("macro_roc_auc_ovr")
    if macro_auc is not None:
        print(f"Macro ROC-AUC OVR     : {macro_auc:.4f}")
    else:
        print("Macro ROC-AUC OVR     : unavailable")

    print("\nPer-class Results")
    print("-" * 72)

    for class_index, result in metrics["per_class"].items():
        print(
            f"[{class_index}] {result['class_name']}: "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"specificity={result['specificity']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"support={result['support']}"
        )

    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
