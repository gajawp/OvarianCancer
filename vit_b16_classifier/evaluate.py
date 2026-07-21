from classification_common.classifier_engine import evaluate_classifier
from classification_common.config import get_config
from vit_b16_classifier.model import build_vit_b16


def main() -> None:
    config = get_config()
    config.input_mode = "ground_truth_roi"
    config.image_size = 224
    config.batch_size = 8

    evaluate_classifier(
        config,
        "vit_b16",
        build_vit_b16,
    )


if __name__ == "__main__":
    main()
