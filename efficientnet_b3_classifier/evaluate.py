from classification_common.classifier_engine import evaluate_classifier
from classification_common.config import get_config
from efficientnet_b3_classifier.model import build_efficientnet_b3


def main() -> None:
    config = get_config()
    config.input_mode = "ground_truth_roi"
    config.image_size = 300
    config.batch_size = 8

    evaluate_classifier(
        config,
        "efficientnet_b3",
        build_efficientnet_b3,
    )


if __name__ == "__main__":
    main()
