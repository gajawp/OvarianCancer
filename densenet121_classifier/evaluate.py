from classification_common.classifier_engine import evaluate_classifier
from classification_common.config import get_config
from densenet121_classifier.model import build_densenet121


def main() -> None:
    config = get_config()
    config.input_mode = "ground_truth_roi"
    config.image_size = 224
    config.batch_size = 16

    evaluate_classifier(
        config,
        "densenet121",
        build_densenet121,
    )


if __name__ == "__main__":
    main()
