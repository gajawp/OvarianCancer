from classification_common.classifier_engine import train_classifier
from classification_common.config import get_config
from densenet121_classifier.model import build_densenet121


def main() -> None:
    config = get_config()
    config.input_mode = "ground_truth_roi"
    config.image_size = 224
    config.batch_size = 16
    config.epochs = 50
    config.learning_rate = 1e-4

    train_classifier(
        config,
        "densenet121",
        build_densenet121,
    )


if __name__ == "__main__":
    main()
