from classification_common.classifier_engine import train_classifier
from classification_common.config import get_config
from swin_t_classifier.model import build_swin_t


def main() -> None:
    config = get_config()
    config.input_mode = "ground_truth_roi"
    config.image_size = 224
    config.batch_size = 8
    config.epochs = 50
    config.learning_rate = 1e-4

    train_classifier(
        config,
        "swin_t",
        build_swin_t,
    )


if __name__ == "__main__":
    main()
