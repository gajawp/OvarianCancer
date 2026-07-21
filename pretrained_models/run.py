import argparse
import importlib
from pathlib import Path
import random
import numpy as np
import torch

from common.config import CFG
from common.engine import evaluate_model, train_model


MODELS = {"deeplabv3plus", "unetplusplus", "segformer"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("action", choices=["train", "evaluate", "all"])
    args = parser.parse_args()
    seed_everything(CFG.seed)

    module = importlib.import_module(f"{args.model}.model")
    model = module.build_model(pretrained=True)
    optimizer = module.build_optimizer(model, CFG.encoder_lr, CFG.decoder_lr, CFG.weight_decay)
    output_dir = Path(__file__).resolve().parent / args.model / "weights"
    checkpoint = output_dir / "best_model.pth"

    if args.action in {"train", "all"}:
        checkpoint = train_model(model, args.model, optimizer, output_dir)
    if args.action in {"evaluate", "all"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train the model first.")
        evaluate_model(model, checkpoint, args.model, Path(__file__).resolve().parent / "results" / args.model)


if __name__ == "__main__":
    main()
