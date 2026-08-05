#!/usr/bin/env bash
# Boost experiments on top of the best recipe (sampling + strong aug + MixUp).
# Tries: (2) TTA, (3) EMA, (5) balanced-softmax long-tail loss, (6) longer training.
# Single seed (42) for a quick signal; multi-seed the winner afterwards.
#
# Baseline reference (already run): recipe e100 s42 = 81.45% acc / 76.94 macro-F1
#   tag: MTA-Swin_pretrained_noval_e100_augstrong_mixup_samp0.5_s42
#
# Run:  bash experiments/mmotu_cls/run_boost.sh
set -u

# cd to the mta-swin-code root so relative paths work from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

RUN="python experiments/mmotu_cls/run_mmotu_mta_swin.py --no-val --seed 42"
RECIPE="--sampler balanced --aug strong --mixup"   # current best data-side recipe

echo "############## (3) recipe + EMA ##############"
$RUN --epochs 100 $RECIPE --ema

echo "############## (2) recipe + TTA ##############"
$RUN --epochs 100 $RECIPE --tta

echo "############## (2+3) recipe + EMA + TTA ##############"
$RUN --epochs 100 $RECIPE --ema --tta

echo "############## (6) recipe longer, 200 epochs ##############"
$RUN --epochs 200 $RECIPE

echo "############## (6+2+3) recipe 200ep + EMA + TTA ##############"
$RUN --epochs 200 $RECIPE --ema --tta

echo "############## (5) balanced-softmax loss (long-tail), no sampler/mixup ##############"
$RUN --epochs 100 --aug strong --loss balanced_softmax

echo "############## (5+2+3) balanced-softmax + EMA + TTA ##############"
$RUN --epochs 100 --aug strong --loss balanced_softmax --ema --tta

echo "############## DONE -- 7 runs. Compare vs the e100 recipe baseline (81.45 / 76.94). ##############"
