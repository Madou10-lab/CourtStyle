#!/usr/bin/env bash
# Serve branch, end to end. Run from the repository root.
#
#   bash scripts/run_serve.sh
#
# Inputs (see docs/data_format.md):
#   data/serve/fingerprint_serve.csv
#   data/serve/test_set/australian_open_2017.csv
#   data/serve/test_set/australian_open_2017_annotations.csv
# The report stages need a local Ollama server with qwen2.5:32b and
# tensortemplar/prometheus2:7b-fp16.
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
TEST_CSV=data/serve/test_set/australian_open_2017.csv
ANNOTATIONS=data/serve/test_set/australian_open_2017_annotations.csv

# 1. Synthetic records for pre-training, then preprocessing
python -m courtstyle.data.synthetic_serve \
    --input data/serve/fingerprint_serve.csv --output data/serve/synthetic_serve_50k.csv
python -m courtstyle.data.serve_dataset \
    --real-input data/serve/fingerprint_serve.csv \
    --synthetic-input data/serve/synthetic_serve_50k.csv \
    --output-dir data/serve/processed

# 2. Two-stage contrastive training
python -m courtstyle.training.train_serve --config configs/serve.yaml --work-dir outputs/serve --device "$DEVICE"

# 3. Style discovery (k-NN graph, Leiden, profile merging)
python -m courtstyle.discovery.serve_discovery \
    --ckpt-path outputs/serve/serve_encoder_best.pt \
    --processed-dir data/serve/processed --work-dir outputs/serve
RUN=$(ls -d outputs/serve/community_run_* | sort | tail -n 1)

# 4. Evidence statements and language-model reports
python -m courtstyle.interpretation.serve_interpretation \
    --community-run-dir "$RUN" --processed-dir data/serve/processed
python -m courtstyle.interpretation.serve_reports \
    --prompts-dir "$RUN/interpretation/prompts_coach_v2" --model qwen2.5

# 5. Held-out tournament: style assignment and report evaluation
python -m courtstyle.evaluation.evaluate_serve_heldout \
    --community-run-dir "$RUN" \
    --ckpt-path outputs/serve/serve_encoder_best.pt \
    --preprocessor-path data/serve/processed/preprocessor.pkl \
    --test-csv "$TEST_CSV" --annotations-csv "$ANNOTATIONS"
python -m courtstyle.evaluation.evaluate_reports \
    --community-run-dir "$RUN" \
    --annotations-csv "$ANNOTATIONS" \
    --predictions-csv "$RUN/test_set_eval/inference/predictions.csv" \
    --raw-stats-csv "$TEST_CSV" \
    --train-stats-csv data/serve/fingerprint_serve.csv
