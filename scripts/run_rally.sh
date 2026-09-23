#!/usr/bin/env bash
# Rally branch, end to end. Run from the repository root.
#
#   bash scripts/run_rally.sh
#
# Inputs (see docs/data_format.md):
#   data/rally/shots_merged.csv
#   data/rally/points/<match_id>_<...>.points.csv
# The report stage needs a local Ollama server with qwen2.5:32b.
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DISCOVERY=outputs/rally/discovery

# 1. Preprocessing: train/test match split, train/validation match split
python -m courtstyle.data.rally_dataset \
    --input data/rally/shots_merged.csv --points-dir data/rally/points --output-dir data/rally

# 2. Contrastive training with the order loss
python -m courtstyle.training.train_rally --config configs/rally.yaml --work-dir outputs/rally --device "$DEVICE"

# 3. Style discovery on the training matches; validation and test rallies are
#    assigned to the nearest centroid
python -m courtstyle.discovery.rally_discovery --config configs/rally.yaml \
    --checkpoint outputs/rally/models/best.pt --work-dir "$DISCOVERY" \
    --splits train_all,val,test --device "$DEVICE"

# 4. Evidence statements and language-model report
python -m courtstyle.interpretation.rally_interpretation --run-dir "$DISCOVERY"
python -m courtstyle.interpretation.rally_reports \
    --prompts-dir "$DISCOVERY/interpretation/prompts_specialist_v1" --model qwen2.5

# 5. Held-out transfer to the test matches
python -m courtstyle.evaluation.rally_transfer --config configs/rally.yaml --discovery-dir "$DISCOVERY"
