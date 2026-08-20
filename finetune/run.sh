#!/usr/bin/env bash
# Fine-tune a TakZero model towards human-like play.
#
# Usage:
#   ./run.sh [extra finetune args...]
#
# Defaults:
#   data   = ../human_games_tournament_1900.tsv  (rating 1900+, tournament only, no bots, 6x6 2 komi)
#   model  = ../_models/undirected_2850000.ot
#   out    = ../_models/finetuned_human.ot
#   steps  = 3000, batch = 512, frozen blocks = 14, lr = 1e-3, symmetry augmentation on
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd .. && pwd)"
DATA_DIR="$(cd ../.. && pwd)"

# Use the project's Python environment for libtorch (CUDA build).
source "$REPO/.venv/bin/activate"
TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$TORCH_LIB/libtorch.so"
export RUST_LOG=info

exec "$REPO/target/release/finetune" \
    --data "$DATA_DIR/human_games_tournament_1900.tsv" \
    --model "$REPO/_models/undirected_2850000.ot" \
    --out "$REPO/_models/finetuned_human.ot" \
    --steps 3000 \
    --batch-size 512 \
    --frozen-blocks 14 \
    --lr 1e-3 \
    --augment \
    --log-every 50 \
    --save-every 500 \
    "$@"
