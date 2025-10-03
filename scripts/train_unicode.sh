#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="configs/unicode_dat_base.yaml"
DATA_PATH="datasets/raw/dataset"
SPLIT_FILE="datasets/metadata/character_splits.json"
LABEL_MAP="datasets/metadata/unicode_label_mapping.json"
OUTPUT_ROOT=${OUTPUT_ROOT:-"output"}
TAG=${TAG:-"unicode"}
AMP_FLAG=${AMP_FLAG:---amp}

echo "=== 推奨バッチサイズを推定します ==="
BEST_BS=$(uv run python scripts/estimate_batch_size.py \
  --cfg ${CONFIG_PATH} \
  --data-path ${DATA_PATH} \
  --split-file ${SPLIT_FILE} \
  --label-map ${LABEL_MAP} \
  --max-power ${MAX_POWER:-10} \
  | awk '/^[0-9]+$/{val=$0} END{print val}')

if [[ -z "${BEST_BS}" ]]; then
  echo "推定バッチサイズの取得に失敗しました" >&2
  exit 1
fi

echo "推定バッチサイズ: ${BEST_BS}"

uv run torchrun --nproc_per_node=1 \
  main.py \
  --cfg ${CONFIG_PATH} \
  --data-path ${DATA_PATH} \
  --output ${OUTPUT_ROOT} \
  --tag ${TAG} \
  --batch-size ${BEST_BS} \
  ${AMP_FLAG}

