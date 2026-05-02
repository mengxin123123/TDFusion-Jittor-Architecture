#!/usr/bin/env bash
set -euo pipefail

unset OMP_NUM_THREADS
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
python /root/autodl-tmp/TDFusion-main_jittor/train.py \
  --dataset MSRS \
  --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS \
  --output_dir /root/autodl-tmp/TDFusion-main_jittor/exp \
  --num_epochs 1 \
  --batch_size 2
