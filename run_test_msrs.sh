#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/root/autodl-tmp/jittor-master/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

CKPT=/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/model/ckpt_50.pkl
python /root/autodl-tmp/TDFusion-main_jittor/test.py \
  --ckpt "$CKPT" \
  --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS \
  --dataset MSRS \
  --split test \
  --output_dir /root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/test_msrs
