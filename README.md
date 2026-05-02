# TDFusion Jittor Reproduction

This directory contains a standalone Jittor reproduction of `TDFusion-main`.

Implemented modules:
- `nets/ReFusion.py` for fusion and learnable loss generation
- `nets/Segformer.py` for segmentation task-network entry points
- `nets/yolo.py` and `nets/yolo_training.py` for detection task-network entry points
- `nets/backbone_SS.py` and `nets/backbone_OD.py` for backbone structures
- `data/dataloader.py` for dataset loading
- `train.py` and `test.py` for end-to-end training/testing flow

## Outputs

Training now records:
- `logs/train.log`
- `loss_curve/loss_history.csv`
- `model/ckpt_*.pkl`
- `samples/ir.png`, `samples/vi.png`, `samples/fused.png`

Testing can optionally save a visualization:

```bash
python test.py --ckpt ./exp/model/ckpt_1.pkl --output_dir ./results --save_vis
```

## Environment fix for GLIBCXX mismatch

If you hit `GLIBCXX_3.4.30 not found` from `jittor_core`, prefer the system libstdc++:

```bash
unset OMP_NUM_THREADS
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

You can also run the helper scripts:

```bash
bash run_train_msrs.sh
bash run_test_msrs.sh
```
