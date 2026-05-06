# TDFusion Jittor Reproduction

This directory contains a standalone Jittor reproduction of `TDFusion-main`.

Implemented modules:
- `nets/ReFusion.py` for fusion and learnable loss generation
- `nets/Segformer.py` for segmentation task-network entry points
- `nets/yolo.py` and `nets/yolo_training.py` for detection task-network entry points
- `nets/backbone_SS.py` and `nets/backbone_OD.py` for backbone structures
- `data/dataloader.py` for dataset loading
- `train.py` and `test.py` for end-to-end training/testing flow
- `train_demo.ipynb` and `test_demo.ipynb` for notebook-based demos with outline support

## 1. Environment setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended runtime notes:
- Jittor will use CUDA automatically when available.
- The training script seeds the run with `seed_everything(42)`.
- Default logging and experiment outputs are written under `./exp` unless `--output_dir` is changed.

## 2. Data preparation

The data loader in `data/dataloader.py` expects paired infrared / visible-image folders and the corresponding labels or masks required by the selected dataset.

Current supported dataset modes in `train.py`:
- `MSRS`

- `M3FD`


Typical usage:
- Pass the dataset name with `--dataset`
- Pass the dataset root with `--data_root`
- For testing, `test.py` reads `ir/` and `vi/` folders from `--data_root/--split` when present, otherwise it falls back to `--data_root/ir` and `--data_root/vi`

Example:

```bash
python train.py --dataset MSRS --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS --output_dir ./exp
```

## 3. Training script

Training is implemented in `train.py`.

Main responsibilities:
- build the fusion model, task model, and learnable loss model
- alternate inner / outer updates
- save checkpoints each epoch
- record scalar logs in `logs/train.log`
- record loss curves in `loss_curve/loss_history.csv`
- save example images in `samples/`

Common command:

```bash
python train.py \
  --dataset MSRS \
  --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS \
  --output_dir ./exp \
  --num_epochs 50 \
  --batch_size 2
```

Useful arguments:
- `--resume` resume from a checkpoint
- `--num_epochs` total training epochs
- `--max_meta_step` max iterations per epoch
- `--lr` learning rate
- `--step_size` scheduler step size
- `--gamma` scheduler decay factor
- `--wandb_project` enable Weights & Biases logging
- `--wandb_name` custom run name

### Training logs produced

For each epoch the script writes:
- `FusionLoss`
- `F_int`
- `F_grad`
- `TaskLoss`

Files written in each experiment directory:
- `model/ckpt_*.pkl`
- `logs/train.log`
- `loss_curve/loss_history.csv`
- `samples/ir.png`
- `samples/vi.png`
- `samples/fused.png`
- `samples/triptych.png`

## 4. Testing script

Testing is implemented in `test.py`.

Main responsibilities:
- load a fusion checkpoint
- read infrared / visible test pairs
- fuse the Y channel in RGB/YCrCb space
- save grayscale fused results and RGB fused results

Common command:

```bash
python test.py \
  --ckpt ./exp/model/ckpt_50.pkl \
  --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS \
  --split test \
  --output_dir ./results
```

Test outputs:
- `results/test/Gray/`
- `results/test/RGB/`

## 5. Experiment log aligned with the PyTorch implementation

The experiment log for this reproduction is stored under:

- `/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/logs/train.log`
- `/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/loss_curve/loss_history.csv`

Summary of the recorded 50-epoch run:
- `FusionLoss` decreased quickly in early epochs and then stayed around `0.063` to `0.067`
- `TaskLoss` decreased from about `0.94` at epoch 1 to about `0.24` at the end of training
- The run contains the full epoch-wise loss trace needed to compare against the PyTorch version

Representative log format:

```text
Epoch 1/50 FusionLoss=0.186199 F_int=0.021728 F_grad=0.164471 TaskLoss=0.937845
Epoch 50/50 FusionLoss=0.065300 F_int=0.014746 F_grad=0.050554 TaskLoss=0.237785
```

## 6. Performance log

The current run already includes the performance-related artifacts needed for inspection:

- training convergence curve: `/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/loss_curve/loss_history.csv`
- saved visual samples: `/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/samples/`
- test fusion outputs are expected under: `/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/test_msrs/test/RGB`

Note:
- The training loss log and loss curve are already present.
- If the test image directory is not yet generated in the current run, run `test.py` with the matching checkpoint and `--output_dir /root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/test_msrs` to create it.

## 7. Notebook demo

Open `train_demo.ipynb` and `test_demo.ipynb` to see the outline structure generated from markdown headings.

## 8. WandB logging

Enable wandb in `train.py`:

```bash
python train.py --dataset MSRS --data_root /root/autodl-tmp/TDFusion-main_jittor/MSRS --num_epochs 1 --batch_size 2 --wandb_project TDFusion-Jittor --wandb_name msrs-demo
```

If `wandb` is installed, the script logs:
- Scalars: `FusionLoss`, `F_int`, `F_grad`, `TaskLoss`
- Images: `ir`, `vi`, `fused`, `triptych`
- Loss curve image: `loss_curve`

## 9. Experiment log contents

### 9.1 `logs/train.log`

```text
Epoch 1/50 FusionLoss=0.186199 F_int=0.021728 F_grad=0.164471 TaskLoss=0.937845
Epoch 2/50 FusionLoss=0.101770 F_int=0.013721 F_grad=0.088050 TaskLoss=0.380037
Epoch 3/50 FusionLoss=0.089206 F_int=0.015748 F_grad=0.073457 TaskLoss=0.343091
Epoch 4/50 FusionLoss=0.075064 F_int=0.013590 F_grad=0.061474 TaskLoss=0.295632
Epoch 5/50 FusionLoss=0.073339 F_int=0.014116 F_grad=0.059223 TaskLoss=0.315609
Epoch 6/50 FusionLoss=0.076875 F_int=0.016189 F_grad=0.060686 TaskLoss=0.306285
Epoch 7/50 FusionLoss=0.072434 F_int=0.014963 F_grad=0.057472 TaskLoss=0.296836
Epoch 8/50 FusionLoss=0.070117 F_int=0.015832 F_grad=0.054285 TaskLoss=0.264324
Epoch 9/50 FusionLoss=0.069571 F_int=0.015434 F_grad=0.054137 TaskLoss=0.293060
Epoch 10/50 FusionLoss=0.066447 F_int=0.015579 F_grad=0.050868 TaskLoss=0.265447
Epoch 11/50 FusionLoss=0.065414 F_int=0.014993 F_grad=0.050421 TaskLoss=0.254510
Epoch 12/50 FusionLoss=0.065899 F_int=0.015443 F_grad=0.050456 TaskLoss=0.241076
Epoch 13/50 FusionLoss=0.064521 F_int=0.014433 F_grad=0.050089 TaskLoss=0.223843
Epoch 14/50 FusionLoss=0.069485 F_int=0.017276 F_grad=0.052209 TaskLoss=0.256231
Epoch 15/50 FusionLoss=0.064257 F_int=0.014130 F_grad=0.050128 TaskLoss=0.223287
Epoch 16/50 FusionLoss=0.062060 F_int=0.014532 F_grad=0.047528 TaskLoss=0.239964
Epoch 17/50 FusionLoss=0.063451 F_int=0.014744 F_grad=0.048707 TaskLoss=0.243541
Epoch 18/50 FusionLoss=0.066619 F_int=0.014929 F_grad=0.051690 TaskLoss=0.212287
Epoch 19/50 FusionLoss=0.064136 F_int=0.015604 F_grad=0.048532 TaskLoss=0.212801
Epoch 20/50 FusionLoss=0.067628 F_int=0.016167 F_grad=0.051461 TaskLoss=0.229583
Epoch 21/50 FusionLoss=0.065292 F_int=0.015311 F_grad=0.049980 TaskLoss=0.222623
Epoch 22/50 FusionLoss=0.066447 F_int=0.014680 F_grad=0.051766 TaskLoss=0.214831
Epoch 23/50 FusionLoss=0.065375 F_int=0.014727 F_grad=0.050648 TaskLoss=0.200870
Epoch 24/50 FusionLoss=0.066411 F_int=0.015536 F_grad=0.050876 TaskLoss=0.205259
Epoch 25/50 FusionLoss=0.064536 F_int=0.014357 F_grad=0.050179 TaskLoss=0.217019
Epoch 26/50 FusionLoss=0.064150 F_int=0.014014 F_grad=0.050137 TaskLoss=0.204440
Epoch 27/50 FusionLoss=0.067800 F_int=0.016423 F_grad=0.051377 TaskLoss=0.211404
Epoch 28/50 FusionLoss=0.067300 F_int=0.014795 F_grad=0.052505 TaskLoss=0.209669
Epoch 29/50 FusionLoss=0.064127 F_int=0.012941 F_grad=0.051185 TaskLoss=0.223165
Epoch 30/50 FusionLoss=0.066954 F_int=0.015542 F_grad=0.051411 TaskLoss=0.213874
Epoch 31/50 FusionLoss=0.067675 F_int=0.015695 F_grad=0.051981 TaskLoss=0.224849
Epoch 32/50 FusionLoss=0.066329 F_int=0.015143 F_grad=0.051186 TaskLoss=0.225592
Epoch 33/50 FusionLoss=0.067667 F_int=0.015705 F_grad=0.051962 TaskLoss=0.212737
Epoch 34/50 FusionLoss=0.064379 F_int=0.014614 F_grad=0.049765 TaskLoss=0.208508
Epoch 35/50 FusionLoss=0.065582 F_int=0.015737 F_grad=0.049845 TaskLoss=0.211416
Epoch 36/50 FusionLoss=0.066723 F_int=0.015287 F_grad=0.051436 TaskLoss=0.217316
Epoch 37/50 FusionLoss=0.065921 F_int=0.015084 F_grad=0.050838 TaskLoss=0.226163
Epoch 38/50 FusionLoss=0.065172 F_int=0.015467 F_grad=0.049704 TaskLoss=0.220939
Epoch 39/50 FusionLoss=0.064901 F_int=0.015013 F_grad=0.049889 TaskLoss=0.206123
Epoch 40/50 FusionLoss=0.063829 F_int=0.013747 F_grad=0.050082 TaskLoss=0.219906
Epoch 41/50 FusionLoss=0.063687 F_int=0.015137 F_grad=0.048550 TaskLoss=0.212825
Epoch 42/50 FusionLoss=0.066133 F_int=0.014863 F_grad=0.051270 TaskLoss=0.222673
Epoch 43/50 FusionLoss=0.066489 F_int=0.015740 F_grad=0.050750 TaskLoss=0.226045
Epoch 44/50 FusionLoss=0.066336 F_int=0.015316 F_grad=0.051021 TaskLoss=0.219745
Epoch 45/50 FusionLoss=0.065181 F_int=0.014398 F_grad=0.050783 TaskLoss=0.212025
Epoch 46/50 FusionLoss=0.065924 F_int=0.015745 F_grad=0.050179 TaskLoss=0.204102
Epoch 47/50 FusionLoss=0.065733 F_int=0.015258 F_grad=0.050475 TaskLoss=0.216710
Epoch 48/50 FusionLoss=0.064814 F_int=0.013796 F_grad=0.051018 TaskLoss=0.216217
Epoch 49/50 FusionLoss=0.066609 F_int=0.016702 F_grad=0.049907 TaskLoss=0.239777
Epoch 50/50 FusionLoss=0.065300 F_int=0.014746 F_grad=0.050554 TaskLoss=0.237785
```

### 9.2 `loss_curve/loss_history.csv`

```text
1.0,0.1861990200355649,0.021727625129860827,0.16447139468044042,0.9378453396260739
2.0,0.10177044942043721,0.01372065924864728,0.08804979026317597,0.38003710709512234
3.0,0.08920560976490378,0.01574829590914305,0.07345731372013688,0.3430905082449317
4.0,0.07506430982612074,0.013590330732404255,0.06147397892549634,0.29563236363232137
5.0,0.07333909352310002,0.014116306686191819,0.059222786761820315,0.31560937870293854
6.0,0.0768752456177026,0.016188952279044315,0.06068629358895123,0.3062846156582236
7.0,0.07243449557572604,0.01496256942016771,0.05747192591428757,0.29683574721217154
8.0,0.07011726767756045,0.015832107313908637,0.054285160144791005,0.26432383976876733
9.0,0.0695707427058369,0.015433680105779786,0.05413706235587597,0.2930604652687907
10.0,0.06644702569581568,0.015579034270485863,0.05086799141019583,0.26544677179306747
11.0,0.06541415261104704,0.014993288003606721,0.05042086452245712,0.2545102860406041
12.0,0.06589917197823525,0.015442686748574488,0.05045648516155779,0.24107610166072846
13.0,0.06452131091617047,0.014432668242952786,0.05008864265866578,0.22384338410571217
14.0,0.06948526016436517,0.01727631934452802,0.05220894105732441,0.25623073427006604
15.0,0.06425742173567414,0.014129798792419023,0.050127622801810504,0.22328729443252088
16.0,0.062059658616781234,0.014532079696655273,0.04752757901325822,0.2399639805406332
17.0,0.06345100678503514,0.014743881032336504,0.04870712579227984,0.24354112777858972
18.0,0.06661883847787976,0.014928921549289953,0.05168991685844958,0.2122868950664997
19.0,0.06413606906309724,0.015604445118806326,0.04853162426501512,0.212800951898098
20.0,0.06762752456590533,0.016166594949318096,0.051460929638706146,0.22958281386643647
21.0,0.06529230831190944,0.015311999545083381,0.049980308841913935,0.2226234438456595
22.0,0.0664466510899365,0.014680314346333035,0.05176633669063449,0.2148305222019553
23.0,0.06537545989267528,0.014727227475959807,0.05064823243767023,0.2008703976869583
24.0,0.06641133052296937,0.015535774589516223,0.050875555919483305,0.2052591341175139
25.0,0.06453587620519102,0.014356527479103534,0.05017934866249561,0.21701855164021253
26.0,0.06415034353733062,0.014013530407100915,0.050136813204735516,0.20444044718518853
27.0,0.06780012987554074,0.016422752058133483,0.05137737770564854,0.21140431677922605
28.0,0.06729979051277041,0.01479483397110016,0.05250495656393468,0.209669286608696
29.0,0.06412666160613298,0.012941357354284264,0.051185303954407575,0.22316492257639767
30.0,0.06695363949984312,0.015542288476135582,0.05141135082580149,0.2138739669509232
31.0,0.06767530481331051,0.01569460709579289,0.051980697782710195,0.22484893966466188
32.0,0.06632902511395514,0.015142921651131474,0.05118610356934369,0.22559160033240913
33.0,0.06766701739281417,0.01570486450509634,0.05196215289644897,0.21273651087656617
34.0,0.06437931491993368,0.014614304724964312,0.04976501018740237,0.20850776640698312
35.0,0.06558230455964803,0.015737102676648646,0.04984520211815834,0.21141641220077872
36.0,0.06672321391291916,0.015286554403137415,0.05143665951676667,0.21731561109423636
37.0,0.06592136031016707,0.015083847799687647,0.05083751242607832,0.22616272630169987
38.0,0.06517178723588586,0.01546741401514737,0.04970437320880592,0.22093940047547223
39.0,0.06490124876610935,0.015012643906811719,0.049888604944571854,0.2061226156540215
40.0,0.06382871520705521,0.01374695755133871,0.050081757577136156,0.21990629682317375
41.0,0.06368702813982964,0.015136936733615585,0.048550091115757824,0.21282515712082387
42.0,0.06613298506475986,0.014862775306100957,0.05127020963467657,0.22267313785851
43.0,0.06648938124999404,0.015739662763662635,0.050749718556180594,0.22604472754523158
44.0,0.066336451144889,0.015315888030163477,0.05102056312840432,0.21974548311904074
45.0,0.0651809240039438,0.014398228214413394,0.05078269548714161,0.21202481446787716
46.0,0.06592364797368645,0.01574457329174038,0.050179074639454486,0.20410205967724324
47.0,0.06573327059857548,0.015257843238068745,0.050475427433848384,0.2167102213948965
48.0,0.06481412896886468,0.013796225802216212,0.05101790300570428,0.21621656470000744
49.0,0.0666092601697892,0.016702263449551537,0.049906996432691815,0.23977668408304453
50.0,0.06529990677721799,0.014745709371345584,0.050554197560995814,0.23778526177629827
```
