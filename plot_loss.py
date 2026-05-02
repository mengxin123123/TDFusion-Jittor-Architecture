import csv
import os

import matplotlib.pyplot as plt


CSV_PATH = '/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/loss_curve/loss_history.csv'
OUT_PATH = '/root/autodl-tmp/TDFusion-main_jittor/exp/04_30_02_00/loss_curve/loss_curve.png'


def main():
    epochs, fusion, f_int, f_grad, task = [], [], [], [], []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            epochs.append(int(float(row[0])))
            fusion.append(float(row[1]))
            f_int.append(float(row[2]))
            f_grad.append(float(row[3]))
            task.append(float(row[4]))

    plt.figure(figsize=(10, 6), dpi=160)
    plt.plot(epochs, fusion, label='FusionLoss', linewidth=2)
    plt.plot(epochs, f_int, label='F_int', linewidth=2)
    plt.plot(epochs, f_grad, label='F_grad', linewidth=2)
    plt.plot(epochs, task, label='TaskLoss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    print(OUT_PATH)


if __name__ == '__main__':
    main()
