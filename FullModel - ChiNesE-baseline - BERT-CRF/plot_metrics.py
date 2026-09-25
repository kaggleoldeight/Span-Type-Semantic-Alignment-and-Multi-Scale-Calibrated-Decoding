# -*- coding: utf-8 -*-
"""从训练日志 CSV 绘制 F1/Precision/Recall/Loss 曲线。"""
import csv
from typing import List
import matplotlib.pyplot as plt
from config import TRAIN_LOG_CSV, METRICS_F1_FIG_PATH, METRICS_PRECISION_FIG_PATH, METRICS_RECALL_FIG_PATH, METRICS_LOSS_FIG_PATH


def read_log(csv_path: str):
    epochs: List[int] = []
    loss: List[float] = []
    f1: List[float] = []
    precision: List[float] = []
    recall: List[float] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            loss.append(float(row['loss']))
            f1.append(float(row['f1']))
            precision.append(float(row['precision']))
            recall.append(float(row['recall']))
    return epochs, loss, f1, precision, recall


def plot_curves():
    epochs, loss, f1, precision, recall = read_log(TRAIN_LOG_CSV)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1, label='F1', marker='o')
    plt.xlabel('Epoch')
    plt.title('F1 Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    if len(f1) > 0:
        best_idx = max(range(len(f1)), key=lambda i: f1[i])
        best_epoch = epochs[best_idx]
        best_f1 = f1[best_idx]
        plt.scatter([best_epoch], [best_f1], color='red', zorder=5)
        plt.annotate(f'Best F1 = {best_f1:.6f} @ epoch {best_epoch}',
                     xy=(best_epoch, best_f1),
                     xytext=(best_epoch, best_f1 + 0.01),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')
    plt.tight_layout()
    plt.savefig(METRICS_F1_FIG_PATH, dpi=200)
    plt.close()
    print(f'Saved F1 figure to: {METRICS_F1_FIG_PATH}')

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, precision, label='Precision', marker='^')
    plt.xlabel('Epoch')
    plt.title('Precision Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(METRICS_PRECISION_FIG_PATH, dpi=200)
    plt.close()
    print(f'Saved Precision figure to: {METRICS_PRECISION_FIG_PATH}')

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, recall, label='Recall', marker='s')
    plt.xlabel('Epoch')
    plt.title('Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(METRICS_RECALL_FIG_PATH, dpi=200)
    plt.close()
    print(f'Saved Recall figure to: {METRICS_RECALL_FIG_PATH}')

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Loss', marker='x')
    plt.xlabel('Epoch')
    plt.title('Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(METRICS_LOSS_FIG_PATH, dpi=200)
    plt.close()
    print(f'Saved Loss figure to: {METRICS_LOSS_FIG_PATH}')


if __name__ == '__main__':
    plot_curves()
