#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _split_tokens(line):
    return [token for token in line.strip().split() if token]


def parse_classification_section(lines):
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "Classification Report:":
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("Could not find 'Classification Report:' section")

    header_idx = None
    for idx in range(start_idx + 1, len(lines)):
        tokens = _split_tokens(lines[idx])
        if tokens[:4] == ["precision", "recall", "f1-score", "support"]:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Could not find classification report header")

    class_rows = []
    for idx in range(header_idx + 1, len(lines)):
        raw = lines[idx].strip()
        if not raw:
            continue
        if raw.startswith("accuracy"):
            break
        if raw.startswith("macro avg") or raw.startswith("weighted avg"):
            continue

        tokens = _split_tokens(raw)
        if len(tokens) < 5:
            continue
        label = tokens[0]
        try:
            precision = float(tokens[1])
            recall = float(tokens[2])
            f1 = float(tokens[3])
            support = int(float(tokens[4]))
        except ValueError:
            continue
        class_rows.append((label, precision, recall, f1, support))

    if not class_rows:
        raise ValueError("No class rows found in classification report")

    labels = [row[0] for row in class_rows]
    precision = np.array([row[1] for row in class_rows], dtype=float)
    recall = np.array([row[2] for row in class_rows], dtype=float)
    f1 = np.array([row[3] for row in class_rows], dtype=float)
    support = np.array([row[4] for row in class_rows], dtype=int)
    return labels, precision, recall, f1, support


def parse_confusion_matrix_section(lines):
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "Confusion Matrix:":
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("Could not find 'Confusion Matrix:' section")

    header_idx = start_idx + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    if header_idx >= len(lines):
        raise ValueError("Could not find confusion matrix header row")

    col_labels = _split_tokens(lines[header_idx])
    if not col_labels:
        raise ValueError("Empty confusion matrix header row")

    row_labels = []
    matrix_rows = []

    for idx in range(header_idx + 1, len(lines)):
        raw = lines[idx].strip()
        if not raw:
            continue
        tokens = _split_tokens(raw)
        if len(tokens) < len(col_labels) + 1:
            continue

        row_label = tokens[0]
        values = tokens[1 : 1 + len(col_labels)]
        try:
            values = [float(v) for v in values]
        except ValueError:
            continue

        row_labels.append(row_label)
        matrix_rows.append(values)

    if not matrix_rows:
        raise ValueError("No confusion matrix rows found")

    matrix = np.array(matrix_rows, dtype=float)
    return row_labels, col_labels, matrix


def plot_confusion_matrix(row_labels, col_labels, matrix, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized value")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_classification_report(labels, precision, recall, f1, support, out_path):
    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(x - width, precision, width=width, label="precision", color="#4C78A8")
    ax1.bar(x, recall, width=width, label="recall", color="#F58518")
    ax1.bar(x + width, f1, width=width, label="f1-score", color="#54A24B")

    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_title("Classification Report by Class")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, support, color="#B279A2", marker="o", linewidth=2, label="support")
    ax2.set_ylabel("Support")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Parse classification_report.txt and plot confusion matrix + classification metrics"
    )
    parser.add_argument(
        "--report-file",
        default="/home/nadira/partnr-planner/TDOST/train/classification_report.txt",
        help="Path to classification_report.txt",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--cm-out",
        default="confusion_matrix_from_report.png",
        help="Output filename for confusion matrix plot",
    )
    parser.add_argument(
        "--cr-out",
        default="classification_report_from_report.png",
        help="Output filename for classification report plot",
    )
    args = parser.parse_args()

    report_path = Path(args.report_file)
    if not report_path.exists():
        raise SystemExit(f"Report file not found: {report_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = report_path.read_text(encoding="utf-8").splitlines()

    labels, precision, recall, f1, support = parse_classification_section(lines)
    row_labels, col_labels, matrix = parse_confusion_matrix_section(lines)

    cm_path = out_dir / args.cm_out
    cr_path = out_dir / args.cr_out

    plot_confusion_matrix(row_labels, col_labels, matrix, cm_path)
    plot_classification_report(labels, precision, recall, f1, support, cr_path)

    print(f"Saved confusion matrix plot to {cm_path}")
    print(f"Saved classification report plot to {cr_path}")


if __name__ == "__main__":
    main()
