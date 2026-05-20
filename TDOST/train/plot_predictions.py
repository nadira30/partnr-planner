import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def coalesce_segments(labels):
    # labels: list of label strings for consecutive timesteps
    segs = []
    if len(labels) == 0:
        return segs
    cur = labels[0]
    start = 0
    for i, l in enumerate(labels[1:], start=1):
        if l != cur:
            segs.append((start, i - start, cur))
            cur = l
            start = i
    segs.append((start, len(labels) - start, cur))
    return segs


def summarize_duration_windows(df):
    group_cols = ["window_id", "sequence_id", "window_start_sec", "window_end_sec"]
    rows = []

    for _, sub in df.sort_values(group_cols + ["timestep"]).groupby(group_cols, sort=True):
        true_label = sub["true_label"].iloc[0]
        pred_mode = sub["pred_label"].mode(dropna=True)
        pred_label = pred_mode.iloc[0] if len(pred_mode) else sub["pred_label"].iloc[0]
        rows.append({
            "window_id": sub["window_id"].iloc[0],
            "sequence_id": int(sub["sequence_id"].iloc[0]),
            "window_start_sec": int(sub["window_start_sec"].iloc[0]),
            "window_end_sec": int(sub["window_end_sec"].iloc[0]),
            "true_label": true_label,
            "pred_label": pred_label,
        })

    return pd.DataFrame(rows)


def plot_test_sample_predictions(df, out_path=None, figsize=(16, 4)):
    if {"window_id", "sequence_id", "window_start_sec", "window_end_sec"}.issubset(df.columns):
        summary = summarize_duration_windows(df)
    else:
        raise ValueError("test-sample plot mode requires duration-aware predictions")

    summary = summary.sort_values(["window_start_sec", "window_end_sec", "sequence_id", "window_id"]).reset_index(drop=True)
    if summary.empty:
        raise ValueError("prediction file does not contain any rows to plot")

    labels = pd.unique(summary[["true_label", "pred_label"]].values.ravel())
    labels = [label for label in labels if pd.notna(label)]
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize)
    sample_positions = range(len(summary))
    bar_height = 0.35

    for idx, row in summary.iterrows():
        ax.barh(1, 1, left=idx, height=bar_height, color=colors.get(row["true_label"], "#cccccc"), edgecolor="none")
        ax.barh(0, 1, left=idx, height=bar_height, color=colors.get(row["pred_label"], "#cccccc"), edgecolor="none")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Predicted", "Actual"])
    ax.set_xlim(0, len(summary))
    ax.set_xlabel("Test sample order")
    ax.set_title("Actual vs Predicted activity over the test samples")

    tick_count = min(12, len(summary))
    if len(summary) > 0:
        tick_positions = sorted(set(int(round(x)) for x in np.linspace(0, len(summary) - 1, tick_count)))
        tick_labels = []
        for pos in tick_positions:
            row = summary.iloc[pos]
            tick_labels.append(f"{int(row['sequence_id'])}:{int(row['window_start_sec'])}")
        ax.set_xticks([p + 0.5 for p in tick_positions])
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.grid(axis="x", linestyle="--", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", title="Activity")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def build_timeline_segments(df, label_col):
    ordered = df.sort_values(["window_start_sec", "window_end_sec", "sequence_id", "window_id"]).reset_index(drop=True)
    if ordered.empty:
        return []

    segments = []
    current_start = int(ordered.iloc[0]["window_start_sec"])
    current_end = int(ordered.iloc[0]["window_end_sec"])
    current_label = ordered.iloc[0][label_col]

    for _, row in ordered.iloc[1:].iterrows():
        row_start = int(row["window_start_sec"])
        row_end = int(row["window_end_sec"])
        row_label = row[label_col]

        if row_label == current_label and row_start == current_end:
            current_end = row_end
            continue

        segments.append((current_start, current_end, current_label))
        current_start = row_start
        current_end = row_end
        current_label = row_label

    segments.append((current_start, current_end, current_label))
    return segments


def plot_predictions(df, out_path=None, figsize=(12, 6)):
    # Supports both schemas:
    # 1) legacy: sequence_id, timestep, true_label, pred_label
    # 2) duration-aware: window_id, sequence_id, window_start_sec, window_end_sec, timestep, true_label, pred_label
    if "window_id" in df.columns:
        group_col = "window_id"
        display_col = "window_id"
    else:
        group_col = "sequence_id"
        display_col = "sequence_id"

    sort_cols = [group_col, "timestep"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    groups = df[group_col].unique()

    # collect label set
    labels = pd.unique(df[["true_label", "pred_label"]].values.ravel())
    labels = [l for l in labels if pd.notna(l)]
    cmap = plt.get_cmap("tab20")
    colors = {l: cmap(i % 20) for i, l in enumerate(labels)}

    n_seq = len(groups)
    height = max(4, n_seq * 0.5)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    y = 0
    yticks = []
    yticklabels = []
    for group in groups:
        sub = df[df[group_col] == group]
        true_labels = sub["true_label"].tolist()
        pred_labels = sub["pred_label"].tolist()

        # coalesce contiguous segments
        t_segs = coalesce_segments(true_labels)
        p_segs = coalesce_segments(pred_labels)

        # plot true labels on top row for this sequence
        for start, length, lab in t_segs:
            ax.barh(y + 0.25, length, left=start, height=0.4, color=colors.get(lab, "#cccccc"))
        # plot predicted labels on bottom row for this sequence
        for start, length, lab in p_segs:
            ax.barh(y - 0.25, length, left=start, height=0.4, color=colors.get(lab, "#cccccc"), alpha=0.9)

        yticks.append(y)
        if {"window_start_sec", "window_end_sec"}.issubset(df.columns):
            start = int(sub["window_start_sec"].iloc[0])
            end = int(sub["window_end_sec"].iloc[0])
            yticklabels.append(f"{display_col}={group} [{start}-{end}s]")
        else:
            yticklabels.append(f"{display_col}={group}")
        y -= 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Timestep")
    ax.set_title("True (top) vs Predicted (bottom) per-sequence timeline")

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[l]) for l in labels]
    ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pred_file", nargs="?", default="predictions.tsv",
                   help="TSV file with either legacy or duration-aware prediction columns")
    p.add_argument("--out", default=None, help="Output image path (png)")
    args = p.parse_args()

    path = Path(args.pred_file)
    if not path.exists():
        raise SystemExit(f"Prediction file not found: {path}. Create a TSV with columns sequence_id,timestep,true_label,pred_label")

    df = pd.read_csv(path, sep="\t")
    required_common = {"timestep", "true_label", "pred_label"}
    if not required_common.issubset(df.columns):
        raise SystemExit(f"pred_file must contain columns: {required_common}")

    has_legacy = {"sequence_id"}.issubset(df.columns)
    has_duration = {"window_id", "sequence_id", "window_start_sec", "window_end_sec"}.issubset(df.columns)
    if not (has_legacy or has_duration):
        raise SystemExit(
            "pred_file must contain either legacy columns (sequence_id, timestep, true_label, pred_label) "
            "or duration-aware columns (window_id, sequence_id, window_start_sec, window_end_sec, timestep, true_label, pred_label)"
        )

    if {"window_id", "sequence_id", "window_start_sec", "window_end_sec"}.issubset(df.columns):
        plot_test_sample_predictions(df, out_path=args.out)
    else:
        plot_predictions(df, out_path=args.out)


if __name__ == "__main__":
    main()
