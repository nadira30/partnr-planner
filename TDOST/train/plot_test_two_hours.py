import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DEFAULT_TIMELINE_START = "2026-05-11T00:00:00"


def summarize_windows(df):
    group_cols = ["window_id", "sequence_id", "window_start_sec", "window_end_sec"]
    rows = []

    for _, sub in df.sort_values(group_cols + ["timestep"]).groupby(group_cols, sort=True):
        pred_mode = sub["pred_label"].mode(dropna=True)
        rows.append(
            {
                "window_id": sub["window_id"].iloc[0],
                "sequence_id": int(sub["sequence_id"].iloc[0]),
                "window_start_sec": int(sub["window_start_sec"].iloc[0]),
                "window_end_sec": int(sub["window_end_sec"].iloc[0]),
                "true_label": sub["true_label"].iloc[0],
                "pred_label": pred_mode.iloc[0] if len(pred_mode) else sub["pred_label"].iloc[0],
            }
        )

    return pd.DataFrame(rows)


def parse_anchor_datetime(raw_value):
    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "timeline start must be an ISO datetime like 2026-05-11T00:00:00"
        ) from exc


def plot_two_hour_test_samples(df, hours=2, out_path=None, figsize=(16, 4), start_datetime=None):
    required = {"window_id", "sequence_id", "window_start_sec", "window_end_sec", "timestep", "true_label", "pred_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"predictions file must contain columns: {sorted(required)}")

    summary = summarize_windows(df)
    if summary.empty:
        raise ValueError("prediction file does not contain any rows to plot")

    summary = summary.sort_values(["window_start_sec", "window_end_sec", "sequence_id", "window_id"]).reset_index(drop=True)
    anchor = start_datetime or parse_anchor_datetime(DEFAULT_TIMELINE_START)
    if anchor is None:
        raise ValueError("unable to determine a start datetime for the timeline")

    # choose a safe start index (use 50 if available, otherwise start at first window)
    start_idx = 0 if len(summary) > 50 else 0
    start_sec = int(summary["window_start_sec"].iloc[start_idx])
    end_sec = start_sec + int(hours *3600)
    summary = summary[(summary["window_start_sec"] >= start_sec) & (summary["window_start_sec"] < end_sec)].reset_index(drop=True)

    if summary.empty:
        raise ValueError("no samples found in the requested two-hour span")

    labels = pd.unique(summary[["true_label", "pred_label"]].values.ravel())
    labels = [label for label in labels if pd.notna(label)]
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = {"Predicted": 0, "Actual": 1}

    for idx, row in summary.iterrows():
        left = mdates.date2num(anchor + timedelta(seconds=int(row["window_start_sec"])))
        width = (int(row["window_end_sec"]) - int(row["window_start_sec"])) / 86400.0
        ax.barh(y_positions["Actual"], width, left=left, height=0.35, color=colors.get(row["true_label"], "#cccccc"), edgecolor="none")
        ax.barh(y_positions["Predicted"], width, left=left, height=0.35, color=colors.get(row["pred_label"], "#cccccc"), edgecolor="none")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Predicted", "Actual"])
    ax.set_xlabel("Actual time")
    ax.set_title(f"Actual vs Predicted activity over the first {hours} hours of test samples")

    ax.xaxis_date()
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlim(
        mdates.date2num(anchor + timedelta(seconds=start_sec)),
        mdates.date2num(anchor + timedelta(seconds=end_sec)),
    )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", nargs="?", default="predictions.tsv", help="TSV file produced by TDOST/train/main.py")
    parser.add_argument("--hours", type=float, default=24.0, help="How many hours of test samples to visualize")
    parser.add_argument("--out", default="plot_two_hour_test_samples.png", help="Output image path (png)")
    parser.add_argument(
        "--timeline-start-datetime",
        default=None,
        help="Optional anchor datetime for the x-axis, for example 2026-05-11T00:00:00",
    )
    args = parser.parse_args()

    path = Path(args.pred_file)
    if not path.exists():
        raise SystemExit(f"Prediction file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    start_datetime = parse_anchor_datetime(args.timeline_start_datetime) if args.timeline_start_datetime else None
    plot_two_hour_test_samples(df, hours=args.hours, out_path=args.out, start_datetime=start_datetime)


if __name__ == "__main__":
    main()