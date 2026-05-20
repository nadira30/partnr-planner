import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MAX_LEN = 100
WINDOW_SECONDS = 120
DAY_SECONDS = 24 * 3600

BATCH_SIZE = 32
EPOCHS = 90
PATIENCE = 5
MIN_DELTA = 1e-4

LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

TDOST_INPUT_PATH = Path("/home/nadira/partnr-planner/TDOST/tdost_dense_test/actual/")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_STORE_PATH = ARTIFACT_DIR / "tdost_windows.jsonl"
CACHE_DIR = ARTIFACT_DIR / "window_embedding_cache"

LOSS_PLOT_PATH = ARTIFACT_DIR / "training_loss.png"
REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.tsv"
CHECKPOINT_PATH = ARTIFACT_DIR / "model_checkpoint.pt"


# ---------------------------------------------------------------------
# Label and time utilities
# ---------------------------------------------------------------------

def canonicalize_activity_label(label):
    normalized = label.strip().lower()
    normalized = normalized.replace(" ", "_")

    mapping = {
        "bed_to_toilet": "bed_to_toilet",
        "cooking": "cook",
        "putting_away_groceries": "cook",
        "eating": "eat",
        "drinking": "eat",
        "work": "work",
        "working": "work",
        "housekeeping": "work",
        "wash_dishes": "work",
        "washing_dishes": "work",
        "reading": "relax",
        "watch_tv": "relax",
        "watching_tv": "relax",
        "sleep": "sleep",
        "sleeping": "sleep",
        "exercising": "other",
        "meditating": "other",
        "painting": "other",
        "waking_up": "other",
        "volunteering": "other",
        "leaving_home": "other",
        "gardening": "other",
        "entering_home": "other",
        "enter_home": "other",
    }

    return mapping.get(normalized, normalized)


def parse_hms_seconds(raw_event):
    """
    Extract trailing HH:MM:SS from raw_event and convert to seconds.
    """
    tokens = raw_event.strip().split()
    if not tokens:
        return None

    candidate = tokens[-1]
    parts = candidate.split(":")

    if len(parts) != 3:
        return None

    try:
        hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None

    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        return None

    return hh * 3600 + mm * 60 + ss


# ---------------------------------------------------------------------
# Lazy TDOST file reading
# ---------------------------------------------------------------------

def iter_tdost_rows(path):
    """
    Lazily read TDOST files row-by-row.

    This avoids loading every TDOST row into one large DataFrame.
    """
    input_path = Path(path)

    if input_path.is_dir():
        file_paths = sorted(p for p in input_path.glob("*.tsv") if p.is_file())
    else:
        file_paths = [input_path]

    if not file_paths:
        raise FileNotFoundError(f"No TDOST .txt files found in {input_path}")

    skipped = 0
    llm_errors = 0
    valid = 0

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip("\n")

                if not line.strip():
                    continue

                # Must have exactly 3 tabs to split into 4 columns.
                if line.count("\t") != 3:
                    skipped += 1
                    continue

                raw_event, structured_event, descriptions, label = [
                    part.strip() for part in line.split("\t")
                ]

                # Skip rows with empty raw_event or structured_event.
                if not raw_event or not structured_event:
                    llm_errors += 1
                    continue

                # Skip rows where descriptions look like LLM error messages.
                if descriptions.startswith("I'm sorry") or descriptions.startswith("Sorry"):
                    llm_errors += 1
                    continue

                time_sec = parse_hms_seconds(raw_event)
                if time_sec is None:
                    skipped += 1
                    continue

                valid += 1

                yield {
                    "raw_event": raw_event,
                    "structured_event": structured_event,
                    "descriptions": descriptions,
                    "label": label,
                    "activity_label": canonicalize_activity_label(label),
                    "time_sec": time_sec,
                    "source_file": file_path.name,
                    "source_line": line_num,
                }

    print(f"Loaded {valid} valid rows lazily")

    if llm_errors > 0:
        print(f"  (filtered out {llm_errors} LLM error rows)")

    if skipped > 0:
        print(f"  (skipped {skipped} malformed/invalid rows)")


def iter_file_groups(path):
    """
    Group rows by source file while streaming.

    This keeps memory bounded by a single source file.
    """
    current_file = None
    rows = []

    for row in iter_tdost_rows(path):
        if current_file is None:
            current_file = row["source_file"]

        if row["source_file"] != current_file:
            yield current_file, rows
            current_file = row["source_file"]
            rows = []

        rows.append(row)

    if rows:
        yield current_file, rows


def add_rollover_for_file(rows):
    """
    Add absolute_time_sec and day_index for one source file.

    Midnight rollover is handled per file.
    """
    rows = sorted(rows, key=lambda r: r["source_line"])

    day_offset = 0
    prev_t = None

    for row in rows:
        t = row["time_sec"]

        if prev_t is not None and t < prev_t:
            day_offset += DAY_SECONDS

        absolute_time_sec = int(t) + day_offset

        row["absolute_time_sec"] = absolute_time_sec
        row["day_index"] = absolute_time_sec // DAY_SECONDS

        prev_t = t

    return rows


# ---------------------------------------------------------------------
# Window store creation
# ---------------------------------------------------------------------

def build_window_store(input_path, output_path):
    """
    Build a lightweight JSONL store of windows.

    Each JSONL row stores text and metadata only.
    Embeddings are not computed here.
    """
    metadata = []
    all_labels = []

    global_sequence_id = -1
    previous_source_file = None
    previous_activity = None

    padded_count = 0
    truncated_count = 0
    seq_lengths = []

    with open(output_path, "w", encoding="utf-8") as out:
        for source_file, file_rows in iter_file_groups(input_path):
            file_rows = add_rollover_for_file(file_rows)

            # Assign sequence ids.
            for row in file_rows:
                source_changed = row["source_file"] != previous_source_file
                activity_changed = row["activity_label"] != previous_activity

                if source_changed or activity_changed:
                    global_sequence_id += 1

                row["sequence_id"] = global_sequence_id

                previous_source_file = row["source_file"]
                previous_activity = row["activity_label"]

            # Use a per-file DataFrame only.
            df_file = pd.DataFrame(file_rows)

            min_time_by_seq = df_file.groupby("sequence_id")["absolute_time_sec"].transform("min")

            df_file["window_idx"] = (
                (df_file["absolute_time_sec"] - min_time_by_seq) // WINDOW_SECONDS
            ).astype(int)

            for (seq_id, win_idx), group in df_file.groupby(
                ["sequence_id", "window_idx"],
                sort=True,
            ):
                group = group.sort_values("absolute_time_sec")

                label = group["activity_label"].iloc[0]
                texts = group["descriptions"].tolist()

                original_len = len(texts)
                seq_len = min(original_len, MAX_LEN)

                seq_lengths.append(seq_len)

                if original_len < MAX_LEN:
                    padded_count += 1
                elif original_len > MAX_LEN:
                    truncated_count += 1

                start = int(group["absolute_time_sec"].min())
                window_start = start - (start % WINDOW_SECONDS)
                window_end = window_start + WINDOW_SECONDS

                day_key = f"{group['source_file'].iloc[0]}::{int(group['day_index'].iloc[0])}"

                window_record = {
                    "texts": texts[:MAX_LEN],
                    "label": label,
                    "sequence_id": int(seq_id),
                    "window_idx": int(win_idx),
                    "window_id": f"{seq_id}_{win_idx}",
                    "source_file": group["source_file"].iloc[0],
                    "day_index": int(group["day_index"].iloc[0]),
                    "day_key": day_key,
                    "window_start_sec": window_start,
                    "window_end_sec": window_end,
                    "seq_len": seq_len,
                    "original_len": original_len,
                }

                offset = out.tell()
                out.write(json.dumps(window_record) + "\n")

                metadata.append({
                    "offset": offset,
                    "label": label,
                    "sequence_id": int(seq_id),
                    "window_id": f"{seq_id}_{win_idx}",
                    "day_key": day_key,
                    "window_start_sec": window_start,
                    "window_end_sec": window_end,
                    "seq_len": seq_len,
                    "original_len": original_len,
                })

                all_labels.append(label)

    print(f"\nWindowing config: WINDOW_SECONDS={WINDOW_SECONDS}, MAX_LEN={MAX_LEN}")
    print(f"  Built windows: {len(metadata)}")
    print(f"  Padded windows: {padded_count}")
    print(f"  Truncated windows: {truncated_count}")

    if seq_lengths:
        print(
            "  Events per window before pad/truncate: "
            f"mean={float(np.mean(seq_lengths)):.2f}, max={int(np.max(seq_lengths))}"
        )

    print(f"Saved lightweight window store to {output_path}")

    return metadata, all_labels


def filter_rare_classes(metadata):
    """
    Remove labels with fewer than 2 samples.

    This mirrors the original rare-class cleanup, but works on metadata instead of X/y arrays.
    """
    labels = [m["label"] for m in metadata]
    label_counts = pd.Series(labels).value_counts()

    rare_labels = set(label_counts[label_counts < 2].index)

    if not rare_labels:
        return metadata

    removed = sum(m["label"] in rare_labels for m in metadata)

    print(f"\nWarning: Classes with < 2 samples: {sorted(rare_labels)}")
    print(f"Removing {removed} windows with rare classes...")

    filtered = [m for m in metadata if m["label"] not in rare_labels]

    print(f"After filtering: {len(filtered)} windows retained")

    return filtered


# ---------------------------------------------------------------------
# Split and class weights
# ---------------------------------------------------------------------

def chronological_split(metadata):
    """
    Split by final 10% of days, preserving chronological order.
    """
    day_df = pd.DataFrame({
        "day_key": [m["day_key"] for m in metadata],
        "window_start_sec": [m["window_start_sec"] for m in metadata],
    })

    day_order = (
        day_df
        .groupby("day_key", sort=False)["window_start_sec"]
        .min()
        .sort_values()
    )

    unique_days = day_order.index.tolist()

    if len(unique_days) < 2:
        raise ValueError("Need at least 2 distinct days to create a chronological train/test split")

    num_days = len(unique_days)
    test_day_count = max(1, int(math.ceil(num_days * 0.10)))

    if test_day_count >= num_days:
        test_day_count = 1

    train_day_count = num_days - test_day_count

    train_days = set(unique_days[:train_day_count])
    test_days = set(unique_days[train_day_count:])

    train_meta = [m for m in metadata if m["day_key"] in train_days]
    test_meta = [m for m in metadata if m["day_key"] in test_days]

    print(f"\nDay-based split: {len(train_days)} train days, {len(test_days)} test days")
    print(f"  Train set: {len(train_meta)} windows")
    print(f"  Test set: {len(test_meta)} windows")

    return train_meta, test_meta


def compute_class_weights(train_meta, label_encoder):
    y_train = label_encoder.transform([m["label"] for m in train_meta])

    num_classes = len(label_encoder.classes_)
    counts = np.bincount(y_train, minlength=num_classes)

    # Avoid division by zero for classes absent in the chronological train split.
    counts = np.where(counts == 0, 1, counts)

    class_weights = len(y_train) / (num_classes * counts)

    return torch.tensor(class_weights, dtype=torch.float).to(DEVICE)


# ---------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------

def build_tensor_cache(
    jsonl_path,
    metadata,
    label_encoder,
    embedder,
    cache_dir,
    embedding_dim,
):
    """
    Encode each window once and save it as a .npy file.

    Training then reads these cached files instead of re-running SentenceTransformer
    every epoch.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_metadata = []

    for i, item in enumerate(metadata):
        cache_path = cache_dir / f"window_{i:08d}.npy"

        if not cache_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                f.seek(item["offset"])
                record = json.loads(f.readline())

            texts = record["texts"][:MAX_LEN]

            X = np.zeros((MAX_LEN, embedding_dim), dtype=np.float32)

            if texts:
                embeddings = embedder.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

                X[: len(texts), :] = embeddings

            np.save(cache_path, X)

        cached_item = dict(item)
        cached_item["cache_path"] = str(cache_path)
        cached_item["label_id"] = int(label_encoder.transform([item["label"]])[0])

        cached_metadata.append(cached_item)

        if (i + 1) % 500 == 0:
            print(f"Cached {i + 1}/{len(metadata)} windows")

    print(f"Embedding cache ready at {cache_dir}")

    return cached_metadata


# ---------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------

class CachedWindowDataset(Dataset):
    """
    Loads precomputed window tensors from disk.

    mmap_mode='r' helps avoid pulling every cached tensor into RAM at once.
    """
    def __init__(self, metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        X = np.load(item["cache_path"], mmap_mode="r")
        X = np.asarray(X, dtype=np.float32)

        return {
            "X": torch.from_numpy(X).float(),
            "lengths": torch.tensor(item["seq_len"], dtype=torch.long),
            "y": torch.tensor(item["label_id"], dtype=torch.long),
            "meta": item,
        }


def cached_collate(batch):
    X = torch.stack([item["X"] for item in batch])
    lengths = torch.stack([item["lengths"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    meta = [item["meta"] for item in batch]

    return {
        "X": X,
        "lengths": lengths,
        "y": y,
        "meta": meta,
    }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class TDOSTBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, return_per_timestep=False):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)

        # Mask padded timesteps.
        with torch.no_grad():
            mask = (x.abs().sum(dim=2) > 0).to(torch.float)

        if return_per_timestep:
            return self.classifier(out)

        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        summed = (out * mask.unsqueeze(2)).sum(dim=1)
        pooled = summed / lengths

        pooled = self.dropout(pooled)

        return self.classifier(pooled)


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_model(model, train_loader, criterion, optimizer):
    train_losses = []

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            Xb = batch["X"].to(DEVICE, non_blocking=True)
            Yb = batch["y"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(Xb)
                loss = criterion(logits, Yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            del Xb, Yb, logits, loss

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}, loss={epoch_loss:.4f}")

        if epoch_loss < best_loss - MIN_DELTA:
            best_loss = epoch_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses


def save_training_loss_plot(train_losses, output_path):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", linewidth=2)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved training loss plot to {output_path}")


def evaluate_model(model, test_loader, label_encoder):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in test_loader:
            Xb = batch["X"].to(DEVICE, non_blocking=True)
            Yb = batch["y"]

            logits = model(Xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_true.extend(Yb.numpy().tolist())

            del Xb, Yb, logits

    report = classification_report(
        all_true,
        all_preds,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    cm = confusion_matrix(all_true, all_preds, normalize="true")
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm_df)

    return all_true, all_preds, report, cm_df


def save_report(report, cm_df, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

    print(f"Saved classification report and confusion matrix to {output_path}")


def save_per_timestep_predictions(model, test_loader, label_encoder, output_path):
    model.eval()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "window_id",
            "sequence_id",
            "window_start_sec",
            "window_end_sec",
            "timestep",
            "true_label",
            "pred_label",
            "probs",
        ])

        with torch.no_grad():
            for batch in test_loader:
                Xb = batch["X"].to(DEVICE, non_blocking=True)
                Yb = batch["y"]
                meta = batch["meta"]

                logits = model(Xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                per_ts_logits = model(Xb, return_per_timestep=True)
                per_ts_preds = torch.argmax(per_ts_logits, dim=2).cpu().numpy()

                for b, item in enumerate(meta):
                    seq_len = int(item["seq_len"])

                    true_label_id = int(Yb[b].item())
                    true_label = label_encoder.classes_[true_label_id]

                    for t in range(seq_len):
                        pred_label_id = int(per_ts_preds[b, t])
                        pred_label = label_encoder.classes_[pred_label_id]

                        writer.writerow([
                            item["window_id"],
                            item["sequence_id"],
                            item["window_start_sec"],
                            item["window_end_sec"],
                            t,
                            true_label,
                            pred_label,
                            probs[b].tolist(),
                        ])

                del Xb, Yb, logits, probs, per_ts_logits, per_ts_preds

    print(f"Saved per-timestep predictions to {output_path}")


def save_checkpoint(model, label_encoder, embedding_dim, output_path):
    checkpoint = {
        "model_state": model.state_dict(),
        "label_encoder_classes": label_encoder.classes_,
        "embedding_dim": embedding_dim,
        "hidden_dim": HIDDEN_DIM,
        "max_len": MAX_LEN,
        "window_seconds": WINDOW_SECONDS,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
    }

    torch.save(checkpoint, output_path)

    print(f"Saved model checkpoint to {output_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # -------------------------------------------------------------
    # 1. Build lightweight JSONL window store.
    # -------------------------------------------------------------
    metadata, _ = build_window_store(
        input_path=TDOST_INPUT_PATH,
        output_path=WINDOW_STORE_PATH,
    )

    metadata = filter_rare_classes(metadata)

    labels_after_filter = [m["label"] for m in metadata]

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_after_filter)

    print(f"\nClasses: {label_encoder.classes_}")

    # -------------------------------------------------------------
    # 2. Chronological split before cache training.
    # -------------------------------------------------------------
    train_meta, test_meta = chronological_split(metadata)

    class_weights = compute_class_weights(train_meta, label_encoder)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # -------------------------------------------------------------
    # 3. Load embedder once and build disk cache once.
    # -------------------------------------------------------------
    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=DEVICE,
    )

    probe_embedding = embedder.encode(
        ["probe"],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embedding_dim = int(probe_embedding.shape[1])
    del probe_embedding

    cached_metadata = build_tensor_cache(
        jsonl_path=WINDOW_STORE_PATH,
        metadata=metadata,
        label_encoder=label_encoder,
        embedder=embedder,
        cache_dir=CACHE_DIR,
        embedding_dim=embedding_dim,
    )

    # Free SentenceTransformer memory before LSTM training.
    del embedder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Split cached metadata using the same chronological logic.
    train_cached_meta, test_cached_meta = chronological_split(cached_metadata)

    # -------------------------------------------------------------
    # 4. DataLoaders.
    # -------------------------------------------------------------
    train_dataset = CachedWindowDataset(train_cached_meta)
    test_dataset = CachedWindowDataset(test_cached_meta)

    # persistent_workers requires num_workers > 0.
    num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=cached_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=cached_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # -------------------------------------------------------------
    # 5. Model.
    # -------------------------------------------------------------
    model = TDOSTBiLSTM(
        input_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(label_encoder.classes_),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------------------------------------
    # 6. Train.
    # -------------------------------------------------------------
    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    save_training_loss_plot(train_losses, LOSS_PLOT_PATH)

    # -------------------------------------------------------------
    # 7. Evaluate.
    # -------------------------------------------------------------
    _, _, report, cm_df = evaluate_model(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
    )

    save_report(report, cm_df, REPORT_PATH)

    # -------------------------------------------------------------
    # 8. Stream predictions.
    # -------------------------------------------------------------
    save_per_timestep_predictions(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        output_path=PREDICTIONS_PATH,
    )

    # -------------------------------------------------------------
    # 9. Save checkpoint.
    # -------------------------------------------------------------
    save_checkpoint(
        model=model,
        label_encoder=label_encoder,
        embedding_dim=embedding_dim,
        output_path=CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()