#!/usr/bin/env python3
"""
Repair invalid name_to_receptacle mappings in an exported episode JSON/JSON.GZ file.

Strategy:
1) Discover active simulator receptacles for the episode.
2) For each object->receptacle mapping:
   - Keep as-is if it is 'floor' or already active.
   - Otherwise, replace with an active receptacle sharing the same parent handle.
3) Save repaired episode to output file.

Example:
    conda run -n habitat python visualization/episode_info/repair_episode_receptacles.py \
      --episode-file visualization/data/episode_100_modified_2026-03-05_16-02-45.json.gz \
      --dataset-path data/datasets/partnr_episodes/v0_0/val_mini.json.gz
"""

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_json_or_gz(path: Path) -> Dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_or_gz(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _episode_from_dataset_or_single(payload: Dict) -> Tuple[Dict, bool]:
    if isinstance(payload, dict) and "episodes" in payload and payload["episodes"]:
        return payload["episodes"][0], True
    return payload, False


def _base_handle(handle: str) -> str:
    return handle.split("_:")[0] if "_:" in handle else handle


def _discover_receptacles(
    project_root: Path,
    episode_id: str,
    dataset_path: str,
    temp_output: Path,
) -> Dict:
    script = project_root / "visualization" / "episode_info" / "get_episode_receptacles.py"
    cmd = [
        sys.executable,
        str(script),
        f"+episode_id={episode_id}",
        f"+dataset_path={dataset_path}",
        f"+output_file={temp_output}",
        "hydra.run.dir=.",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Receptacle discovery failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not temp_output.exists():
        raise RuntimeError(f"Discovery output not found: {temp_output}")

    payload = _read_json_or_gz(temp_output)
    return payload.get("receptacles", {})


def _pick_replacement(receptacles: Dict, current_value: str) -> Optional[str]:
    if "|" not in current_value:
        return None

    parent = current_value.split("|", 1)[0]
    parent_base = _base_handle(parent)

    by_parent_handle = receptacles.get("by_parent_handle", {})
    by_parent_base = receptacles.get("by_parent_base", {})

    candidates: List[str] = []
    for key in (parent, parent_base, f"{parent_base}_:0000"):
        candidates.extend(by_parent_handle.get(key, []))
    candidates.extend(by_parent_base.get(parent_base, []))

    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    if not deduped:
        return None

    # If current mesh token matches any candidate token, prefer that.
    current_mesh = current_value.split("|", 1)[1] if "|" in current_value else ""
    current_mesh_base = current_mesh.split(".")[0]
    for c in deduped:
        c_mesh = c.split("|", 1)[1] if "|" in c else ""
        if c_mesh.split(".")[0] == current_mesh_base:
            return c

    return deduped[0]


def repair_episode_receptacles(
    episode_file: Path,
    dataset_path: str,
    output_file: Optional[Path] = None,
    keep_temp: bool = False,
) -> Dict:
    project_root = Path(__file__).resolve().parent.parent.parent

    payload = _read_json_or_gz(episode_file)
    episode, wrapped = _episode_from_dataset_or_single(payload)

    episode_id = str(episode.get("episode_id", ""))
    if not episode_id:
        raise ValueError("episode_id missing from episode file")

    temp_output = project_root / "visualization" / f"temp_episode_{episode_id}_receptacles_repair.json"
    receptacles = _discover_receptacles(project_root, episode_id, dataset_path, temp_output)
    active = set(receptacles.get("all", []))

    mapping = episode.get("name_to_receptacle", {})
    if not isinstance(mapping, dict):
        raise ValueError("name_to_receptacle missing or invalid")

    original_items = list(mapping.items())
    repaired_items = []

    unchanged_valid = 0
    unchanged_floor = 0
    repaired = 0
    unresolved = 0
    unresolved_examples: List[Tuple[str, str]] = []

    for obj_handle, rec_value in original_items:
        if rec_value == "floor":
            repaired_items.append((obj_handle, rec_value))
            unchanged_floor += 1
            continue

        if rec_value in active:
            repaired_items.append((obj_handle, rec_value))
            unchanged_valid += 1
            continue

        replacement = _pick_replacement(receptacles, rec_value)
        if replacement is None:
            repaired_items.append((obj_handle, rec_value))
            unresolved += 1
            if len(unresolved_examples) < 12:
                unresolved_examples.append((obj_handle, rec_value))
        else:
            repaired_items.append((obj_handle, replacement))
            repaired += 1

    episode["name_to_receptacle"] = dict(repaired_items)

    if wrapped:
        payload["episodes"][0] = episode
    else:
        payload = episode

    if output_file is None:
        suffix = "_receptacles_repaired"
        if episode_file.suffix == ".gz":
            stem = episode_file.name[:-8] if episode_file.name.endswith(".json.gz") else episode_file.stem
            output_file = episode_file.with_name(f"{stem}{suffix}.json.gz")
        else:
            output_file = episode_file.with_name(f"{episode_file.stem}{suffix}{episode_file.suffix}")

    _write_json_or_gz(output_file, payload)

    if not keep_temp:
        try:
            temp_output.unlink(missing_ok=True)
        except Exception:
            pass

    # Post-check
    repaired_mapping = episode.get("name_to_receptacle", {})
    still_invalid = [v for v in repaired_mapping.values() if v != "floor" and v not in active]

    return {
        "episode_id": episode_id,
        "output_file": str(output_file),
        "active_receptacles": len(active),
        "total_mappings": len(repaired_mapping),
        "unchanged_valid": unchanged_valid,
        "unchanged_floor": unchanged_floor,
        "repaired": repaired,
        "unresolved": unresolved,
        "still_invalid_after_repair": len(still_invalid),
        "unresolved_examples": unresolved_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair invalid episode receptacle mappings")
    parser.add_argument("--episode-file", required=True, help="Path to episode JSON or JSON.GZ file")
    parser.add_argument(
        "--dataset-path",
        default="data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
        help="Dataset path used to load the same episode scene",
    )
    parser.add_argument("--output-file", default=None, help="Optional output path")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary receptacle discovery JSON")
    args = parser.parse_args()

    summary = repair_episode_receptacles(
        episode_file=Path(args.episode_file),
        dataset_path=args.dataset_path,
        output_file=Path(args.output_file) if args.output_file else None,
        keep_temp=args.keep_temp,
    )

    print("\nReceptacle Repair Summary")
    print("=" * 60)
    for k, v in summary.items():
        if k == "unresolved_examples":
            continue
        print(f"{k}: {v}")

    if summary["unresolved_examples"]:
        print("unresolved_examples:")
        for obj, rec in summary["unresolved_examples"]:
            print(f"  {obj} -> {rec}")


if __name__ == "__main__":
    main()
