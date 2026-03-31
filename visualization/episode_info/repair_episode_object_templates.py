#!/usr/bin/env python3
"""
Repair missing object template handles in an exported episode JSON/JSON.GZ.

This fixes cases like:
- alarm_clock.object_config.json (missing)
- toothbrush.object_config.json (missing)
- water_bottle.object_config.json (missing)

It updates:
1) rigid_objs template paths
2) name_to_receptacle object keys (with collision-safe instance suffixing)
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple


def read_json_or_gz(path: Path) -> Dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_or_gz(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def split_handle(handle: str) -> Tuple[str, int]:
    if "_:" not in handle:
        return handle, 0
    base, suffix = handle.split("_:", 1)
    try:
        return base, int(suffix)
    except ValueError:
        return base, 0


def next_instance(existing_keys, base: str) -> int:
    nums = []
    prefix = f"{base}_:"
    for k in existing_keys:
        if k.startswith(prefix):
            _, n = split_handle(k)
            nums.append(n)
    return (max(nums) + 1) if nums else 0


def repair_templates(payload: Dict) -> Dict:
    if "episodes" in payload:
        ep = payload["episodes"][0]
        wrapped = True
    else:
        ep = payload
        wrapped = False

    # Deterministic mapping to known-valid template IDs in this dataset
    remap = {
        "alarm_clock": "Alarm_Clock_4",
        "toothbrush": "Sonicare_2_Series_Toothbrush_Plaque_Control",
        "water_bottle": "ce33c1228cfca3da78e22645019258d7a92af3a9",
    }

    # 1) Repair rigid_objs
    rigid_repaired = 0
    for rigid_obj in ep.get("rigid_objs", []):
        template = rigid_obj[0]
        base = template.replace(".object_config.json", "")
        if base in remap:
            rigid_obj[0] = f"{remap[base]}.object_config.json"
            rigid_repaired += 1

    # 2) Repair name_to_receptacle keys collision-safely
    old_map = ep.get("name_to_receptacle", {})
    if not isinstance(old_map, dict):
        old_map = {}

    new_map = {}
    key_repaired = 0

    for old_key, rec_value in old_map.items():
        base, idx = split_handle(old_key)
        if base not in remap:
            new_map[old_key] = rec_value
            continue

        new_base = remap[base]
        candidate = f"{new_base}_:{idx:04d}"

        # Avoid collisions with keys already present/created
        if candidate in new_map:
            new_idx = next_instance(list(new_map.keys()) + list(old_map.keys()), new_base)
            candidate = f"{new_base}_:{new_idx:04d}"

        new_map[candidate] = rec_value
        key_repaired += 1

    ep["name_to_receptacle"] = new_map

    summary = {
        "rigid_templates_repaired": rigid_repaired,
        "name_to_receptacle_keys_repaired": key_repaired,
        "total_mappings": len(new_map),
    }

    if wrapped:
        payload["episodes"][0] = ep
    else:
        payload = ep

    payload.setdefault("_repair_summary", {})
    payload["_repair_summary"]["template_repair"] = summary
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    in_path = Path(args.episode_file)
    payload = read_json_or_gz(in_path)
    repaired = repair_templates(payload)

    if args.output_file:
        out_path = Path(args.output_file)
    else:
        if in_path.name.endswith(".json.gz"):
            out_path = in_path.with_name(in_path.name[:-8] + "_templates_repaired.json.gz")
        else:
            out_path = in_path.with_name(in_path.stem + "_templates_repaired" + in_path.suffix)

    write_json_or_gz(out_path, repaired)
    print(f"Saved repaired file: {out_path}")
    print(json.dumps(repaired.get("_repair_summary", {}).get("template_repair", {}), indent=2))


if __name__ == "__main__":
    main()
