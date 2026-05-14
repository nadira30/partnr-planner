#!/usr/bin/env python3
"""
Remap skill_runner command object names to actual graph node names for a specific episode file.

Example:
  python visualization/episode_info/remap_skill_runner_commands.py \
    --commands-file skill_runner_commands.txt \
    --dataset-path visualization/data/episode_100_modified_2026-03-30_13-22-10.json.gz \
    --episode-id 100
"""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


ENTITY_PATTERN = re.compile(r"^[A-Za-z0-9_]+_\d+$")


LEGACY_CATEGORY_ALIASES = {
    "alarm_clock": ("clock", "first"),
    "toothbrush": ("box", "first"),
    "water_bottle": ("bottle", "last"),
}


def load_episode(dataset_path: Path, episode_id: str) -> Dict:
    if dataset_path.suffix == ".gz":
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if isinstance(payload, dict) and "episodes" in payload:
        for ep in payload.get("episodes", []):
            if str(ep.get("episode_id")) == str(episode_id):
                return ep
        raise ValueError(f"Episode {episode_id} not found in {dataset_path}")

    # single-episode payload
    ep = payload
    if str(ep.get("episode_id")) != str(episode_id):
        raise ValueError(f"Episode id mismatch: expected {episode_id}, found {ep.get('episode_id')}")
    return ep


def load_category_map(project_root: Path) -> Dict[str, str]:
    rows = []
    csv_paths = [
        project_root / "data" / "hssd-hab" / "metadata" / "object_categories_filtered.csv",
        project_root / "visualization" / "objects" / "object_categories_one_per_class.csv",
    ]
    for path in csv_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = (row.get("id") or "").strip()
                category = (row.get("category") or row.get("clean_category") or "").strip()
                if obj_id and category:
                    rows.append((obj_id, category))

    category_map = {}
    for obj_id, category in rows:
        category_map.setdefault(obj_id, category)
    return category_map


def normalize_template_id(template: str) -> str:
    obj_id = template.replace(".object_config.json", "")
    if "_:" in obj_id:
        obj_id = obj_id.split("_:", 1)[0]
    if "/" in obj_id:
        obj_id = obj_id.split("/")[-1]
    return obj_id


def build_graph_object_names(episode: Dict, category_map: Dict[str, str]) -> List[str]:
    names = []
    for idx, rigid_obj in enumerate(episode.get("rigid_objs", [])):
        if not rigid_obj:
            continue
        template = rigid_obj[0]
        obj_id = normalize_template_id(template)
        category = category_map.get(obj_id, obj_id.lower())
        names.append(f"{category}_{idx}")
    return names


def build_alias_map(graph_object_names: List[str]) -> Dict[str, str]:
    alias = {}

    # exact aliases
    for name in graph_object_names:
        alias[name] = name

    by_category: Dict[str, List[Tuple[int, str]]] = {}
    for name in graph_object_names:
        category, idx_s = name.rsplit("_", 1)
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        by_category.setdefault(category, []).append((idx, name))

    for category, items in by_category.items():
        items = sorted(items, key=lambda x: x[0])
        # one-based aliases: category_1, category_2, ...
        for one_based, (_, actual_name) in enumerate(items, start=1):
            alias[f"{category}_{one_based}"] = actual_name
        # zero-based aliases for convenience
        for zero_based, (_, actual_name) in enumerate(items):
            alias[f"{category}_{zero_based}"] = actual_name

    # Compatibility aliases for object classes that can be category-remapped by
    # template repairs (e.g. alarm_clock->clock, toothbrush->box, water_bottle->bottle).
    for legacy_category, (target_category, strategy) in LEGACY_CATEGORY_ALIASES.items():
        target_items = sorted(by_category.get(target_category, []), key=lambda x: x[0])
        if not target_items:
            continue
        ordered_items = list(reversed(target_items)) if strategy == "last" else target_items

        for one_based, (_, actual_name) in enumerate(ordered_items, start=1):
            alias.setdefault(f"{legacy_category}_{one_based}", actual_name)
        for zero_based, (_, actual_name) in enumerate(ordered_items):
            alias.setdefault(f"{legacy_category}_{zero_based}", actual_name)

    return alias


def remap_entity_token(token: str, alias_map: Dict[str, str]) -> str:
    if token in ("None", "none", "NULL", "null", ""):
        return token
    return alias_map.get(token, token)


def remap_line(line: str, alias_map: Dict[str, str]) -> str:
    stripped = line.strip()
    if not stripped:
        return line

    parts = stripped.split()
    if len(parts) < 3:
        return line

    command = parts[0]
    if command == "Wait":
        if len(parts) >= 3:
            wait_time = parts[2]
            try:
                scaled_wait = float(wait_time) / 5.0
                if scaled_wait.is_integer():
                    scaled_wait_text = str(int(scaled_wait))
                else:
                    scaled_wait_text = str(scaled_wait)
                new_line = line.replace(wait_time, scaled_wait_text, 1)
                return new_line
            except ValueError:
                return line

    if command in {"Pick", "Navigate", "Open", "Close"}:
        entity = parts[2]
        new_entity = remap_entity_token(entity, alias_map)
        if new_entity != entity:
            return line.replace(entity, new_entity, 1)
        return line

    if command == "Place":
        # Expected format: Place <agent> <ent0,rel0,ent1,rel1,ent2>
        payload = " ".join(parts[2:])
        chunks = payload.split(",")
        if len(chunks) >= 5:
            chunks[0] = remap_entity_token(chunks[0], alias_map)
            chunks[2] = remap_entity_token(chunks[2], alias_map)
            chunks[4] = remap_entity_token(chunks[4], alias_map)
            new_payload = ",".join(chunks)
            prefix = " ".join(parts[:2])
            return f"{prefix} {new_payload}\n" if line.endswith("\n") else f"{prefix} {new_payload}"

    return line


def collect_unresolved(lines: List[str], alias_map: Dict[str, str]) -> List[str]:
    unresolved = set()
    for line in lines:
        for token in re.findall(r"[A-Za-z0-9_]+_\d+", line):
            if token not in alias_map and ENTITY_PATTERN.match(token):
                unresolved.add(token)
    return sorted(unresolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    commands_file = Path(args.commands_file)
    dataset_path = Path(args.dataset_path)

    episode = load_episode(dataset_path, args.episode_id)
    category_map = load_category_map(project_root)
    graph_names = build_graph_object_names(episode, category_map)
    alias_map = build_alias_map(graph_names)

    lines = commands_file.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = [remap_line(line, alias_map) for line in lines]

    output_file = Path(args.output_file) if args.output_file else commands_file.with_name(commands_file.stem + "_mapped.txt")
    output_file.write_text("".join(new_lines), encoding="utf-8")

    unresolved = collect_unresolved(new_lines, alias_map)

    print(f"Wrote remapped commands: {output_file}")
    print(f"Graph object count: {len(graph_names)}")
    print(f"Alias count: {len(alias_map)}")
    print(f"Unresolved object-like tokens: {len(unresolved)}")
    if unresolved:
        for token in unresolved[:30]:
            print(f"  - {token}")


if __name__ == "__main__":
    main()
