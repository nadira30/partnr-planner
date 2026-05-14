#!/usr/bin/env python3
"""
Enhance furniture_handles mapping file to include room associations.

Room labels are derived from the actual scene geometry by loading each episode
into Habitat and reading the WorldGraph furniture-to-room edges.
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import hydra
import omegaconf

# Add the project root to path so local package changes are used.
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import fix_config, setup_config


def extract_furniture_rooms_from_world_graph(world_graph, furniture_names: Set[str]) -> Dict[str, str]:
    """
    Extract furniture to room mapping directly from the WorldGraph.

    Args:
        world_graph: The active WorldGraph.
        furniture_names: Furniture names to keep from the mapping.

    Returns:
        Dictionary mapping furniture_name -> room_name.
    """
    furniture_rooms: Dict[str, str] = {}
    furniture_to_room_map = world_graph.get_furniture_to_room_map()

    for furniture_node, room_node in furniture_to_room_map.items():
        furniture_name = getattr(furniture_node, "name", "")
        room_name = getattr(room_node, "name", "")

        if not furniture_name or not room_name:
            continue
        if furniture_name.startswith("floor"):
            continue
        if furniture_names and furniture_name not in furniture_names:
            continue

        furniture_rooms[furniture_name] = room_name

    return furniture_rooms


def enhance_furniture_handles_mapping(mapping_file: str, env_interface: EnvironmentInterface) -> dict:
    """
    Enhance furniture_handles mapping to include room information.

    Changes structure from:
        {
            "episode_id": {
                "scene_id": "...",
                "furniture_handles": {...}
            }
        }

    To:
        {
            "episode_id": {
                "scene_id": "...",
                "furniture_handles": {...},
                "furniture_rooms": {...}
            }
        }
    """
    with open(mapping_file, "r") as f:
        mapping_data = json.load(f)

    print(f"✓ Loaded mapping for {len(mapping_data)} episodes")

    enhanced_count = 0
    episode_ids = list(mapping_data.keys())

    for episode_id in episode_ids:
        episode_data = mapping_data[episode_id]
        try:
            print(f"\n📍 Processing episode {episode_id}...")

            env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(episode_id)
            env_interface.reset_environment()

            furniture_names = set(episode_data.get("furniture_handles", {}).keys())
            furniture_rooms = extract_furniture_rooms_from_world_graph(
                env_interface.perception.gt_graph,
                furniture_names,
            )

            if furniture_rooms:
                episode_data["furniture_rooms"] = furniture_rooms
                enhanced_count += 1

                sample_items = sorted(furniture_rooms.items())[:5]
                for fname, rname in sample_items:
                    print(f"    {fname} → {rname}")
                if len(furniture_rooms) > 5:
                    print(f"    ... and {len(furniture_rooms) - 5} more")
                print(f"    Total: {len(furniture_rooms)} furniture pieces mapped to rooms")
            else:
                print(f"  ⚠ Could not extract room mapping for episode {episode_id}")

        except Exception as e:
            print(f"  ⚠ Error processing episode {episode_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"✓ Enhanced {enhanced_count}/{len(mapping_data)} episodes with room associations")
    print(f"{'=' * 60}")

    return mapping_data


@hydra.main(
    config_path="../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
    version_base=None,
)
def main(config: omegaconf.DictConfig):
    mapping_file = Path(__file__).parent / "data" / "furniture_handles_val_mini.json"
    dataset_path = Path(__file__).parent.parent / "data" / "datasets" / "partnr_episodes" / "v0_0" / "val_mini.json.gz"

    if not mapping_file.exists():
        print(f"Error: Mapping file not found: {mapping_file}")
        sys.exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    print(f"📂 Mapping file: {mapping_file}")
    print(f"📂 Dataset file: {dataset_path}\n")

    fix_config(config)

    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
        config.habitat.dataset.data_path = str(dataset_path)

        if not hasattr(config, "dataset_path"):
            config.dataset_path = str(dataset_path)

    config = setup_config(config, 47668090)
    remove_visual_sensors(config)
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"✓ Loaded dataset with {len(dataset.episodes)} episodes")

    env_interface = EnvironmentInterface(config, dataset=dataset)

    enhanced_mapping = enhance_furniture_handles_mapping(str(mapping_file), env_interface)

    backup_file = mapping_file.with_suffix(".json.backup")
    print(f"\n💾 Saving backup to: {backup_file}")
    with open(backup_file, "w") as f:
        json.dump(enhanced_mapping, f, indent=2)

    print(f"💾 Saving enhanced mapping to: {mapping_file}")
    with open(mapping_file, "w") as f:
        json.dump(enhanced_mapping, f, indent=2)

    print("\n✓ Successfully enhanced furniture_handles mapping with room associations!")


if __name__ == "__main__":
    main()
