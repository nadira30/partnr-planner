#!/usr/bin/env python3
# isort: skip_file

"""
Discover simulator receptacles for a specific episode.

Usage:
    python visualization/episode_info/get_episode_receptacles.py \
        +episode_id=100 \
        +dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
        +output_file=visualization/temp_episode_100_receptacles.json \
        hydra.run.dir=.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
import omegaconf
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.sim import find_receptacles


def _base_handle(handle: str) -> str:
    return handle.split("_:")[0] if "_:" in handle else handle


def extract_receptacles(sim) -> Dict:
    receptacles = find_receptacles(sim, filter_receptacles=True)

    all_unique: List[str] = []
    by_parent_handle: Dict[str, List[str]] = {}
    by_parent_base: Dict[str, List[str]] = {}

    for rec in receptacles:
        unique_name = rec.unique_name
        parent_handle = rec.parent_object_handle
        parent_base = _base_handle(parent_handle)

        all_unique.append(unique_name)
        by_parent_handle.setdefault(parent_handle, []).append(unique_name)
        by_parent_base.setdefault(parent_base, []).append(unique_name)

    all_unique = sorted(set(all_unique))
    by_parent_handle = {
        k: sorted(set(v)) for k, v in sorted(by_parent_handle.items())
    }
    by_parent_base = {k: sorted(set(v)) for k, v in sorted(by_parent_base.items())}

    return {
        "count": len(all_unique),
        "all": all_unique,
        "by_parent_handle": by_parent_handle,
        "by_parent_base": by_parent_base,
    }


@hydra.main(
    config_path="../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
    version_base=None,
)
def main(config: omegaconf.DictConfig):
    fix_config(config)
    seed = 47668090

    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict

        if not hasattr(config, "episode_id"):
            raise ValueError("episode_id is required. Use +episode_id=<id>")
        if not hasattr(config, "dataset_path"):
            config.dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
        if not hasattr(config, "output_file"):
            raise ValueError("output_file is required. Use +output_file=<path>")

        config.habitat.dataset.data_path = config.dataset_path

    config = setup_config(config, seed)
    remove_visual_sensors(config)

    register_sensors(config)
    register_actions(config)
    register_measures(config)

    dataset = CollaborationDatasetV0(config.habitat.dataset)
    env_interface = EnvironmentInterface(config, dataset=dataset)

    episode_id = str(config.episode_id)
    env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(episode_id)
    env_interface.reset_environment()

    sim = env_interface.sim
    receptacle_data = extract_receptacles(sim)

    output_data = {
        "episode_id": episode_id,
        "scene_id": sim.ep_info.scene_id,
        "receptacles": receptacle_data,
    }

    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved simulator receptacles for episode {episode_id} to {output_path}")


if __name__ == "__main__":
    main()
