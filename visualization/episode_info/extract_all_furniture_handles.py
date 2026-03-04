#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to extract furniture names and handles from all episodes in a dataset.
Creates a mapping file that can be used to look up furniture handles without
needing to run the simulator.

Usage:
    python extract_all_furniture_handles.py dataset_path=<path_to_dataset> output_file=<output_path>
    
Example:
    python extract_all_furniture_handles.py dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz output_file=visualization/data/furniture_handles_val_mini.json
    python extract_all_furniture_handles.py dataset_path=data/datasets/partnr_episodes/v0_0/val.json.gz output_file=visualization/data/furniture_handles_val.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import omegaconf
import hydra
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import setup_config, fix_config


def extract_furniture_handles(world_graph) -> Dict[str, str]:
    """
    Extract furniture names and their corresponding sim handles.
    
    :param world_graph: The WorldGraph instance
    :return: Dictionary mapping furniture names to handles
    """
    furniture_handles = {}
    for entity in world_graph.get_all_furnitures():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        furniture_handles[entity.name] = sim_handle
    
    return furniture_handles


@hydra.main(
    config_path="../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
    version_base=None,
)
def main(config: omegaconf.DictConfig):
    """
    Main function to extract furniture handles from all episodes.
    
    :param config: Hydra config with CLI overrides
    """
    fix_config(config)
    
    # Setup seed
    seed = 47668090
    
    # Setup metadata path
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict
        
        # Set defaults if not provided
        if not hasattr(config, "dataset_path"):
            config.dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
        if not hasattr(config, "output_file"):
            config.output_file = "visualization/data/furniture_handles.json"
        if not hasattr(config, "max_episodes"):
            config.max_episodes = None  # Process all episodes
            
        # Override dataset path
        config.habitat.dataset.data_path = config.dataset_path
    
    config = setup_config(config, seed)
    remove_visual_sensors(config)
    
    # Register components
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    
    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    print(f"Total episodes in dataset: {len(dataset.episodes)}")
    
    # Storage for all furniture handles across episodes
    all_furniture_data = {}
    
    # Create environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset)
    
    # Determine which episodes to process
    episode_ids = [str(ep.episode_id) for ep in dataset.episodes]
    if config.max_episodes:
        episode_ids = episode_ids[:config.max_episodes]
    
    print(f"Processing {len(episode_ids)} episodes...")
    print("="*80)
    
    # Process each episode
    for idx, episode_id in enumerate(episode_ids):
        try:
            print(f"\n[{idx+1}/{len(episode_ids)}] Processing episode {episode_id}...")
            
            # Load episode
            env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(episode_id)
            env_interface.reset_environment()
            
            # Get episode info
            sim = env_interface.sim
            scene_id = sim.ep_info.scene_id
            
            # Extract furniture handles from world graph
            world_graph = env_interface.perception.gt_graph
            furniture_handles = extract_furniture_handles(world_graph)
            
            # Store episode data
            all_furniture_data[episode_id] = {
                "scene_id": scene_id,
                "furniture_handles": furniture_handles
            }
            
            print(f"  ✓ Extracted {len(furniture_handles)} furniture items")
            print(f"  Scene: {scene_id}")
            
        except Exception as e:
            print(f"  ✗ Error processing episode {episode_id}: {e}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Save to output file
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_furniture_data, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Furniture handles saved to: {output_path}")
    print(f"  Total episodes processed: {len(all_furniture_data)}")
    
    # Calculate total unique furniture across all episodes
    total_furniture = sum(len(ep_data["furniture_handles"]) for ep_data in all_furniture_data.values())
    print(f"  Total furniture items: {total_furniture}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Furniture Handle Extractor - Extract handles from all episodes")
    print("="*80 + "\n")
    
    main()
    
    print("\n" + "="*80)
    print("Extraction Complete")
    print("="*80 + "\n")
