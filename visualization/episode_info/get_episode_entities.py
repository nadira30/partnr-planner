#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to extract and display all entities, furniture, objects, and receptacles
from a given episode in the partnr-planner dataset.

Usage:
    python get_episode_entities.py episode_id=<episode_id> dataset_path=<path_to_dataset>
    
Example:
    python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz
    python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz print_handles=true
    python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz output_file=outputs/episode_100.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import omegaconf
import hydra
from hydra.core.config_store import ConfigStore
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import setup_config, fix_config
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)


def extract_entities_data(world_graph) -> Dict[str, List[str]]:
    """
    Extract all entity information from the world graph.
    
    :param world_graph: The WorldGraph instance
    :return: Dictionary containing lists of rooms, furniture, objects, and receptacles
    """
    return {
        "rooms": sorted([node.name for node in world_graph.get_all_rooms()]),
        "furniture": sorted([node.name for node in world_graph.get_all_furnitures()]),
        "objects": sorted([node.name for node in world_graph.get_all_objects()]),
        "receptacles": sorted([node.name for node in world_graph.get_all_receptacles()]),
    }


def extract_entity_handles(world_graph) -> Dict[str, Dict[str, str]]:
    """
    Extract entity names and their corresponding sim handles.
    
    :param world_graph: The WorldGraph instance
    :return: Dictionary containing furniture and object name-to-handle mappings
    """
    furniture_handles = {}
    for entity in world_graph.get_all_furnitures():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        furniture_handles[entity.name] = sim_handle
    
    object_handles = {}
    for entity in world_graph.get_all_objects():
        sim_handle = world_graph.get_node_from_name(entity.name).sim_handle
        object_handles[entity.name] = sim_handle
    
    return {
        "furniture_handles": furniture_handles,
        "object_handles": object_handles,
    }


def extract_furniture_locations(world_graph) -> Dict[str, str]:
    """
    Extract furniture names and their room locations.
    
    :param world_graph: The WorldGraph instance
    :return: Dictionary mapping furniture names to room names
    """
    furniture_to_room_map = world_graph.get_furniture_to_room_map()
    return {
        furniture.name: room.name
        for furniture, room in furniture_to_room_map.items()
        if not furniture.name.startswith('unknown')
    }


def extract_object_locations(world_graph) -> Dict[str, Dict[str, str]]:
    """
    Extract object locations including their furniture and room.
    
    :param world_graph: The WorldGraph instance
    :return: Dictionary mapping object names to their furniture and room locations
    """
    object_locations = {}
    
    for obj in world_graph.get_all_objects():
        location_info = {}
        
        # Find furniture the object is on/in
        furniture = world_graph.find_furniture_for_object(obj)
        if furniture:
            location_info['furniture'] = furniture.name
            
            # Find room the furniture is in
            try:
                room = world_graph.get_room_for_entity(furniture)
                location_info['room'] = room.name
            except ValueError:
                location_info['room'] = 'unknown'
        else:
            location_info['furniture'] = 'unknown'
            location_info['room'] = 'unknown'
        
        object_locations[obj.name] = location_info
    
    return object_locations


def print_furniture_locations(world_graph) -> None:
    """
    Print furniture names grouped by their room locations.
    
    :param world_graph: The WorldGraph instance
    """
    from habitat_llm.utils.core import cprint
    
    print("\n")
    cprint("Furniture Locations by Room:", "green")
    
    # Group furniture by room
    furniture_by_room = world_graph.group_furniture_by_room()
    
    # Sort rooms for consistent output
    for room_name in sorted(furniture_by_room.keys()):
        furniture_list = furniture_by_room[room_name]
        # Filter out unknown furniture
        furniture_names = sorted([f.name for f in furniture_list if not f.name.startswith('unknown')])
        if furniture_names:  # Only show room if it has non-unknown furniture
            cprint(f" {room_name}:", "green")
            cprint(f"  {furniture_names}", "yellow")
    
    print("\n")


def print_object_locations(world_graph) -> None:
    """
    Print object names with their furniture and room locations.
    
    :param world_graph: The WorldGraph instance
    """
    from habitat_llm.utils.core import cprint
    
    print("\n")
    cprint("Object Locations:", "green")
    
    object_locations = extract_object_locations(world_graph)
    
    # Sort objects for consistent output
    for obj_name in sorted(object_locations.keys()):
        location = object_locations[obj_name]
        location_str = f"{location['room']} -> {location['furniture']}"
        cprint(f" {obj_name}: {location_str}", "yellow")
    
    print("\n")


@hydra.main(
    config_path="../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
    version_base=None,
)
def main(config: omegaconf.DictConfig):
    """
    Main function to extract entities from an episode.
    
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
        if not hasattr(config, "episode_id"):
            raise ValueError("episode_id is required. Use: episode_id=<id>")
        if not hasattr(config, "dataset_path"):
            config.dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
        if not hasattr(config, "print_handles"):
            config.print_handles = False
        if not hasattr(config, "output_file"):
            config.output_file = None
            
        # Override dataset path
        config.habitat.dataset.data_path = config.dataset_path
    
    config = setup_config(config, seed)
    remove_visual_sensors(config)
    
    # Register components
    register_sensors(config)
    register_actions(config)
    register_measures(config)
    
    # Create dataset and environment
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")
    
    env_interface = EnvironmentInterface(config, dataset=dataset)
    
    # Load specific episode
    episode_id = str(config.episode_id)
    print(f"Loading episode_id = {episode_id}")
    try:
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(episode_id)
    except Exception as e:
        print(f"Error: Could not find episode with id '{args.episode_id}'")
        print(f"Exception: {e}")
    except Exception as e:
        print(f"Error: Could not find episode with id '{args.episode_id}'")
        print(f"Exception: {e}")
        sys.exit(1)
    
    env_interface.reset_environment()
    
    # Get episode info
    sim = env_interface.sim
    print(f"\n{'='*80}")
    print(f"Episode ID: {sim.ep_info.episode_id}")
    print(f"Scene ID: {sim.ep_info.scene_id}")
    print(f"Info: {sim.ep_info.info}")
    print(f"{'='*80}\n")
    
    # Print entities using existing utility functions
    print_all_entities(env_interface.perception.gt_graph)
    
    # Print furniture locations by room
    print_furniture_locations(env_interface.perception.gt_graph)
    
    # Print object locations
    print_object_locations(env_interface.perception.gt_graph)
    
    if config.print_handles:
        print_furniture_entity_handles(env_interface.perception.gt_graph)
        print_object_entity_handles(env_interface.perception.gt_graph)
    
    # Extract data for JSON output if requested
    if config.output_file:
        entity_data = extract_entities_data(env_interface.perception.gt_graph)
        handle_data = extract_entity_handles(env_interface.perception.gt_graph)
        furniture_locations = extract_furniture_locations(env_interface.perception.gt_graph)
        object_locations = extract_object_locations(env_interface.perception.gt_graph)
        
        output_data = {
            "episode_id": sim.ep_info.episode_id,
            "scene_id": sim.ep_info.scene_id,
            "episode_info": sim.ep_info.info,
            "entities": entity_data,
            "handles": handle_data,
            "furniture_locations": furniture_locations,
            "object_locations": object_locations,
        }
        
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Entity data saved to: {output_path}")
    
    # Print summary statistics
    entity_data = extract_entities_data(env_interface.perception.gt_graph)
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"  Total Rooms: {len(entity_data['rooms'])}")
    print(f"  Total Furniture: {len(entity_data['furniture'])}")
    print(f"  Total Objects: {len(entity_data['objects'])}")
    print(f"  Total Receptacles: {len(entity_data['receptacles'])}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Episode Entity Extractor - Get all entities from an episode")
    print("="*80 + "\n")
    
    main()
    
    print("\n" + "="*80)
    print("Extraction Complete")
    print("="*80 + "\n")
