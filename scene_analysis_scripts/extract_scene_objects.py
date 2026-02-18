#!/usr/bin/env python3
"""
Extract comprehensive scene object information from episodes.
Creates a JSON file with all objects, their categories, locations, and properties.
"""

import gzip
import json
import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_episode_data(dataset_path: str, episode_id: Optional[str] = None):
    """Load episode data from dataset."""
    with gzip.open(dataset_path, 'rt') as f:
        data = json.load(f)
    
    if episode_id:
        for ep in data['episodes']:
            if ep['episode_id'] == episode_id:
                return ep
        raise ValueError(f"Episode {episode_id} not found")
    
    return data['episodes'][0]


def run_skill_runner_and_extract(episode_id: str, dataset_path: str) -> Dict[str, Any]:
    """
    Run skill_runner to get object and furniture information from the environment.
    """
    print(f"Loading environment for episode {episode_id}...")
    
    cmd = [
        'python', '-m', 'habitat_llm.examples.skill_runner',
        'hydra.run.dir=.',
        f'habitat.dataset.data_path={dataset_path}',
        f'+skill_runner_episode_id={episode_id}',
        '+skill_runner_show_videos=False',
        'evaluation.save_video=False'
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Send entities command to get object information, then exit
    output, _ = process.communicate(input='entities\nexit\n', timeout=120)
    
    return parse_skill_runner_output(output)


def parse_skill_runner_output(output: str) -> Dict[str, Any]:
    """
    Parse the output from skill_runner to extract object and furniture information.
    """
    lines = output.split('\n')
    
    info = {
        'furniture': {},
        'objects': {},
        'rooms': {}
    }
    
    # Parse furniture names to handles
    in_furniture = False
    for line in lines:
        if 'Furniture Names to Handles:' in line:
            in_furniture = True
            continue
        if in_furniture:
            if 'Object Names to Handles:' in line:
                break
            match = re.match(r'\s+(\w+_[\w\d]+)\s*:\s*(.+)', line)
            if match:
                name = match.group(1).strip()
                handle = match.group(2).strip()
                info['furniture'][handle] = {
                    'name': name,
                    'category': name.rsplit('_', 1)[0] if '_' in name else name
                }
    
    # Parse object names to handles
    in_objects = False
    for line in lines:
        if 'Object Names to Handles:' in line:
            in_objects = True
            continue
        if in_objects:
            if line.strip() == '' or 'Available skills' in line or 'Enter a skill name' in line:
                break
            match = re.match(r'\s+(\w+_\d+)\s*:\s*(.+)', line)
            if match:
                name = match.group(1).strip()
                handle = match.group(2).strip()
                info['objects'][handle] = {
                    'name': name,
                    'category': name.rsplit('_', 1)[0] if '_' in name else name
                }
    
    # Parse world graph for room information
    in_furniture_section = False
    current_room = None
    
    for line in lines:
        if line.strip().startswith('Furniture:'):
            in_furniture_section = True
            continue
        
        if in_furniture_section:
            if line.strip().startswith('Objects:'):
                in_furniture_section = False
                continue
            
            # Room line format: "room_name: furniture1, furniture2, ..."
            if ':' in line and not line.strip().startswith('floor_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    room_name = parts[0].strip()
                    furniture_list = [f.strip() for f in parts[1].split(',')]
                    
                    if room_name not in info['rooms']:
                        info['rooms'][room_name] = {'furniture': []}
                    
                    info['rooms'][room_name]['furniture'] = furniture_list
    
    # Parse object-to-furniture mapping from "Objects:" section
    in_objects_section = False
    for line in lines:
        if line.strip().startswith('Objects:'):
            in_objects_section = True
            continue
        
        if in_objects_section:
            # Object line format: "object_name: furniture_name"
            if ':' in line and not line.strip() == '':
                parts = line.split(':', 1)
                if len(parts) == 2:
                    obj_name = parts[0].strip()
                    furniture_name = parts[1].strip()
                    
                    # Find the object handle for this object name
                    for handle, obj_info in info['objects'].items():
                        if obj_info['name'] == obj_name:
                            obj_info['on_furniture'] = furniture_name
                            break
    
    return info


def extract_scene_info(dataset_path: str, episode_id: str) -> Dict[str, Any]:
    """
    Extract complete scene information including all objects and their properties.
    """
    # Load episode data
    episode = load_episode_data(dataset_path, episode_id)
    scene_id = episode['scene_id']
    
    print(f"\n{'='*80}")
    print(f"Extracting Scene Information")
    print(f"{'='*80}")
    print(f"Episode: {episode_id}")
    print(f"Scene: {scene_id}")
    print(f"Instruction: {episode['instruction']}")
    print(f"{'='*80}\n")
    
    # Get object information from skill_runner
    env_info = run_skill_runner_and_extract(episode_id, dataset_path)
    
    # Build scene info structure
    scene_info = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "instruction": episode['instruction'],
        "objects": []
    }
    
    object_counter = 0
    
    # Process objects from rigid_objs in episode
    rigid_objs_map = {}
    for obj_data in episode.get('rigid_objs', []):
        obj_file = obj_data[0]
        obj_name_raw = obj_file.replace('.object_config.json', '')
        transform = obj_data[1]
        position = [transform[0][3], transform[1][3], transform[2][3]]
        rigid_objs_map[obj_name_raw] = position
    
    # Process each object
    for obj_handle, obj_info in env_info['objects'].items():
        obj_category = obj_info['category']
        obj_name = obj_info['name']
        
        # Find position from rigid_objs
        position = [0.0, 0.0, 0.0]
        for raw_name, pos in rigid_objs_map.items():
            if raw_name in obj_handle:
                position = pos
                break
        
        # Infer properties based on category
        properties = []
        if obj_category:
            properties.append("GRABBABLE")
            properties.append("MOVABLE")
            
            # Furniture items typically have surfaces
            furniture_items = ['table', 'counter', 'desk', 'shelves', 'cabinet', 
                              'chair', 'couch', 'bed', 'bench', 'stool']
            if any(furn in obj_category.lower() for furn in furniture_items):
                properties.append("SURFACES")
            
            # Sittable items
            sittable = ['chair', 'couch', 'bench', 'stool', 'bed']
            if any(item in obj_category.lower() for item in sittable):
                properties.append("SITTABLE")
            
            # Lieable items
            lieable = ['bed', 'couch']
            if any(item in obj_category.lower() for item in lieable):
                properties.append("LIEABLE")
        
        # Find room for this object
        room_name = "unknown"
        on_furniture = obj_info.get('on_furniture', 'floor')
        
        # Find which room contains this furniture
        for room, room_data in env_info['rooms'].items():
            if on_furniture in room_data.get('furniture', []):
                room_name = room
                break
        
        # Build object info dictionary
        obj_data = {
            "id": object_counter,
            "category": obj_category,
            "class_name": obj_category,
            "prefab_name": obj_name,
            "handle": obj_handle,
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "properties": properties,
            "states": [],
            "room": room_name,
            "on_furniture": on_furniture
        }
        
        scene_info["objects"].append(obj_data)
        object_counter += 1
    
    # Add furniture information
    scene_info["furniture"] = []
    for furn_handle, furn_info in env_info['furniture'].items():
        furniture_data = {
            "id": len(scene_info["objects"]) + len(scene_info["furniture"]),
            "category": "Furniture",
            "class_name": furn_info['category'],
            "prefab_name": furn_info['name'],
            "handle": furn_handle,
            "properties": ["SURFACES"],
            "states": []
        }
        scene_info["furniture"].append(furniture_data)
    
    # Add room information
    scene_info["rooms"] = []
    for room_name, room_data in env_info['rooms'].items():
        room_info = {
            "id": len(scene_info["objects"]) + len(scene_info["furniture"]) + len(scene_info["rooms"]),
            "category": "Rooms",
            "class_name": room_name.rsplit('_', 1)[0] if '_' in room_name else room_name,
            "prefab_name": room_name,
            "properties": [],
            "states": [],
            "furniture": room_data.get('furniture', [])
        }
        scene_info["rooms"].append(room_info)
    
    print(f"Extracted {len(scene_info['objects'])} objects, {len(scene_info['furniture'])} furniture items, and {len(scene_info['rooms'])} rooms")
    
    return scene_info


def save_scene_info(scene_info: Dict[str, Any], output_path: str):
    """Save scene information to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(scene_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Scene information saved to: {output_file}")
    print(f"{'='*80}\n")


def extract_all_episodes(dataset_path: str, output_dir: str = "scene_objects"):
    """Extract scene information for all unique scenes in the dataset."""
    
    # Load dataset to get all episodes
    with gzip.open(dataset_path, 'rt') as f:
        data = json.load(f)
    
    # Get unique scenes
    scenes = {}
    for episode in data['episodes']:
        scene_id = episode['scene_id']
        if scene_id not in scenes:
            scenes[scene_id] = episode['episode_id']
    
    print(f"\nFound {len(scenes)} unique scenes")
    print("Processing scenes...\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_scenes_info = {}
    
    for idx, (scene_id, episode_id) in enumerate(scenes.items(), 1):
        print(f"\n[{idx}/{len(scenes)}] Processing scene: {scene_id}")
        
        try:
            scene_info = extract_scene_info(dataset_path, episode_id)
            
            # Save individual scene file
            scene_file = output_path / f"scene_{scene_id}.json"
            save_scene_info(scene_info, str(scene_file))
            
            # Add to combined output in the requested format
            env_key = f"env_{idx-1}"
            all_scenes_info[env_key] = []
            
            # Combine rooms, furniture, and objects
            all_scenes_info[env_key].extend(scene_info.get("rooms", []))
            all_scenes_info[env_key].extend(scene_info.get("furniture", []))
            all_scenes_info[env_key].extend(scene_info.get("objects", []))
            
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined output
    combined_file = output_path / "all_scenes_objects.json"
    with open(combined_file, 'w') as f:
        json.dump(all_scenes_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All scenes information saved to: {combined_file}")
    print(f"Individual scene files saved to: {output_path}/")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract scene object information from episodes'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/datasets/partnr_episodes/v0_0/val_mini.json.gz',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--episode-id',
        type=str,
        default=None,
        help='Specific episode ID to process (if not provided, processes all unique scenes)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scene_objects',
        help='Output directory for scene information'
    )
    
    args = parser.parse_args()
    
    if args.episode_id:
        # Process single episode
        scene_info = extract_scene_info(args.dataset, args.episode_id)
        output_file = f"{args.output}/scene_{scene_info['scene_id']}.json"
        save_scene_info(scene_info, output_file)
    else:
        # Process all unique scenes
        extract_all_episodes(args.dataset, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
