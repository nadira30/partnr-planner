#!/usr/bin/env python3
"""
Extract scene object information directly from episode data.
Creates a JSON file with all objects, their categories, locations, and properties.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_dataset(dataset_path: str):
    """Load dataset from gzip file."""
    with gzip.open(dataset_path, 'rt') as f:
        return json.load(f)


def get_object_properties(category: str) -> List[str]:
    """Infer object properties based on category."""
    properties = []
    
    if category:
        # Most movable objects are grabbable
        properties.extend(["GRABBABLE", "MOVABLE"])
        
        # Furniture items typically have surfaces
        furniture_items = ['table', 'counter', 'desk', 'shelves', 'cabinet', 
                          'chair', 'couch', 'bed', 'bench', 'stool']
        if any(furn in category.lower() for furn in furniture_items):
            properties.append("SURFACES")
        
        # Sittable items
        sittable = ['chair', 'couch', 'bench', 'stool', 'bed']
        if any(item in category.lower() for item in sittable):
            properties.append("SITTABLE")
        
        # Lieable items
        lieable = ['bed', 'couch']
        if any(item in category.lower() for item in lieable):
            properties.append("LIEABLE")
    
    return properties


def extract_object_category_from_handle(handle: str) -> str:
    """Extract object category from handle or filename."""
    # Remove file extension
    name = handle.replace('.object_config.json', '')
    # Remove hash suffixes (long hex strings)
    name = name.split('_:')[0]
    # Try to extract meaningful name
    parts = name.split('_')
    # If it's a product ID or hash, try to infer from context
    if len(parts) > 0 and not all(c.isdigit() or c in 'abcdef' for c in ''.join(parts)):
        # Has some alphabetic parts, use as-is
        return name.lower().replace('_', ' ')
    return 'object'


def build_furniture_handle_to_name_map(episodes: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a mapping from furniture handles to semantic names by correlating
    initial_state furniture_names with actual object placements.
    """
    furniture_map = {}
    
    for episode in episodes:
        initial_state = episode.get('info', {}).get('initial_state', [])
        name_to_rec = episode.get('name_to_receptacle', {})
        rigid_objs = episode.get('rigid_objs', [])
        
        # Create obj file to handle mapping
        obj_file_to_handle = {}
        for obj_data in rigid_objs:
            obj_file = obj_data[0].replace('.object_config.json', '')
            for handle_key in name_to_rec.keys():
                if obj_file in handle_key:
                    obj_file_to_handle[obj_file] = handle_key
                    break
        
        # Track which objects should go on which furniture (by name)
        furniture_name_to_obj_count = defaultdict(int)
        furniture_handle_to_obj_count = defaultdict(int)
        
        for state_element in initial_state:
            if 'furniture_names' not in state_element or not state_element['furniture_names']:
                continue
            
            furniture_name = state_element['furniture_names'][0]
            num_objects = state_element.get('number', 1)
            if isinstance(num_objects, str):
                try:
                    num_objects = int(num_objects)
                except:
                    num_objects = 1
            
            furniture_name_to_obj_count[furniture_name] += num_objects
        
        # Count actual placements per handle
        for obj_handle, rec_info in name_to_rec.items():
            furn_handle = rec_info.split('|')[0]
            if furn_handle != 'floor':
                furniture_handle_to_obj_count[furn_handle] += 1
        
        # Match furniture names to handles based on object counts
        # Sort both by count to match them up
        name_counts = sorted(furniture_name_to_obj_count.items(), key=lambda x: x[1], reverse=True)
        handle_counts = sorted(furniture_handle_to_obj_count.items(), key=lambda x: x[1], reverse=True)
        
        # Map handles to names based on matching counts
        for (name, name_count), (handle, handle_count) in zip(name_counts, handle_counts):
            if handle not in furniture_map:
                furniture_map[handle] = name
    
    return furniture_map


def extract_scene_from_episode(episode: Dict[str, Any], env_id: int = 0, furniture_map: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Extract scene information from a single episode.
    Returns a list of all entities (rooms, furniture, objects) in the scene.
    """
    scene_id = episode['scene_id']
    episode_id = episode['episode_id']
    
    print(f"Processing episode {episode_id} (scene: {scene_id})")
    
    all_entities = []
    entity_id = 0
    
    # Track objects by category for numbering
    object_counts = defaultdict(int)
    furniture_items = set()
    rooms = set()
    
    # Extract initial_state to get semantic names for task objects
    initial_state = episode.get('info', {}).get('initial_state', [])
    task_object_classes = []
    for state_element in initial_state:
        if 'object_classes' in state_element and len(state_element['object_classes']) > 0:
            obj_class = state_element['object_classes'][0]
            num_objects = state_element.get('number', 1)
            if isinstance(num_objects, str):
                try:
                    num_objects = int(num_objects)
                except:
                    num_objects = 1
            for _ in range(num_objects):
                task_object_classes.append(obj_class)
        
        # Collect rooms and furniture
        for region in state_element.get('allowed_regions', []):
            rooms.add(region)
        for furn in state_element.get('furniture_names', []):
            furniture_items.add(furn)
    
    # Extract ALL objects from rigid_objs and their placement from name_to_receptacle
    name_to_receptacle = episode.get('name_to_receptacle', {})
    rigid_objs = episode.get('rigid_objs', [])
    
    # The first N objects in rigid_objs correspond to task objects from initial_state
    # Remaining objects are "clutter" or common_sense objects
    task_obj_idx = 0
    
    # Process all rigid objects
    for obj_idx, obj_data in enumerate(rigid_objs):
        obj_file = obj_data[0]
        obj_base_name = obj_file.replace('.object_config.json', '')
        
        # Try to find matching handle in name_to_receptacle
        obj_handle = None
        for handle_key in name_to_receptacle.keys():
            if obj_base_name in handle_key:
                obj_handle = handle_key
                break
        
        if not obj_handle:
            continue
        
        # Get receptacle info
        receptacle_info = name_to_receptacle.get(obj_handle, '')
        furniture_handle = receptacle_info.split('|')[0] if receptacle_info else 'floor'
        
        # Determine object category - use semantic name from initial_state if available
        if task_obj_idx < len(task_object_classes):
            obj_category = task_object_classes[task_obj_idx]
            task_obj_idx += 1
        else:
            # For clutter objects, try to extract from filename
            obj_category = extract_object_category_from_handle(obj_file)
            # Mark as clutter/common_sense object
            if obj_category == 'object':
                obj_category = 'clutter_object'
        
        # Generate object name with counter
        obj_name = f"{obj_category}_{object_counts[obj_category]}"
        object_counts[obj_category] += 1
        
        # Get position from transform matrix
        transform = obj_data[1]
        position = {
            "x": float(transform[0][3]),
            "y": float(transform[1][3]),
            "z": float(transform[2][3])
        }
        
        # Track furniture
        if furniture_handle != 'floor':
            furniture_items.add(furniture_handle)
        
        # Get semantic furniture name if available
        furniture_name = furniture_map.get(furniture_handle, furniture_handle) if furniture_map else furniture_handle
        
        entity = {
            "id": entity_id,
            "category": "Object",
            "class_name": obj_category,
            "prefab_name": obj_name,
            "handle": obj_handle,
            "properties": get_object_properties(obj_category),
            "states": [],
            "position": position,
            "on_furniture": furniture_name,
            "on_furniture_handle": furniture_handle
        }
        
        all_entities.append(entity)
        entity_id += 1
    
    # Add rooms as entities
    for room in sorted(rooms):
        room_entity = {
            "id": entity_id,
            "category": "Rooms",
            "class_name": room.replace('_0', '').replace('_1', '').replace('_', ' '),
            "prefab_name": room,
            "properties": [],
            "states": []
        }
        all_entities.append(room_entity)
        entity_id += 1
    
    # Add furniture as entities
    for furniture_handle in sorted(furniture_items):
        # Extract furniture type from name (e.g., "table_1" -> "table")
        furniture_name = furniture_map.get(furniture_handle, furniture_handle) if furniture_map else furniture_handle
        
        if '_' in furniture_name and furniture_name[0].islower():
            furn_type = furniture_name.rsplit('_', 1)[0]
        else:
            furn_type = furniture_name
        
        furn_entity = {
            "id": entity_id,
            "category": "Furniture",
            "class_name": furn_type,
            "prefab_name": furniture_name,
            "handle": furniture_handle,
            "properties": get_object_properties(furn_type),
            "states": []
        }
        all_entities.append(furn_entity)
        entity_id += 1
    
    return all_entities


def extract_all_scenes(dataset_path: str, output_dir: str = "scene_objects"):
    """Extract scene information for all episodes in each unique scene."""
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Group episodes by scene
    scenes_to_episodes = defaultdict(list)
    for episode in data['episodes']:
        scene_id = episode['scene_id']
        scenes_to_episodes[scene_id].append(episode)
    
    print(f"\n{'='*80}")
    print(f"Extracting Scene Information from All Episodes")
    print(f"{'='*80}")
    print(f"Found {len(scenes_to_episodes)} unique scenes")
    print(f"Total episodes: {len(data['episodes'])}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_scenes_info = {}
    
    for idx, (scene_id, episodes) in enumerate(sorted(scenes_to_episodes.items()), 1):
        print(f"[{idx}/{len(scenes_to_episodes)}] Scene: {scene_id}")
        print(f"  Episodes using this scene: {len(episodes)}")
        
        try:
            # Build furniture handle to name mapping for this scene
            print(f"    Building furniture mapping...")
            furniture_map = build_furniture_handle_to_name_map(episodes)
            print(f"    Mapped {len(furniture_map)} furniture items")
            
            # Track all unique entities across all episodes in this scene
            all_entities_map = {}  # Use dict to track unique entities by handle
            entity_id_counter = 0
            
            # Process each episode in this scene
            for ep_idx, episode in enumerate(episodes):
                if ep_idx % 10 == 0 and ep_idx > 0:
                    print(f"    Processing episode {ep_idx}/{len(episodes)}...")
                
                episode_entities = extract_scene_from_episode(episode, idx-1, furniture_map)
                
                # Add unique entities
                for entity in episode_entities:
                    # Use handle as unique key for objects
                    if entity['category'] == 'Object' and 'handle' in entity:
                        key = entity['handle']
                        if key not in all_entities_map:
                            entity['id'] = entity_id_counter
                            all_entities_map[key] = entity
                            entity_id_counter += 1
                    # For rooms and furniture, use prefab_name as key
                    elif entity['category'] in ['Rooms', 'Furniture']:
                        key = f"{entity['category']}_{entity['prefab_name']}"
                        if key not in all_entities_map:
                            entity['id'] = entity_id_counter
                            all_entities_map[key] = entity
                            entity_id_counter += 1
            
            # Convert to list
            all_entities = list(all_entities_map.values())
            
            # Save individual scene file with all episodes' objects
            scene_data = {
                "scene_id": scene_id,
                "num_episodes": len(episodes),
                "episode_ids": [ep['episode_id'] for ep in episodes],
                "entities": all_entities
            }
            
            scene_file = output_path / f"scene_{scene_id}.json"
            with open(scene_file, 'w') as f:
                json.dump(scene_data, f, indent=2)
            
            # Add to combined output
            env_key = f"env_{idx-1}"
            all_scenes_info[env_key] = all_entities
            
            # Count by category
            obj_count = len([e for e in all_entities if e['category'] == 'Object'])
            furn_count = len([e for e in all_entities if e['category'] == 'Furniture'])
            room_count = len([e for e in all_entities if e['category'] == 'Rooms'])
            
            print(f"  ✓ Saved {len(all_entities)} total entities")
            print(f"    └─ {obj_count} objects, {furn_count} furniture, {room_count} rooms")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined output
    combined_file = output_path / "all_scenes_objects.json"
    with open(combined_file, 'w') as f:
        json.dump(all_scenes_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Extraction Complete")
    print(f"{'='*80}")
    print(f"Combined file: {combined_file}")
    print(f"Individual files: {output_path}/scene_*.json")
    print(f"{'='*80}\n")


def extract_single_episode(dataset_path: str, episode_id: str, output_dir: str = "scene_objects"):
    """Extract scene information for a single episode."""
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Find episode
    episode = None
    for ep in data['episodes']:
        if ep['episode_id'] == episode_id:
            episode = ep
            break
    
    if episode is None:
        raise ValueError(f"Episode {episode_id} not found in dataset")
    
    print(f"\n{'='*80}")
    print(f"Extracting Scene Information")
    print(f"{'='*80}")
    print(f"Episode: {episode_id}")
    print(f"Scene: {episode['scene_id']}")
    print(f"Instruction: {episode['instruction']}")
    print(f"{'='*80}\n")
    
    # Build furniture mapping for this episode
    furniture_map = build_furniture_handle_to_name_map([episode])
    
    # Extract entities
    entities = extract_scene_from_episode(episode, 0, furniture_map)
    
    # Save output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_data = {
        "episode_id": episode['episode_id'],
        "scene_id": episode['scene_id'],
        "instruction": episode['instruction'],
        "entities": entities
    }
    
    scene_file = output_path / f"scene_{episode['scene_id']}.json"
    with open(scene_file, 'w') as f:
        json.dump(scene_data, f, indent=2)
    
    print(f"Extracted {len(entities)} entities")
    print(f"\n{'='*80}")
    print(f"Scene information saved to: {scene_file}")
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
        extract_single_episode(args.dataset, args.episode_id, args.output)
    else:
        # Process all unique scenes
        extract_all_scenes(args.dataset, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
