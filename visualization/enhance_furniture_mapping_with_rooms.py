#!/usr/bin/env python3
"""
Enhance furniture_handles mapping file to include room associations.
Uses episode data directly without requiring habitat_sim.
"""

import json
import gzip
import sys
from pathlib import Path
from collections import defaultdict


def extract_furniture_rooms_from_episode(episode: dict, furniture_names: set) -> dict:
    """
    Extract furniture to room mapping from episode data by analyzing initial_state.
    
    Returns dict: {furniture_name: room_name}
    """
    furniture_rooms = {}
    
    # Try to extract from initial_state
    initial_state = episode.get('info', {}).get('initial_state', [])
    for state in initial_state:
        furniture_names_in_state = state.get('furniture_names', [])
        allowed_regions = state.get('allowed_regions', [])
        
        if furniture_names_in_state and allowed_regions:
            # Each furniture name in this state belongs to the first allowed region
            room = allowed_regions[0]
            for furn_name in furniture_names_in_state:
                furniture_rooms[furn_name] = room
    
    return furniture_rooms


def enhance_furniture_handles_mapping(mapping_file: str, dataset_path: str) -> dict:
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
    # Load existing mapping
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    print(f"✓ Loaded mapping for {len(mapping_data)} episodes")
    
    # Load dataset
    if dataset_path.endswith('.gz'):
        with gzip.open(dataset_path, 'rt') as f:
            dataset = json.load(f)
    else:
        with open(dataset_path) as f:
            dataset = json.load(f)
    
    print(f"✓ Loaded dataset with {len(dataset.get('episodes', []))} episodes")
    
    # Create lookup for episodes
    episodes_by_id = {}
    for ep in dataset.get('episodes', []):
        episodes_by_id[str(ep.get('episode_id'))] = ep
    
    # Enhance each episode with room information
    enhanced_count = 0
    for episode_id, episode_data in mapping_data.items():
        try:
            print(f"\n📍 Processing episode {episode_id}...")
            
            if episode_id not in episodes_by_id:
                print(f"  ⚠ Episode {episode_id} not found in dataset")
                continue
            
            episode = episodes_by_id[episode_id]
            
            # Get furniture names from current mapping
            furniture_names = set(episode_data.get('furniture_handles', {}).keys())
            furniture_names.discard('floor_')  # Discard floor entries
            
            # Extract room mapping
            furniture_rooms = extract_furniture_rooms_from_episode(episode, furniture_names)
            
            if furniture_rooms:
                episode_data['furniture_rooms'] = furniture_rooms
                enhanced_count += 1
                
                # Print sample
                sample_items = sorted(list(furniture_rooms.items())[:5])
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
    
    print(f"\n{'='*60}")
    print(f"✓ Enhanced {enhanced_count}/{len(mapping_data)} episodes with room associations")
    print(f"{'='*60}")
    
    return mapping_data


def main():
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
    
    # Enhance the mapping
    enhanced_mapping = enhance_furniture_handles_mapping(str(mapping_file), str(dataset_path))
    
    # Save enhanced mapping
    backup_file = mapping_file.with_suffix('.json.backup')
    print(f"\n💾 Saving backup to: {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(enhanced_mapping, f, indent=2)
    
    # Overwrite original with enhanced version
    print(f"💾 Saving enhanced mapping to: {mapping_file}")
    with open(mapping_file, 'w') as f:
        json.dump(enhanced_mapping, f, indent=2)
    
    print(f"\n✓ Successfully enhanced furniture_handles mapping with room associations!")


if __name__ == "__main__":
    main()
