#!/usr/bin/env python3
"""
Script to save episode data with added objects from the API to a new JSON file.

This script:
1. Reads the original episode data from val_mini.json.gz
2. Fetches added objects from the Flask API
3. Merges the added objects into the episode data
4. Saves the modified episode to a timestamped file in visualization/data/

Usage:
    python save_episode_with_additions.py --episode 100
    python save_episode_with_additions.py --episode 100 --api-url http://localhost:5002
    python save_episode_with_additions.py --episode 100 --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz
"""

import argparse
import gzip
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_episode_from_gz(dataset_path: Path, episode_id: str) -> Dict[str, Any]:
    """Load a specific episode from the gzipped JSON dataset."""
    print(f"Reading dataset from: {dataset_path}")
    
    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find the episode
    episodes = data.get('episodes', [])
    for episode in episodes:
        if str(episode.get('episode_id')) == str(episode_id):
            print(f"✓ Found episode {episode_id}")
            return episode
    
    raise ValueError(f"Episode {episode_id} not found in dataset")


def get_added_objects_from_api(api_url: str, episode_id: str) -> List[Dict[str, Any]]:
    """Fetch added objects from the Flask API."""
    try:
        response = requests.get(f"{api_url}/api/episode/{episode_id}/export-config", timeout=10)
        
        if response.status_code == 404 or response.status_code == 400:
            # No added objects
            print(f"ℹ No objects have been added via the API for episode {episode_id}")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        # Extract the added objects detail
        added_objs = data.get('metadata', {}).get('objects_detail', [])
        print(f"✓ Retrieved {len(added_objs)} added objects from API")
        return added_objs
        
    except requests.exceptions.ConnectionError:
        print(f"⚠ Warning: Could not connect to API at {api_url}")
        print(f"  Make sure the Flask app is running.")
        return []
    except Exception as e:
        print(f"⚠ Warning: Error fetching added objects from API: {e}")
        return []


def merge_added_objects_into_episode(episode_data: Dict[str, Any], added_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge added objects into the episode's additional_obj_config_paths structure."""
    if not added_objects:
        print("No added objects to merge")
        return episode_data
    
    # Create a copy to avoid modifying the original
    modified_episode = episode_data.copy()
    
    # Ensure additional_obj_config_paths exists
    if 'additional_obj_config_paths' not in modified_episode:
        modified_episode['additional_obj_config_paths'] = []
    
    # Group objects by (room, furniture, object_category) for efficient config format
    grouped = {}
    for obj in added_objects:
        key = (obj['room'], obj['furniture'], obj['object_category'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(obj)
    
    # Create config items
    config_items = []
    for (room, furniture, obj_category), objs in grouped.items():
        config_items.append({
            "number": len(objs),
            "object_classes": [obj_category],
            "allowed_regions": [room],
            "furniture_names": [furniture]
        })
    
    # Create a config structure for the added objects
    added_config = {
        "type": "added_via_visualizer",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_state": config_items,
        "objects_detail": added_objects
    }
    
    # Add to the episode's config
    modified_episode['additional_obj_config_paths'].append(added_config)
    
    print(f"✓ Merged {len(added_objects)} objects into episode data")
    print(f"  - Created {len(config_items)} config items")
    
    return modified_episode


def save_modified_episode(episode_data: Dict[str, Any], output_dir: Path, episode_id: str):
    """Save the modified episode to a timestamped JSON file."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"episode_{episode_id}_modified_{timestamp}.json"
    output_path = output_dir / filename
    
    # Save as pretty-printed JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved modified episode to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    
    return output_path


def save_full_dataset_with_modifications(original_dataset_path: Path, episode_id: str, 
                                         modified_episode: Dict[str, Any], output_dir: Path):
    """
    Save the entire dataset with the modified episode.
    This creates a new complete dataset file with one episode modified.
    """
    print(f"\nCreating full dataset with modified episode {episode_id}...")
    
    # Load the full dataset
    with gzip.open(original_dataset_path, 'rt', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Find and replace the episode
    episodes = full_data.get('episodes', [])
    modified = False
    
    for i, episode in enumerate(episodes):
        if str(episode.get('episode_id')) == str(episode_id):
            episodes[i] = modified_episode
            modified = True
            print(f"✓ Replaced episode {episode_id} in dataset")
            break
    
    if not modified:
        print(f"⚠ Warning: Episode {episode_id} not found in dataset for replacement")
        return None
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"val_mini_modified_{timestamp}.json.gz"
    output_path = output_dir / filename
    
    # Save as gzipped JSON
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved full modified dataset to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print(f"  Total episodes: {len(episodes)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Save episode data with added objects from API to JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save episode 100 with added objects
  python save_episode_with_additions.py --episode 100
  
  # Save with custom API URL
  python save_episode_with_additions.py --episode 100 --api-url http://localhost:5002
  
  # Save full dataset (not just single episode)
  python save_episode_with_additions.py --episode 100 --save-full-dataset
        """
    )
    
    parser.add_argument(
        '--episode',
        type=str,
        required=True,
        help='Episode ID to save'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/datasets/partnr_episodes/v0_0/val_mini.json.gz',
        help='Path to the dataset file (default: data/datasets/partnr_episodes/v0_0/val_mini.json.gz)'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:5002',
        help='Base URL for the Flask API (default: http://localhost:5002)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualization/data',
        help='Output directory for saved files (default: visualization/data)'
    )
    
    parser.add_argument(
        '--save-full-dataset',
        action='store_true',
        help='Save the entire dataset with modifications (not just single episode)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Episode Save Script with Added Objects")
    print("="*80 + "\n")
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        return 1
    
    try:
        # Step 1: Load original episode
        print(f"Step 1: Loading episode {args.episode} from dataset...")
        episode_data = load_episode_from_gz(dataset_path, args.episode)
        
        # Step 2: Fetch added objects from API
        print(f"\nStep 2: Fetching added objects from API...")
        added_objects = get_added_objects_from_api(args.api_url, args.episode)
        
        # Step 3: Merge added objects into episode
        print(f"\nStep 3: Merging added objects into episode data...")
        modified_episode = merge_added_objects_into_episode(episode_data, added_objects)
        
        # Step 4: Save modified episode
        print(f"\nStep 4: Saving modified episode...")
        episode_file = save_modified_episode(modified_episode, output_dir, args.episode)
        
        # Step 5: Optionally save full dataset
        if args.save_full_dataset:
            print(f"\nStep 5: Saving full dataset with modifications...")
            dataset_file = save_full_dataset_with_modifications(
                dataset_path, args.episode, modified_episode, output_dir
            )
        
        print("\n" + "="*80)
        print("✓ SUCCESS")
        print("="*80)
        print(f"\nSaved files:")
        print(f"  • Single episode: {episode_file}")
        if args.save_full_dataset:
            print(f"  • Full dataset:   {dataset_file}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
