#!/usr/bin/env python3
"""
Script to generate scene information files for all scenes in val_mini.json.gz
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict


def load_dataset(dataset_path):
    """Load and parse the dataset file."""
    print(f"Loading dataset: {dataset_path}")
    with gzip.open(dataset_path, 'rt') as f:
        data = json.load(f)
    return data


def load_scene_file(scene_id, scenes_dir="data/hssd-hab/scenes-partnr-filtered"):
    """Load the scene JSON file."""
    scene_path = Path(scenes_dir) / f"{scene_id}.scene_instance.json"
    
    if not scene_path.exists():
        print(f"  Warning: Scene file not found: {scene_path}")
        return None
    
    with open(scene_path, 'r') as f:
        return json.load(f)


def create_scene_info_file(output_folder, scene_id, episodes_info, scene_data):
    """Create a text file with scene information."""
    scene_folder = Path(output_folder) / scene_id
    scene_folder.mkdir(exist_ok=True, parents=True)
    
    info_file = scene_folder / "scene_info.txt"
    
    with open(info_file, 'w') as f:
        f.write(f"Scene ID: {scene_id}\n")
        f.write(f"{'='*80}\n\n")
        
        # Scene stats
        if scene_data and 'object_instances' in scene_data:
            f.write(f"Total Objects in Scene: {len(scene_data['object_instances'])}\n")
        
        f.write(f"Total Episodes using this scene: {len(episodes_info)}\n\n")
        f.write(f"Sample Episodes:\n")
        f.write(f"{'-'*80}\n\n")
        
        for i, ep_info in enumerate(episodes_info[:10]):  # First 10 episodes
            f.write(f"Episode {ep_info['index']} (ID: {ep_info['episode_id']}):\n")
            f.write(f"  {ep_info['instruction']}\n\n")
        
        if len(episodes_info) > 10:
            f.write(f"... and {len(episodes_info) - 10} more episodes\n")
    
    print(f"  Created info file: {info_file}")


def main():
    # Configuration
    dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
    output_folder = "scene_visualizations"
    scenes_dir = "data/hssd-hab/scenes-partnr-filtered"
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SCENE INFORMATION GENERATOR")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Get scenes and their episodes
    scene_to_episodes = defaultdict(list)
    for i, episode in enumerate(dataset['episodes']):
        scene_id = episode['scene_id']
        scene_to_episodes[scene_id].append({
            'index': i,
            'episode_id': episode['episode_id'],
            'instruction': episode['instruction']
        })
    
    print(f"\nFound {len(scene_to_episodes)} unique scenes")
    print(f"Total episodes: {len(dataset['episodes'])}\n")
    
    # Process each scene
    for scene_idx, (scene_id, episodes_info) in enumerate(sorted(scene_to_episodes.items()), 1):
        print(f"\n[{scene_idx}/{len(scene_to_episodes)}] Processing Scene: {scene_id}")
        print(f"  Episodes using this scene: {len(episodes_info)}")
        
        first_episode = episodes_info[0]
        print(f"  Sample instruction: {first_episode['instruction'][:80]}...")
        
        # Load scene file
        scene_data = load_scene_file(scene_id, scenes_dir)
        
        # Create info file
        create_scene_info_file(output_folder, scene_id, episodes_info, scene_data)
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"All scene information files saved to: {output_folder}/")
    print(f"\nScene folders created:")
    
    for scene_id in sorted(scene_to_episodes.keys()):
        scene_folder = Path(output_folder) / scene_id
        if scene_folder.exists():
            info_files = list(scene_folder.glob("scene_info.txt"))
            print(f"  • {scene_id}/ ({len(info_files)} info file(s))")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
