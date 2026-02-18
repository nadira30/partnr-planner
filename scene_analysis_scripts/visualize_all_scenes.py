#!/usr/bin/env python3
"""
Script to visualize all unique scenes in val_mini.json.gz
and save top-down layout images to a folder.
"""

import subprocess
import json
import gzip
import os
from collections import defaultdict
from pathlib import Path


def load_dataset(dataset_path):
    """Load and parse the dataset file."""
    print(f"Loading dataset: {dataset_path}")
    with gzip.open(dataset_path, 'rt') as f:
        data = json.load(f)
    return data


def get_scenes_and_episodes(dataset):
    """Get mapping of scenes to episode IDs."""
    scene_to_episodes = defaultdict(list)
    
    for i, episode in enumerate(dataset['episodes']):
        scene_id = episode['scene_id']
        scene_to_episodes[scene_id].append({
            'index': i,
            'episode_id': episode['episode_id'],
            'instruction': episode['instruction']
        })
    
    return scene_to_episodes


def visualize_scene(episode_id, dataset_path, output_folder, scene_id):
    """
    Run skill_runner to visualize a scene and save the top-down image.
    Returns path to the generated image if successful, None otherwise.
    """
    print(f"  Visualizing episode {episode_id}...")
    
    # Create a temporary directory for this scene's output
    temp_dir = Path(output_folder) / "temp" / scene_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Track existing topdown files before running
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    existing_topdowns = set(results_dir.glob("topdown*.png"))
    
    cmd = [
        'python', '-m', 'habitat_llm.examples.skill_runner',
        f'hydra.run.dir={temp_dir}',
        '+skill_runner_show_topdown=True',
        f'habitat.dataset.data_path={dataset_path}',
        f'+skill_runner_episode_id={episode_id}',
        '+skill_runner_show_videos=False',
        'evaluation.save_video=True'  # Required for topdown image to be saved
    ]
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Send exit command immediately to just generate the topdown view
        output, _ = process.communicate(input='exit\n', timeout=120)
        
        # Look for NEW topdown images (created after we started)
        new_topdowns = set(results_dir.glob("topdown*.png")) - existing_topdowns
        
        if new_topdowns:
            # Return the newest one
            return max(new_topdowns, key=lambda p: p.stat().st_mtime)
        else:
            print(f"  Warning: No topdown image generated for episode {episode_id}")
            return None
        
    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout for episode {episode_id}")
        process.kill()
        return None
    except Exception as e:
        print(f"  Error visualizing episode {episode_id}: {e}")
        return None


def organize_output(topdown_image_path, scene_id, episode_id, output_folder):
    """
    Move and rename the generated topdown image to a more organized location.
    """
    if not topdown_image_path or not topdown_image_path.exists():
        return None
    
    # Create scene-specific folder
    scene_folder = Path(output_folder) / scene_id
    scene_folder.mkdir(exist_ok=True)
    
    # Copy to organized location with scene name
    import shutil
    dest_file = scene_folder / f"scene_{scene_id}_topdown.png"
    shutil.copy2(topdown_image_path, dest_file)
    print(f"  Saved: {dest_file}")
    return dest_file


def create_scene_info_file(output_folder, scene_id, episodes_info):
    """Create a text file with scene information."""
    scene_folder = Path(output_folder) / scene_id
    scene_folder.mkdir(exist_ok=True)
    
    info_file = scene_folder / "scene_info.txt"
    
    with open(info_file, 'w') as f:
        f.write(f"Scene ID: {scene_id}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total Episodes: {len(episodes_info)}\n\n")
        f.write(f"Sample Episodes:\n")
        f.write(f"{'-'*80}\n\n")
        
        for i, ep_info in enumerate(episodes_info[:5]):  # First 5 episodes
            f.write(f"Episode {ep_info['index']} (ID: {ep_info['episode_id']}):\n")
            f.write(f"  {ep_info['instruction']}\n\n")
        
        if len(episodes_info) > 5:
            f.write(f"... and {len(episodes_info) - 5} more episodes\n")
    
    print(f"  Created info file: {info_file}")


def main():
    # Configuration
    dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
    output_folder = "scene_visualizations"
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SCENE VISUALIZATION SCRIPT")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Get scenes and their episodes
    scene_to_episodes = get_scenes_and_episodes(dataset)
    
    print(f"\nFound {len(scene_to_episodes)} unique scenes")
    print(f"Total episodes: {len(dataset['episodes'])}\n")
    
    # Visualize each scene
    for scene_idx, (scene_id, episodes_info) in enumerate(sorted(scene_to_episodes.items()), 1):
        print(f"\n[{scene_idx}/{len(scene_to_episodes)}] Processing Scene: {scene_id}")
        print(f"  Episodes using this scene: {len(episodes_info)}")
        
        # Use the first episode for this scene
        first_episode = episodes_info[0]
        episode_id = first_episode['episode_id']
        
        print(f"  Sample instruction: {first_episode['instruction'][:80]}...")
        
        # Visualize the scene
        topdown_image = visualize_scene(episode_id, dataset_path, output_folder, scene_id)
        
        if topdown_image:
            # Organize the output
            organize_output(topdown_image, scene_id, episode_id, output_folder)
            
            # Create info file
            create_scene_info_file(output_folder, scene_id, episodes_info)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"All scene visualizations saved to: {output_folder}/")
    print(f"\nScene folders created:")
    
    for scene_id in sorted(scene_to_episodes.keys()):
        scene_folder = Path(output_folder) / scene_id
        if scene_folder.exists():
            files = list(scene_folder.glob("*.png"))
            print(f"  • {scene_id}/ ({len(files)} image(s))")
    
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
