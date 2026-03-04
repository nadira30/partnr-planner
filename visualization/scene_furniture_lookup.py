#!/usr/bin/env python3
"""
Utility to get furniture/receptacle positions from scene files.
Maps episode IDs to scene data and extracts furniture information.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SceneFurnitureLookup:
    """Lookup furniture positions and details from scene files."""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the lookup utility.
        
        Args:
            dataset_path: Path to the episode dataset (e.g., val_mini.json.gz)
        """
        if dataset_path is None:
            # Default: relative to visualization directory
            dataset_path = Path(__file__).parent.parent / "data" / "datasets" / "partnr_episodes" / "v0_0" / "val_mini.json.gz"
        
        self.dataset_path = Path(dataset_path)
        
        # scene_dir should be in the same data/ folder as datasets/
        # e.g., if dataset is /project/data/datasets/..., then scenes are in /project/data/hssd-hab/...
        if self.dataset_path.is_absolute():
            # Go up from .../data/datasets/partnr_episodes/v0_0/val_mini.json.gz
            # to .../data/ then add hssd-hab/scenes-partnr-filtered
            data_root = self.dataset_path.parent.parent.parent.parent  # Go up 4 levels to reach data/
            self.scene_dir = data_root / "hssd-hab" / "scenes-partnr-filtered"
        else:
            self.scene_dir = Path("../data/hssd-hab/scenes-partnr-filtered")
        
        self.episodes_cache = None
    
    def load_episodes(self) -> Dict:
        """Load the episode dataset."""
        if self.episodes_cache is not None:
            return self.episodes_cache
        
        if self.dataset_path.suffix == '.gz':
            with gzip.open(self.dataset_path, 'rt') as f:
                self.episodes_cache = json.load(f)
        else:
            with open(self.dataset_path) as f:
                self.episodes_cache = json.load(f)
        
        return self.episodes_cache
    
    def get_scene_id_from_episode(self, episode_id: str) -> Optional[str]:
        """
        Get scene ID from episode ID.
        
        Args:
            episode_id: The episode ID (e.g., "100")
        
        Returns:
            Scene ID string or None if not found
        """
        data = self.load_episodes()
        
        for episode in data['episodes']:
            if str(episode['episode_id']) == str(episode_id):
                return episode['scene_id'].split('/')[-1].replace('.scene_instance.json', '')
        
        return None
    
    def load_scene_data(self, scene_id: str) -> Optional[Dict]:
        """
        Load scene data from scene_instance.json file.
        
        Args:
            scene_id: The scene ID
        
        Returns:
            Scene data dictionary or None if file not found
        """
        scene_file = self.scene_dir / f"{scene_id}.scene_instance.json"
        
        if not scene_file.exists():
            print(f"Scene file not found: {scene_file}")
            return None
        
        with open(scene_file) as f:
            return json.load(f)
    
    def get_furniture_by_handle(self, scene_id: str, handle: str) -> Optional[Dict]:
        """
        Get furniture information by its template handle.
        
        Args:
            scene_id: The scene ID
            handle: Template handle (e.g., "62d5b81040a4546e5fda73df2e6a9648eb6ceb52")
        
        Returns:
            Furniture info dictionary or None if not found
        """
        scene_data = self.load_scene_data(scene_id)
        if scene_data is None:
            return None
        
        # Search in object_instances
        if 'object_instances' in scene_data:
            for idx, obj in enumerate(scene_data['object_instances']):
                template = obj.get('template_name', '')
                if template == handle or template.startswith(handle):
                    translation = obj.get('translation', [0, 0, 0])
                    rotation = obj.get('rotation', [0, 0, 0, 1])
                    
                    return {
                        'position': tuple(translation),
                        'rotation': tuple(rotation),
                        'template_name': template,
                        'object_type': 'object',
                        'index': idx,
                        'handle': template
                    }
        
        # Search in articulated_object_instances
        if 'articulated_object_instances' in scene_data:
            for idx, obj in enumerate(scene_data['articulated_object_instances']):
                template = obj.get('template_name', '')
                if template == handle or template.startswith(handle):
                    translation = obj.get('translation', [0, 0, 0])
                    rotation = obj.get('rotation', [0, 0, 0, 1])
                    
                    return {
                        'position': tuple(translation),
                        'rotation': tuple(rotation),
                        'template_name': template,
                        'object_type': 'articulated_object',
                        'index': idx,
                        'handle': template
                    }
        
        return None
    
    def get_furniture_by_name(self, scene_id: str, furniture_name: str) -> Optional[Dict]:
        """
        Get furniture information by its name from scene data.
        Note: Scene files don't have 'name' fields, so this searches by handle.
        Use get_furniture_by_handle() for direct lookups.
        
        Args:
            scene_id: The scene ID
            furniture_name: Name of furniture or handle
        
        Returns:
            Dictionary containing position, rotation, etc. or None if not found
        """
        # Try as handle first
        return self.get_furniture_by_handle(scene_id, furniture_name)
    
    def get_all_furniture(self, scene_id: str) -> List[Dict]:
        """
        Get all furniture/objects in a scene.
        
        Args:
            scene_id: The scene ID
        
        Returns:
            List of furniture info dictionaries
        """
        scene_data = self.load_scene_data(scene_id)
        if scene_data is None:
            return []
        
        furniture = []
        
        # Get object_instances
        if 'object_instances' in scene_data:
            for idx, obj in enumerate(scene_data['object_instances']):
                template = obj.get('template_name', '')
                translation = obj.get('translation', [0, 0, 0])
                rotation = obj.get('rotation', [0, 0, 0, 1])
                
                furniture.append({
                    'position': tuple(translation),
                    'rotation': tuple(rotation),
                    'template_name': template,
                    'object_type': 'object',
                    'index': idx,
                    'handle': template
                })
        
        # Get articulated_object_instances
        if 'articulated_object_instances' in scene_data:
            for idx, obj in enumerate(scene_data['articulated_object_instances']):
                template = obj.get('template_name', '')
                translation = obj.get('translation', [0, 0, 0])
                rotation = obj.get('rotation', [0, 0, 0, 1])
                
                furniture.append({
                    'position': tuple(translation),
                    'rotation': tuple(rotation),
                    'template_name': template,
                    'object_type': 'articulated_object',
                    'index': idx,
                    'handle': template
                })
        
        return furniture
    
    def get_furniture_from_episode(self, episode_id: str, furniture_name: str) -> Optional[Dict]:
        """
        Get furniture information directly from episode ID.
        
        Args:
            episode_id: The episode ID
            furniture_name: Name of furniture
        
        Returns:
            Furniture info dictionary or None
        """
        scene_id = self.get_scene_id_from_episode(episode_id)
        if scene_id is None:
            return None
        
        return self.get_furniture_by_name(scene_id, furniture_name)
    
    def get_all_furniture_from_episode(self, episode_id: str) -> List[Dict]:
        """
        Get all furniture from episode ID.
        
        Args:
            episode_id: The episode ID
        
        Returns:
            List of furniture info dictionaries
        """
        scene_id = self.get_scene_id_from_episode(episode_id)
        if scene_id is None:
            return []
        
        return self.get_all_furniture(scene_id)


def main():
    """Example usage."""
    import sys
    
    lookup = SceneFurnitureLookup()
    
    # Test with episode 100
    episode_id = sys.argv[1] if len(sys.argv) > 1 else "100"
    furniture_handle = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Looking up episode {episode_id}")
    
    scene_id = lookup.get_scene_id_from_episode(episode_id)
    if scene_id:
        print(f"Scene ID: {scene_id}\n")
        
        if furniture_handle:
            # Look up specific furniture by handle
            print(f"Looking for furniture handle: {furniture_handle}")
            info = lookup.get_furniture_by_handle(scene_id, furniture_handle)
            if info:
                print(f"Found:")
                print(f"  Position: {info['position']}")
                print(f"  Rotation: {info['rotation']}")
                print(f"  Type: {info['object_type']}")
                print(f"  Handle: {info['handle']}")
            else:
                print(f"Not found")
        else:
            # List all furniture
            print("All objects/furniture in scene:")
            all_furniture = lookup.get_all_furniture(scene_id)
            
            print(f"Total: {len(all_furniture)} objects")
            print("\nFirst 10 objects:")
            for i, info in enumerate(all_furniture[:10]):
                pos = info['position']
                handle = info['handle'][:40] + '...' if len(info['handle']) > 40 else info['handle']
                print(f"  {i:3d}. {handle:45s} at ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) [{info['object_type']}]")
    else:
        print(f"Episode {episode_id} not found")


if __name__ == '__main__':
    main()
