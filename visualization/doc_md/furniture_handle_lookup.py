#!/usr/bin/env python3
"""
Helper class to look up furniture handles from pre-extracted mapping files.
This avoids needing to run the simulator just to get furniture handles.

Usage:
    from furniture_handle_lookup import FurnitureHandleLookup
    
    lookup = FurnitureHandleLookup("visualization/data/furniture_handles_val_mini.json")
    handle = lookup.get_furniture_handle(episode_id="100", furniture_name="table_36")
"""

import json
from pathlib import Path
from typing import Optional, Dict


class FurnitureHandleLookup:
    """Helper class to look up furniture handles from pre-extracted data."""
    
    def __init__(self, mapping_file: str):
        """
        Initialize the lookup with a mapping file.
        
        :param mapping_file: Path to JSON file with furniture handle mappings
        """
        self.mapping_file = Path(mapping_file)
        self.data = {}
        
        if self.mapping_file.exists():
            with open(self.mapping_file, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Loaded furniture handles for {len(self.data)} episodes")
        else:
            print(f"⚠ Warning: Mapping file not found: {mapping_file}")
            print(f"  Run extract_all_furniture_handles.py to generate it")
    
    def get_furniture_handle(self, episode_id: str, furniture_name: str) -> Optional[str]:
        """
        Get the handle for a specific furniture in an episode.
        
        :param episode_id: Episode ID
        :param furniture_name: Furniture name (e.g., "table_36")
        :return: Furniture handle or None if not found
        """
        episode_data = self.data.get(str(episode_id))
        if not episode_data:
            return None
        
        furniture_handles = episode_data.get("furniture_handles", {})
        return furniture_handles.get(furniture_name)
    
    def get_all_furniture_for_episode(self, episode_id: str) -> Dict[str, str]:
        """
        Get all furniture name-to-handle mappings for an episode.
        
        :param episode_id: Episode ID
        :return: Dictionary mapping furniture names to handles
        """
        episode_data = self.data.get(str(episode_id))
        if not episode_data:
            return {}
        
        return episode_data.get("furniture_handles", {})
    
    def get_scene_id(self, episode_id: str) -> Optional[str]:
        """
        Get the scene ID for an episode.
        
        :param episode_id: Episode ID
        :return: Scene ID or None if not found
        """
        episode_data = self.data.get(str(episode_id))
        if not episode_data:
            return None
        
        return episode_data.get("scene_id")
    
    def has_episode(self, episode_id: str) -> bool:
        """Check if episode data exists in the mapping."""
        return str(episode_id) in self.data
