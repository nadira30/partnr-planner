#!/usr/bin/env python3
"""
Flask web application to visualize episode information including rooms, furniture, and objects.

Usage:
    conda activate habitat
    cd /home/nadira/partnr-planner/visualization
    python episode_visualizer_app.py

Then open http://localhost:5002 in your browser and enter an episode ID.
"""

import sys
import json
import gzip
import subprocess
import csv
import numpy as np
import io
import os
import yaml
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from typing import Dict, List, Optional
from collections import defaultdict

# Import scene furniture lookup utility
try:
    from scene_furniture_lookup import SceneFurnitureLookup
    FURNITURE_LOOKUP_AVAILABLE = True
except ImportError:
    print("⚠ Warning: scene_furniture_lookup not available")
    FURNITURE_LOOKUP_AVAILABLE = False

# Import furniture handle lookup utility
try:
    from furniture_handle_lookup import FurnitureHandleLookup
    FURNITURE_HANDLE_LOOKUP_AVAILABLE = True
except ImportError:
    print("⚠ Warning: furniture_handle_lookup not available")
    FURNITURE_HANDLE_LOOKUP_AVAILABLE = False

# Try to import habitat and scene utilities
HABITAT_AVAILABLE = False
SCENE_UTILS_AVAILABLE = False

try:
    import habitat
    from habitat_llm.utils.sim import find_receptacles
    from habitat_llm.world_model import WorldGraph
    HABITAT_AVAILABLE = True
except ImportError:
    print("⚠ Warning: habitat not available, simulator validation disabled")

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scene_analysis_scripts"))
    import scene_utils
    SCENE_UTILS_AVAILABLE = True
except ImportError:
    print("⚠ Warning: scene_utils not available, scene file positioning disabled")

app = Flask(__name__)

# Cache for episode data
_episode_cache = {}

# Storage for full episode JSON per episode
_episode_json_cache = {}

# Storage for added objects per episode
_added_objects = defaultdict(list)

# Object database for validation
object_database = []

# Furniture lookup utility
furniture_lookup = None
if FURNITURE_LOOKUP_AVAILABLE:
    try:
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "data" / "datasets" / "partnr_episodes" / "v0_0" / "val_mini.json.gz"
        furniture_lookup = SceneFurnitureLookup(dataset_path=str(dataset_path))
        print("✓ SceneFurnitureLookup initialized successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize furniture lookup: {e}")

# Furniture handle lookup utility
furniture_handle_lookup = None
if FURNITURE_HANDLE_LOOKUP_AVAILABLE:
    try:
        # Use absolute path from project root
        project_root = Path(__file__).parent.parent
        # Try sample file first for testing
        mapping_file = project_root / "visualization" / "data" / "furniture_handles_val_sample.json"
        if not mapping_file.exists():
            mapping_file = project_root / "visualization" / "data" / "furniture_handles_val_mini.json"
        
        if mapping_file.exists():
            furniture_handle_lookup = FurnitureHandleLookup(str(mapping_file))
        else:
            print(f"⚠ Warning: Furniture handle mapping file not found")
            print(f"  Run: python visualization/episode_info/extract_all_furniture_handles.py")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize furniture handle lookup: {e}")


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def find_receptacle_mesh_name(parent_handle: str) -> Optional[str]:
    """Find receptacle mesh name from URDF files."""
    if not SCENE_UTILS_AVAILABLE:
        return None
    
    try:
        parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
        
        # Search URDF files for receptacle mesh
        urdf_paths = [
            Path("data/hssd-hab/objects"),
            Path("data/objects"),
        ]
        
        for urdf_dir in urdf_paths:
            if not urdf_dir.exists():
                continue
            
            for urdf_file in urdf_dir.rglob(f"*{parent_handle_base}*.urdf"):
                with open(urdf_file, 'r') as f:
                    content = f.read()
                    # Look for receptacle_mesh links
                    if 'receptacle_mesh' in content:
                        import re
                        matches = re.findall(r'receptacle_mesh_[a-f0-9]+', content)
                        if matches:
                            return matches[0]
        return None
    except Exception as e:
        print(f"Error finding receptacle mesh: {e}")
        return None


def create_sim_for_episode(episode_dict: Dict) -> Optional[any]:
    """Create a temporary simulator instance for validation."""
    if not HABITAT_AVAILABLE:
        return None
    
    try:
        # This is a simplified version - you may need to adjust based on your config
        import habitat
        from habitat.config.default import get_config
        
        # Create minimal config for validation
        config = get_config()
        config.defrost()
        config.SIMULATOR.SCENE = episode_dict.get("scene_dataset_config", "")
        config.SIMULATOR.SCENE_DATASET = episode_dict.get("scene_id", "")
        config.freeze()
        
        sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)
        return sim
    except Exception as e:
        print(f"Could not create simulator: {e}")
        return None


def load_full_episode_json(episode_id: str, dataset_path: str = None) -> Dict:
    """Load the complete episode JSON from the dataset file."""
    if dataset_path is None:
        dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
    
    cache_key = f"{dataset_path}:{episode_id}"
    if cache_key in _episode_json_cache:
        return _episode_json_cache[cache_key]
    
    try:
        project_root = Path(__file__).parent.parent
        full_path = project_root / dataset_path
        
        with gzip.open(full_path, 'rt', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Find the episode
        for ep in dataset.get('episodes', []):
            if str(ep.get('episode_id')) == str(episode_id):
                _episode_json_cache[cache_key] = ep
                return ep
        
        raise ValueError(f"Episode {episode_id} not found in dataset")
    except Exception as e:
        raise Exception(f"Error loading episode JSON: {e}")


def load_object_database():
    """Load object database from CSV files for validation."""
    global object_database
    
    if object_database:
        return object_database
    
    csv_paths = [
        Path("data/hssd-hab/metadata/object_categories_filtered.csv"),
        Path("visualization/objects/object_categories_one_per_class.csv"),
    ]
    
    project_root = Path(__file__).parent.parent
    
    for csv_path in csv_paths:
        full_path = project_root / csv_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'id' in row:
                            object_database.append({
                                'id': row['id'],
                                'category': row.get('category', row.get('clean_category', ''))
                            })
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")
    
    return object_database


def get_episode_data(episode_id: str, dataset_path: str = None) -> Dict:
    """
    Extract episode data by calling the get_episode_entities script.
    """
    if dataset_path is None:
        dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
    
    # Check cache (but skip cache if there are added objects for this episode)
    cache_key = f"{dataset_path}:{episode_id}"
    if cache_key in _episode_cache and episode_id not in _added_objects:
        return _episode_cache[cache_key]
    
    # If there are added objects, we need to rebuild the data
    if episode_id in _added_objects and cache_key in _episode_cache:
        # Use cached base data and add new objects
        cached_data = _episode_cache[cache_key].copy()
        object_locations = cached_data.get('object_locations', {}).copy()
        
        # Add any objects that were added via the UI
        for added_obj in _added_objects.get(episode_id, []):
            obj_name = added_obj['object_name']
            object_locations[obj_name] = {
                'room': added_obj['room'],
                'furniture': added_obj['furniture']
            }
        
        result_data = cached_data.copy()
        result_data['object_locations'] = object_locations
        return result_data
    
    try:
        # Get the project root
        project_root = Path(__file__).parent.parent
        script_path = project_root / "visualization" / "episode_info" / "get_episode_entities.py"
        output_file = project_root / "visualization" / f"temp_episode_{episode_id}.json"
        
        # Use the same Python interpreter that's running this Flask app
        # This ensures we use the habitat conda environment
        python_exe = sys.executable
        
        # Run the extraction script
        cmd = [
            python_exe,
            str(script_path),
            f"+episode_id={episode_id}",
            f"+dataset_path={dataset_path}",
            f"+output_file={output_file}",
            "+print_handles=false",
            "hydra.run.dir=."
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise Exception(f"Script failed with code {result.returncode}: {result.stderr}")
        
        # Read the generated JSON file
        if not output_file.exists():
            raise Exception(f"Output file was not created at {output_file}")
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Clean up temp file
        output_file.unlink()
        
        # Transform data for frontend
        furniture_by_room = {}
        for furniture_name, room_name in data.get('furniture_locations', {}).items():
            if room_name not in furniture_by_room:
                furniture_by_room[room_name] = []
            furniture_by_room[room_name].append(furniture_name)
        
        # Sort furniture lists
        for room in furniture_by_room:
            furniture_by_room[room] = sorted(furniture_by_room[room])
        
        # Merge original objects with added objects
        object_locations = data.get('object_locations', {}).copy()
        
        # Add any objects that were added via the UI
        for added_obj in _added_objects.get(episode_id, []):
            obj_name = added_obj['object_name']
            object_locations[obj_name] = {
                'room': added_obj['room'],
                'furniture': added_obj['furniture']
            }
        
        result_data = {
            "episode_id": data.get('episode_id'),
            "scene_id": data.get('scene_id'),
            "rooms": sorted(data.get('entities', {}).get('rooms', [])),
            "furniture_by_room": furniture_by_room,
            "object_locations": object_locations,
        }
        
        # Cache result
        _episode_cache[cache_key] = result_data
        
        print(f"✓ Successfully loaded episode {episode_id}")
        
        return result_data
        
    except Exception as e:
        import traceback
        print(f"ERROR loading episode {episode_id}:")
        print(traceback.format_exc())
        raise Exception(f"Error loading episode {episode_id}: {str(e)}")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('episode_visualizer.html')


@app.route('/api/episode/<episode_id>')
def get_episode(episode_id):
    """API endpoint to get episode data."""
    try:
        dataset_path = request.args.get('dataset', 'data/datasets/partnr_episodes/v0_0/val_mini.json.gz')
        data = get_episode_data(episode_id, dataset_path)
        return jsonify(data)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"ERROR loading episode {episode_id}:")
        print(error_msg)
        return jsonify({'error': str(e), 'details': error_msg}), 400


@app.route('/api/object-categories')
def get_object_categories():
    """API endpoint to get available object categories from CSV."""
    try:
        csv_path = Path(__file__).parent / 'objects' / 'object_categories_one_per_class.csv'
        
        if not csv_path.exists():
            return jsonify({'error': 'CSV file not found'}), 404
        
        categories = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories.append({
                    'id': row['id'],
                    'category': row['clean_category']
                })
        
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/episode/<episode_id>/add-object', methods=['POST'])
def add_object(episode_id):
    """Add a new object to the episode with full episode modification."""
    try:
        data = request.get_json()
        
        object_id = data.get("object_id")
        object_class = data.get("object_class") or data.get("object_category")
        room = data.get("room")
        furniture = data.get("furniture")
        preposition = data.get("preposition", "on")
        position = data.get("position", {"x": 0, "y": 0.5, "z": 0})
        
        if not all([object_class, room, furniture]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Look up furniture handle from pre-extracted mapping based on episode_id and furniture name
        receptacle_handle = None
        if furniture != "floor" and furniture_handle_lookup and furniture_handle_lookup.has_episode(episode_id):
            receptacle_handle = furniture_handle_lookup.get_furniture_handle(episode_id, furniture)
            if receptacle_handle:
                print(f"✓ Found furniture handle from mapping: {furniture} -> {receptacle_handle}")
            else:
                print(f"⚠ Furniture '{furniture}' not found in pre-extracted mapping for episode {episode_id}")
        
        # Allow override from request data
        if data.get("receptacle_handle"):
            receptacle_handle = data.get("receptacle_handle")
            print(f"✓ Using receptacle_handle from request: {receptacle_handle}")
        
        # Load object database for validation and lookup
        load_object_database()
        
        # If object_id not provided, look up the actual object ID from the category
        if not object_id:
            # Find object ID from category in database
            matching_objects = [obj for obj in object_database if obj['category'].lower() == object_class.lower()]
            if matching_objects:
                object_id = matching_objects[0]['id']
                print(f"✓ Found object ID '{object_id}' for category '{object_class}'")
            else:
                # Fallback: use category name as object ID
                object_id = object_class
                print(f"⚠ No object ID found for category '{object_class}', using category name")
        
        # Validate that object exists in metadata CSV
        base_object_id = object_id.split('/')[-1].replace('.object_config.json', '')
        if '_:' in base_object_id:
            base_object_id = base_object_id.split('_:')[0]
        
        # Check if object is in database
        object_found = any(obj['id'] == base_object_id for obj in object_database)
        if not object_found:
            print(f"WARNING: Object '{base_object_id}' not found in metadata CSV files!")
            print(f"This object may fail to load or appear as 'unknown' in the scene.")
        
        # Load full episode JSON
        ep = load_full_episode_json(episode_id)
        
        print(f"\n{'='*60}")
        print(f"DEBUGGING ADD OBJECT:")
        print(f"  episode_id: {episode_id}")
        print(f"  furniture: {repr(furniture)}")
        print(f"  receptacle_handle (from lookup): {repr(receptacle_handle)}")
        print(f"  furniture_handle_lookup available: {furniture_handle_lookup is not None}")
        print(f"  has_episode({episode_id}): {furniture_handle_lookup.has_episode(episode_id) if furniture_handle_lookup else 'N/A'}")
        print(f"{'='*60}\n")
        
        # If receptacle_handle not provided, derive it from furniture name
        furniture_handle = None
        furniture_position = None
        if not receptacle_handle and furniture != "floor":
            try:
                print(f"\n🔍 Looking up furniture handle for: {furniture}")
                
                # METHOD 1: Try pre-extracted furniture handle mapping (fastest, no simulator needed)
                if furniture_handle_lookup and furniture_handle_lookup.has_episode(episode_id):
                    furniture_handle = furniture_handle_lookup.get_furniture_handle(episode_id, furniture)
                    if furniture_handle:
                        print(f"✓ Found furniture handle from pre-extracted mapping: {furniture_handle}")
                
                # METHOD 2: Use initial_state to map furniture names to handles (works without simulator!)
                if not furniture_handle:
                    print(f"🔍 Trying to find furniture handle from initial_state...")
                    # Look through initial_state to find objects on this furniture
                    for state in ep.get('info', {}).get('initial_state', []):
                        if state.get('furniture_names') and furniture in state.get('furniture_names', []):
                            # Found an object that should be on this furniture
                            # Now find any of those objects in name_to_receptacle
                            object_classes = state.get('object_classes', [])
                            if object_classes:
                                obj_class = object_classes[0].lower()
                                # Search for this object type in name_to_receptacle
                                for obj_handle, recep in ep.get('name_to_receptacle', {}).items():
                                    if recep !='floor' and obj_class in obj_handle.lower():
                                        # Extract the furniture handle from the receptacle
                                        furniture_handle = recep.split('|')[0] if '|' in recep else recep
                                        furniture_handle = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
                                        print(f"✓ Found furniture handle from initial_state: {furniture_handle}")
                                        break
                            if furniture_handle:
                                break
                
                # METHOD 3: FALLBACK - Search scene file directly by furniture type and instance
                if not furniture_handle and FURNITURE_LOOKUP_AVAILABLE and furniture_lookup:
                    print(f"⚠ No existing objects  found on '{furniture}', searching scene file...")
                    scene_id_clean = ep.get("scene_id", "").split('/')[-1].replace('.scene_instance.json', '')
                    
                    # Get all furniture from scene
                    all_furniture = furniture_lookup.get_all_furniture(scene_id_clean)
                    
                    # Extract furniture type and instance number from name
                    # e.g., "counter_19" -> type: "counter", instance: 19
                    if '_' in furniture:
                        parts = furniture.rsplit('_', 1)
                        if parts[-1].isdigit():
                            furn_type = parts[0]
                            furn_instance = int(parts[-1])
                        else:
                            furn_type = furniture
                            furn_instance = None
                    else:
                        furn_type = furniture
                        furn_instance = None
                    
                    # Search for matching furniture in scene
                    # Match by type and if possible by instance number
                    furn_type_normalized = furn_type.replace('_', '').lower()
                    matching_furniture = []
                    
                    for furn_info in all_furniture:
                        handle = furn_info['handle']
                        obj_type = furn_info.get('object_type', '').lower()
                        
                        # Check if furniture type matches
                        if furn_type in obj_type or furn_type_normalized in obj_type.replace('_', ''):
                            matching_furniture.append(furn_info)
                    
                    if matching_furniture:
                        # If we have an instance number, try to match it
                        # Otherwise just use the first match
                        selected_furn = matching_furniture[0]
                        if furn_instance is not None and len(matching_furniture) > furn_instance:
                            # Try to get the correct instance (furniture are usually ordered)
                            selected_furn = matching_furniture[min(furn_instance, len(matching_furniture) - 1)]
                        
                        furniture_handle = selected_furn['handle']
                        furniture_position = selected_furn['position']
                        print(f"✓ Found furniture in scene file: {furniture_handle}")
                        print(f"✓ Position: {furniture_position}")
                        print(f"  (Found {len(matching_furniture)} matching '{furn_type}' furniture)")
                    else:
                        print(f"⚠ No furniture matching '{furn_type}' found in scene file")
                
                # METHOD 4: Look through existing objects to find one on this furniture (original METHOD 2)
                if not furniture_handle:
                    episode_data_viz = get_episode_data(episode_id)
                    
                    for obj_name, loc in episode_data_viz.get('object_locations', {}).items():
                        if loc.get('furniture') == furniture:
                            # Found an object on this furniture!
                            # Now find its receptacle in name_to_receptacle
                            for obj_handle, recep in ep.get('name_to_receptacle', {}).items():
                                # Match by object name (handle may have instance suffix)
                                obj_base_name = obj_name.split('_')[0]
                                if obj_handle.startswith(obj_base_name):
                                    # Extract furniture handle from receptacle
                                    # Format: "FURNITURE_HANDLE_:0004|receptacle_mesh_..."
                                    if '|' in recep and recep != "floor":
                                        furniture_handle = recep.split('|')[0]
                                        print(f"✓ Found furniture handle from existing object '{obj_name}': {furniture_handle}")
                                        break
                            if furniture_handle:
                                break
                    
                    if furniture_handle:
                        print(f"✓ Found furniture handle from existing objects: {furniture_handle}")
                
                # Get position from scene file if we have a handle
                if furniture_handle and FURNITURE_LOOKUP_AVAILABLE and furniture_lookup:
                    scene_id_clean = ep.get("scene_id", "").split('/')[-1].replace('.scene_instance.json', '')
                    # Remove instance suffix for scene lookup
                    furniture_handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
                    furniture_info = furniture_lookup.get_furniture_by_handle(scene_id_clean, furniture_handle_base)
                    if furniture_info:
                        furniture_position = furniture_info['position']
                        print(f"✓ Found furniture position: {furniture_position}")
                    else:
                        print(f"⚠ Furniture handle found but no position in scene file")
                
                # FALLBACK: If no existing objects on this furniture, search scene file directly
                if not furniture_handle and FURNITURE_LOOKUP_AVAILABLE and furniture_lookup:
                    print(f"⚠ No existing objects found on '{furniture}', searching scene file...")
                    scene_id_clean = ep.get("scene_id", "").split('/')[-1].replace('.scene_instance.json', '')
                    
                    # Get all furniture from scene
                    all_furniture = furniture_lookup.get_all_furniture(scene_id_clean)
                    
                    # Extract furniture type and instance number from name
                    # e.g., "table_36" -> type: "table", instance: 36
                    if '_' in furniture:
                        parts = furniture.rsplit('_', 1)
                        if parts[-1].isdigit():
                            furn_type = parts[0]
                            furn_instance = int(parts[-1])
                        else:
                            furn_type = furniture
                            furn_instance = None
                    else:
                        furn_type = furniture
                        furn_instance = None
                    
                    # Search for matching furniture in scene
                    # Match by type and if possible by instance number
                    furn_type_normalized = furn_type.replace('_', '').lower()
                    matching_furniture = []
                    
                    for furn_info in all_furniture:
                        handle = furn_info['handle']
                        obj_type = furn_info.get('object_type', '').lower()
                        
                        # Check if furniture type matches
                        if furn_type in obj_type or furn_type_normalized in obj_type.replace('_', ''):
                            matching_furniture.append(furn_info)
                    
                    if matching_furniture:
                        # If we have an instance number, try to match it
                        # Otherwise just use the first match
                        selected_furn = matching_furniture[0]
                        if furn_instance is not None and len(matching_furniture) > furn_instance:
                            # Try to get the correct instance (furniture are usually ordered)
                            selected_furn = matching_furniture[min(furn_instance, len(matching_furniture) - 1)]
                        
                        furniture_handle = selected_furn['handle']
                        furniture_position = selected_furn['position']
                        print(f"✓ Found furniture in scene file: {furniture_handle}")
                        print(f"✓ Position: {furniture_position}")
                        print(f"  (Found {len(matching_furniture)} matching '{furn_type}' furniture)")
                    else:
                        print(f"⚠ No furniture matching '{furn_type}' found in scene file")
                
                if not furniture_handle:
                    print(f"⚠ Could not determine furniture handle for '{furniture}'")
                    
            except Exception as e:
                import traceback
                print(f"⚠ Could not get furniture handle from episode data: {e}")
                print(traceback.format_exc())
        
        # ALWAYS retrieve furniture position if we have any handle (receptacle_handle OR furniture_handle)
        # This ensures position is retrieved regardless of where the handle came from
        if furniture != "floor" and not furniture_position:
            # Determine which handle to use
            handle_to_use = furniture_handle if furniture_handle else receptacle_handle
            
            if handle_to_use and FURNITURE_LOOKUP_AVAILABLE and furniture_lookup:
                try:
                    print(f"\n🔍 Retrieving furniture position for handle: {handle_to_use}")
                    scene_id_clean = ep.get("scene_id", "").split('/')[-1].replace('.scene_instance.json', '')
                    
                    # Extract base handle (remove instance suffix and mesh part if present)
                    if '|' in handle_to_use:
                        handle_to_use = handle_to_use.split('|')[0]
                    handle_base = handle_to_use.split('_:')[0] if '_:' in handle_to_use else handle_to_use
                    
                    furniture_info = furniture_lookup.get_furniture_by_handle(scene_id_clean, handle_base)
                    if furniture_info:
                        furniture_position = furniture_info['position']
                        print(f"✓ Retrieved furniture position: {furniture_position}")
                    else:
                        print(f"⚠ Handle '{handle_base}' not found in scene file")
                except Exception as e:
                    print(f"⚠ Could not retrieve furniture position: {e}")
        
        # Count explicit objects from initial_state (excludes clutter/template entries)
        explicit_object_count = 0
        for state in ep["info"]["initial_state"]:
            if (
                "name" not in state
                and "template_task_number" not in state
                and state.get("object_classes")
            ):
                explicit_object_count += 1
        
        # 1. Add to initial_state (before any "common sense"/clutter entries)
        new_state = {
            "number": 1,
            "object_classes": [object_class],
            "allowed_regions": [room],
            "furniture_names": [furniture],
        }
        
        # Find position to insert (before first entry with "name" or "template_task_number")
        state_insert_idx = len(ep["info"]["initial_state"])
        for i, state in enumerate(ep["info"]["initial_state"]):
            if "name" in state or "template_task_number" in state:
                state_insert_idx = i
                break
        
        ep["info"]["initial_state"].insert(state_insert_idx, new_state)
        
        print(f"\n{'='*60}")
        print(f"Adding object: {object_class}")
        print(f"  Object ID: {object_id}")
        print(f"  Room: {room}")
        print(f"  Furniture: {furniture}")
        print(f"  Receptacle handle: {receptacle_handle}")
        print(f"  Position in arrays: {explicit_object_count} (before clutter)")
        print(f"{'='*60}\n")
        
        # 2. Calculate proper object position and rotation based on furniture
        # furniture_position was already retrieved above if available
        furniture_info = None
        rotation_matrix = np.eye(3)  # Default to identity rotation
        
        if furniture_position:
            # We already have the position - just get rotation if available
            if furniture_handle and FURNITURE_LOOKUP_AVAILABLE and furniture_lookup:
                try:
                    scene_id = ep.get("scene_id", "").split('/')[-1].replace('.scene_instance.json', '')
                    furniture_handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
                    furniture_info = furniture_lookup.get_furniture_by_handle(scene_id, furniture_handle_base)
                    if furniture_info:
                        # Convert quaternion to rotation matrix
                        quat = furniture_info['rotation']  # [qx, qy, qz, qw]
                        rotation_matrix = quaternion_to_rotation_matrix(quat)
                        print(f"✓ Using furniture rotation: {furniture_info['rotation']}")
                except Exception as e:
                    print(f"⚠ Warning: Could not get furniture rotation: {e}")
        
        # Calculate object position
        obj_position = None
        
        # IMPORTANT: The name_to_receptacle mapping is what habitat uses for actual placement
        # The rigid_objs position is just an initial approximation
        # Habitat's physics will snap objects to receptacles at runtime
        # HOWEVER: The initial Y position must be reasonably close to the surface height
        # or the kinematic relationship manager won't establish the parent-child relationship!
        
        print(f"\n🔧 Position Calculation Debug:")
        print(f"  furniture_position: {furniture_position}")
        print(f"  furniture_handle: {furniture_handle}")
        print(f"  furniture: {furniture}")
        print(f"  FURNITURE_LOOKUP_AVAILABLE: {FURNITURE_LOOKUP_AVAILABLE}")
        
        if furniture_position and furniture != "floor":
            # STEP 1: Try to find existing objects on the same furniture to get actual surface height
            surface_y = None
            
            # Use furniture_handle to find existing objects (furniture_handle is already available)
            if furniture_handle:
                # Extract base handle without instance suffix
                furniture_handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
                
                print(f"🔍 Looking for existing objects on furniture handle: {furniture_handle_base}")
                
                # Look for existing objects on this furniture
                for obj_handle, obj_receptacle in ep.get('name_to_receptacle', {}).items():
                    # Check if this object is on our target furniture
                    # The receptacle format is: "FURNITURE_HANDLE_:0000|receptacle_mesh_..."
                    if obj_receptacle != "floor" and furniture_handle_base in obj_receptacle:
                        # Find this object's Y position in rigid_objs
                        for rigid_obj in ep.get('rigid_objs', []):
                            obj_config_name = rigid_obj[0].replace('.object_config.json', '')
                            if obj_config_name in obj_handle or obj_handle.startswith(obj_config_name):
                                # Found an object on this furniture! Use its Y coordinate
                                existing_y = rigid_obj[1][1][3]  # transformation matrix [1][3] is Y
                                surface_y = existing_y
                                print(f"✓ Found existing object '{obj_handle}' on furniture with Y={surface_y:.3f}m")
                                break
                    if surface_y is not None:
                        break
            
            # STEP 2: If no existing objects found, use furniture-type-based heuristics
            if surface_y is None:
                # Extract furniture type from name (e.g., "table_25" -> "table")
                furniture_type = furniture.rsplit('_', 1)[0] if '_' in furniture else furniture
                
                # Furniture type to typical surface height mapping (in meters)
                furniture_heights = {
                    'table': 0.75,
                    'desk': 0.85,
                    'counter': 0.90,
                    'kitchen_counter': 0.90,
                    'countertop': 0.90,
                    'bureau': 0.85,
                    'nightstand': 0.60,
                    'bedsidetable': 0.60,
                    'bed': 0.55,
                    'chair': 0.45,
                    'stool': 0.45,
                    'bench': 0.45,
                    'shelf': 1.20,
                    'shelves': 1.20,
                    'bookshelf': 1.20,
                    'cabinet': 0.80,
                    'chest': 0.80,
                    'chest_of_drawers': 0.80,
                    'dresser': 0.80,
                    'fridge': 1.00,
                    'refrigerator': 1.00,
                    'washer': 0.90,
                    'dryer': 0.90,
                    'washer_dryer': 0.90,
                }
                
                # Try to match furniture type
                furniture_type_lower = furniture_type.lower().replace('_', '')
                for ftype, height in furniture_heights.items():
                    if ftype.replace('_', '') in furniture_type_lower:
                        surface_y = furniture_position[1] + height
                        print(f"✓ Using furniture type '{ftype}' height heuristic: Y={surface_y:.3f}m")
                        break
                
                # Final fallback: use generic table height
                if surface_y is None:
                    surface_y = furniture_position[1] + 0.85
                    print(f"⚠ Using default table height heuristic: Y={surface_y:.3f}m")
            
            # Add small random offset to avoid exact overlap
            import random
            random_x_offset = random.uniform(-0.05, 0.05)
            random_z_offset = random.uniform(-0.05, 0.05)
            
            obj_position = (
                furniture_position[0] + random_x_offset, 
                surface_y, 
                furniture_position[2] + random_z_offset
            )
            print(f"✓ Final object position: ({obj_position[0]:.3f}, {obj_position[1]:.3f}, {obj_position[2]:.3f})")
        else:
            # Fallback: use provided position or default
            # NOTE: This position is approximate - habitat uses name_to_receptacle for actual placement
            obj_position = (position.get("x", 0.0), position.get("y", 0.5), position.get("z", 0.0))
            print(f"⚠ Using default position (habitat will adjust based on receptacle): {obj_position}")
        
        # 3. Add to rigid_objs BEFORE clutter objects
        transform_matrix = [
            [float(rotation_matrix[0, 0]), float(rotation_matrix[0, 1]), float(rotation_matrix[0, 2]), float(obj_position[0])],
            [float(rotation_matrix[1, 0]), float(rotation_matrix[1, 1]), float(rotation_matrix[1, 2]), float(obj_position[1])],
            [float(rotation_matrix[2, 0]), float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]), float(obj_position[2])],
            [0.0, 0.0, 0.0, 1.0],
        ]
        
        new_rigid_obj = [
            f"{object_id}.object_config.json",
            transform_matrix,
        ]
        ep["rigid_objs"].insert(explicit_object_count, new_rigid_obj)
        print(f"✓ Added to rigid_objs at position {explicit_object_count}")
        
        # 4. Add to name_to_receptacle BEFORE clutter objects
        recep_value = None
        
        # PRIORITY 1: If we have furniture_handle from world graph, use it directly
        if furniture_handle and furniture != "floor":
            parent_handle = furniture_handle
            parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"
            
            # Try to find receptacle mesh name from URDF
            receptacle_mesh_name = find_receptacle_mesh_name(parent_handle)
            if receptacle_mesh_name:
                recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
                print(f"✓ Using receptacle from furniture handle: {recep_value}")
            else:
                # Construct generic receptacle_mesh name
                parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
                recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
                print(f"✓ Using constructed receptacle from furniture handle: {recep_value}")
        
        # PRIORITY 2: If receptacle_handle was explicitly provided
        elif receptacle_handle and furniture != "floor":
            parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
            parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"
            
            receptacle_mesh_name = find_receptacle_mesh_name(parent_handle)
            if receptacle_mesh_name:
                recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
                print(f"✓ Using receptacle from provided handle: {recep_value}")
            else:
                parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
                recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
                print(f"✓ Using constructed receptacle from provided handle: {recep_value}")
        
        # PRIORITY 3: Only use floor if explicitly requested or no furniture info available
        if recep_value is None:
            if furniture == "floor":
                recep_value = "floor"
                print(f"✓ Using 'floor' as receptacle (explicitly requested)")
            else:
                # This shouldn't happen if world graph worked, but fallback to floor
                recep_value = "floor"
                print(f"⚠ WARNING: No furniture handle found, falling back to floor")
        
        # Insert at position equal to number of explicit objects (before clutter)
        recep_items = list(ep["name_to_receptacle"].items())
        
        # Find the next available instance suffix for this object ID
        existing_instance_nums = []
        for handle in ep["name_to_receptacle"].keys():
            if handle.startswith(f"{object_id}_:"):
                try:
                    instance_str = handle.split("_:")[-1]
                    instance_num = int(instance_str)
                    existing_instance_nums.append(instance_num)
                except (ValueError, IndexError):
                    continue
        
        # Use next available instance number
        if existing_instance_nums:
            next_instance = max(existing_instance_nums) + 1
        else:
            next_instance = 0
        
        new_handle = f"{object_id}_:{next_instance:04d}"
        recep_items.insert(explicit_object_count, (new_handle, recep_value))
        ep["name_to_receptacle"] = dict(recep_items)
        print(f"✓ Added to name_to_receptacle at position {explicit_object_count}")
        print(f"  {new_handle} -> {recep_value[:60]}...")
        
        # Store the modified episode
        cache_key = f"data/datasets/partnr_episodes/v0_0/val_mini.json.gz:{episode_id}"
        _episode_json_cache[cache_key] = ep
        
        # Clear the visualization cache so refresh will reload with new objects
        if cache_key in _episode_cache:
            del _episode_cache[cache_key]
        
        print(f"\n✓ Object '{object_class}' added successfully!")
        print(f"  Will be named: {object_class}_{explicit_object_count}")
        print(f"  Refresh visualization to see the new object.\n")
        
        # Also store in _added_objects for backward compatibility with visualization
        _added_objects[episode_id].append({
            'object_name': f"{object_class}_{explicit_object_count}",
            'object_category': object_class,
            'room': room,
            'furniture': furniture
        })
        
        return jsonify({
            "success": True,
            "object_name": f"{object_class}_{explicit_object_count}",
            "index": explicit_object_count,
            "message": f"Added {object_class}_{explicit_object_count} to {furniture} in {room}"
        })
        
    except Exception as e:
        import traceback
        print(f"ERROR adding object:")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/episode/<episode_id>/add-objects-batch', methods=['POST'])
def add_objects_batch(episode_id):
    """
    Add multiple objects from a YAML configuration.
    
    Accepts either:
    - YAML file upload (multipart/form-data with 'file' field)
    - Raw YAML in request body (Content-Type: application/x-yaml or text/yaml)
    - JSON with 'yaml_content' field (Content-Type: application/json)
    
    YAML Format:
    ```yaml
    objects:
      - object_category: "laptop"
        room: "office_1"
        furniture: "table_36"
        object_id: "Laptop_10"  # optional, will be looked up if not provided
        position: {x: 0, y: 0, z: 0}  # optional, will be calculated
      - object_category: "book"
        room: "bedroom_1"
        furniture: "table_25"
    ```
    
    Returns:
        JSON with results for each object (success/failure)
    """
    try:
        yaml_content = None
        
        # Method 1: File upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                yaml_content = file.read().decode('utf-8')
                print(f"✓ Received YAML file: {file.filename}")
        
        # Method 2: Raw YAML in body
        elif request.content_type and ('yaml' in request.content_type.lower() or request.content_type == 'text/plain'):
            yaml_content = request.data.decode('utf-8')
            print(f"✓ Received raw YAML content")
        
        # Method 3: JSON with yaml_content field
        elif request.is_json:
            data = request.get_json()
            yaml_content = data.get('yaml_content')
            if yaml_content:
                print(f"✓ Received YAML content in JSON")
        
        if not yaml_content:
            return jsonify({
                'error': 'No YAML content provided. Send as file upload, raw YAML body, or JSON with yaml_content field.'
            }), 400
        
        # Parse YAML
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            return jsonify({
                'error': f'Invalid YAML format: {str(e)}'
            }), 400
        
        if not config or 'objects' not in config:
            return jsonify({
                'error': 'YAML must contain an "objects" list'
            }), 400
        
        objects_to_add = config['objects']
        if not isinstance(objects_to_add, list):
            return jsonify({
                'error': '"objects" must be a list'
            }), 400
        
        print(f"\n{'='*60}")
        print(f"BATCH ADD: Processing {len(objects_to_add)} objects for episode {episode_id}")
        print(f"{'='*60}\n")
        
        # Process each object
        results = []
        success_count = 0
        failure_count = 0
        
        for idx, obj_spec in enumerate(objects_to_add):
            print(f"\n[{idx+1}/{len(objects_to_add)}] Processing: {obj_spec.get('object_category', 'unknown')}")
            
            # Validate required fields
            if 'object_category' not in obj_spec:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Missing required field: object_category'
                })
                failure_count += 1
                continue
            
            if 'room' not in obj_spec:
                results.append({
                    'index': idx,
                    'object_category': obj_spec['object_category'],
                    'success': False,
                    'error': 'Missing required field: room'
                })
                failure_count += 1
                continue
            
            if 'furniture' not in obj_spec:
                results.append({
                    'index': idx,
                    'object_category': obj_spec['object_category'],
                    'success': False,
                    'error': 'Missing required field: furniture'
                })
                failure_count += 1
                continue
            
            # Build request data
            obj_data = {
                'object_category': obj_spec['object_category'],
                'room': obj_spec['room'],
                'furniture': obj_spec['furniture'],
            }
            
            # Optional fields
            if 'object_id' in obj_spec:
                obj_data['object_id'] = obj_spec['object_id']
            if 'position' in obj_spec:
                obj_data['position'] = obj_spec['position']
            if 'receptacle_handle' in obj_spec:
                obj_data['receptacle_handle'] = obj_spec['receptacle_handle']
            
            # Call the add_object endpoint internally
            try:
                # Create a mock request context with the object data
                with app.test_request_context(
                    f'/api/episode/{episode_id}/add-object',
                    method='POST',
                    json=obj_data
                ):
                    response = add_object(episode_id)
                    response_data = response.get_json() if hasattr(response, 'get_json') else response[0].get_json()
                    
                    if response_data.get('success'):
                        results.append({
                            'index': idx,
                            'success': True,
                            'object_category': obj_spec['object_category'],
                            'object_name': response_data.get('object_name'),
                            'message': response_data.get('message')
                        })
                        success_count += 1
                        print(f"  ✓ Success: {response_data.get('object_name')}")
                    else:
                        results.append({
                            'index': idx,
                            'success': False,
                            'object_category': obj_spec['object_category'],
                            'error': response_data.get('error', 'Unknown error')
                        })
                        failure_count += 1
                        print(f"  ✗ Failed: {response_data.get('error', 'Unknown error')}")
            
            except Exception as e:
                import traceback
                error_msg = str(e)
                results.append({
                    'index': idx,
                    'success': False,
                    'object_category': obj_spec['object_category'],
                    'error': error_msg,
                    'traceback': traceback.format_exc()
                })
                failure_count += 1
                print(f"  ✗ Exception: {error_msg}")
        
        print(f"\n{'='*60}")
        print(f"BATCH ADD COMPLETE:")
        print(f"  Total: {len(objects_to_add)}")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failure_count}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': failure_count == 0,
            'total': len(objects_to_add),
            'success_count': success_count,
            'failure_count': failure_count,
            'results': results
        })
    
    except Exception as e:
        import traceback
        print(f"ERROR in batch add:")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def save_episode_to_gz(filepath: str, episode_data: dict):
    """Save episode data to a .json.gz file."""
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(episode_data, f, indent=2)


def wrap_episode_in_dataset_format(episode: dict) -> dict:
    """Wrap episode in PARTNR dataset format with episodes array and config field."""
    return {
        "episodes": [episode],
        "config": None
    }


@app.route('/api/episode/<episode_id>/export-config', methods=['GET'])
def export_config(episode_id):
    """
    API endpoint to export the complete modified episode.
    
    Query Parameters:
        format: 'json' (default) or 'gz' - file format
        download: 'client' (default) or 'server' or 'direct' - download method
        path: custom save path (only for server download)
        wrap: 'true' (default) or 'false' - wrap in dataset format
    
    Download Methods:
        - client: Returns JSON with metadata (client handles download via JavaScript)
        - server: Saves file to server disk and returns file path
        - direct: Sends file directly to browser for download
    """
    try:
        # Get query parameters
        format_type = request.args.get('format', 'json')  # 'json' or 'gz'
        download_method = request.args.get('download', 'client')  # 'client', 'server', or 'direct'
        custom_path = request.args.get('path', '')
        wrap_dataset = request.args.get('wrap', 'true').lower() == 'true'  # Wrap in dataset format by default
        
        # Load the modified episode from cache
        cache_key = f"data/datasets/partnr_episodes/v0_0/val_mini.json.gz:{episode_id}"
        
        if cache_key not in _episode_json_cache:
            return jsonify({
                "error": "No modifications found. Please add objects first."
            }), 404
        
        modified_ep = _episode_json_cache[cache_key]
        num_added = len(_added_objects.get(episode_id, []))
        
        # Wrap episode in PARTNR dataset format if requested
        if wrap_dataset:
            export_data = wrap_episode_in_dataset_format(modified_ep)
        else:
            export_data = modified_ep
        
        # CLIENT METHOD: Return JSON with metadata (default)
        if download_method == 'client':
            response = {
                "episode": export_data,  # Will be wrapped in dataset format if wrap=true
                "metadata": {
                    "total_objects_added": num_added,
                    "episode_id": episode_id,
                    "has_modifications": True,
                    "wrapped_in_dataset_format": wrap_dataset
                }
            }
            print(f"✓ Exported modified episode {episode_id} with {num_added} added objects (client download)")
            return jsonify(response)
        
        # SERVER METHOD: Save to server disk
        elif download_method == 'server':
            # Determine save path
            if custom_path:
                base_path = custom_path
            else:
                # Default: save in visualization/data directory
                data_dir = Path(__file__).parent / 'data'
                data_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                base_name = f"episode_{episode_id}_modified_{timestamp}"
                base_path = str(data_dir / base_name)
            
            # Save files
            json_path = base_path + '.json' if not base_path.endswith('.json') else base_path
            gz_path = json_path + '.gz'
            
            # Save uncompressed JSON
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Save compressed JSON.gz
            save_episode_to_gz(gz_path, export_data)
            
            print(f"✓ Saved modified episode {episode_id} to server:")
            print(f"  JSON: {json_path}")
            print(f"  GZ: {gz_path}")
            
            return jsonify({
                "success": True,
                "paths": {
                    "json": json_path,
                    "gz": gz_path
                },
                "metadata": {
                    "total_objects_added": num_added,
                    "episode_id": episode_id,
                    "wrapped_in_dataset_format": wrap_dataset
                }
            })
        
        # DIRECT METHOD: Send file directly to browser
        elif download_method == 'direct':
            if format_type == 'gz':
                # Create in-memory gzip file
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
                    gz_file.write(json.dumps(export_data, indent=2).encode('utf-8'))
                buffer.seek(0)
                
                print(f"✓ Sending modified episode {episode_id} as .json.gz (direct download)")
                return send_file(
                    buffer,
                    mimetype='application/gzip',
                    as_attachment=True,
                    download_name=f'episode_{episode_id}_modified.json.gz'
                )
            else:
                # Create in-memory JSON file
                buffer = io.BytesIO()
                buffer.write(json.dumps(export_data, indent=2).encode('utf-8'))
                buffer.seek(0)
                
                print(f"✓ Sending modified episode {episode_id} as .json (direct download)")
                return send_file(
                    buffer,
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=f'episode_{episode_id}_modified.json'
                )
        
        else:
            return jsonify({"error": f"Invalid download method: {download_method}"}), 400
        
    except Exception as e:
        import traceback
        print(f"ERROR exporting config:")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/episode/<episode_id>/export-data', methods=['GET'])
def export_data(episode_id):
    """Get episode data only (no metadata wrapper) for client-side export."""
    try:
        cache_key = f"data/datasets/partnr_episodes/v0_0/val_mini.json.gz:{episode_id}"
        
        if cache_key in _episode_json_cache:
            modified_ep = _episode_json_cache[cache_key]
            print(f"✓ Returned raw episode data for {episode_id}")
            return jsonify(modified_ep)
        else:
            return jsonify({"error": "No modifications found"}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/status', methods=['GET'])
def debug_status():
    """Debug endpoint to check system status."""
    return jsonify({
        "HABITAT_AVAILABLE": HABITAT_AVAILABLE,
        "SCENE_UTILS_AVAILABLE": SCENE_UTILS_AVAILABLE,
        "FURNITURE_LOOKUP_AVAILABLE": FURNITURE_LOOKUP_AVAILABLE,
        "furniture_lookup_initialized": furniture_lookup is not None,
        "furniture_lookup_dataset": str(furniture_lookup.dataset_path) if furniture_lookup else None,
        "furniture_lookup_scene_dir": str(furniture_lookup.scene_dir) if furniture_lookup else None,
        "scene_dir_exists": furniture_lookup.scene_dir.exists() if furniture_lookup else None
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Episode Visualizer App")
    print("="*80 + "\n")
    print("✓ Server is ready!")
    print("Open http://localhost:5002 in your browser")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5002)
