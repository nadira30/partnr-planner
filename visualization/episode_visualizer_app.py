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
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from typing import Dict, List, Optional
from collections import defaultdict

# Try to import habitat and scene utilities
HABITAT_AVAILABLE = False
SCENE_UTILS_AVAILABLE = False

try:
    import habitat
    from habitat_llm.utils.sim import find_receptacles
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
        receptacle_handle = data.get("receptacle_handle")
        preposition = data.get("preposition", "on")
        position = data.get("position", {"x": 0, "y": 0.5, "z": 0})
        
        if not all([object_class, room, furniture]):
            return jsonify({"error": "Missing required fields"}), 400
        
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
        scene_id = ep.get("scene_id", "")
        
        # Get furniture position and rotation from scene file
        furniture_info = None
        rotation_matrix = np.eye(3)  # Default to identity rotation
        
        if receptacle_handle and furniture != "floor" and SCENE_UTILS_AVAILABLE and scene_id:
            try:
                # Extract the parent furniture handle from receptacle_handle
                furniture_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
                furniture_handle_base = furniture_handle.split('_:')[0] if '_:' in furniture_handle else furniture_handle
                
                print(f"Getting furniture position/rotation from scene file using handle: {furniture_handle_base}")
                furniture_info = scene_utils.get_object_position(scene_id, furniture_handle_base)
                if furniture_info:
                    print(f"✓ Found furniture in scene file:")
                    print(f"  Template: {furniture_info['template_name']}")
                    print(f"  Position: {furniture_info['position']}")
                    print(f"  Rotation (quaternion): {furniture_info['rotation']}")
                    
                    # Convert quaternion to rotation matrix
                    rotation_matrix = quaternion_to_rotation_matrix(furniture_info['rotation'])
                    print(f"  Rotation matrix extracted")
                else:
                    print(f"⚠ Could not find furniture with handle '{furniture_handle_base}' in scene file")
            except Exception as e:
                print(f"⚠ Warning: Could not get furniture info from scene file: {e}")
        
        # Calculate object position
        obj_position = None
        if furniture_info and furniture != "floor":
            # Use furniture position with height offset for top surface
            base_pos = furniture_info['position']
            if abs(base_pos[1]) < 0.1:  # Ground level furniture
                surface_y = base_pos[1]
            else:  # Already elevated
                surface_y = base_pos[1]
            obj_position = (base_pos[0], surface_y, base_pos[2])
            print(f"Using furniture position with surface offset: {obj_position}")
        else:
            obj_position = (position.get("x", 0.0), position.get("y", 0.5), position.get("z", 0.0))
            print(f"Using default/provided position: {obj_position}")
        
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
        if receptacle_handle and furniture != "floor" and HABITAT_AVAILABLE:
            # Try to get receptacle from simulator first (most reliable)
            try:
                temp_sim = create_sim_for_episode(ep)
                if temp_sim is not None:
                    receptacles = find_receptacles(temp_sim, filter_receptacles=False)
                    
                    parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
                    
                    # Find matching receptacles
                    matching_receptacles = []
                    for rec in receptacles:
                        if rec.parent_object_handle == parent_handle or parent_handle in rec.parent_object_handle:
                            matching_receptacles.append(rec)
                    
                    if matching_receptacles:
                        rec = matching_receptacles[0]
                        parent_handle_actual = rec.parent_object_handle
                        if "_:" not in parent_handle_actual:
                            parent_handle_actual = f"{parent_handle_actual}_:0000"
                        recep_value = f"{parent_handle_actual}|{rec.name}"
                        print(f"✓ Validated receptacle exists in simulator: {recep_value}")
                    else:
                        print(f"⚠ Receptacle not found in simulator for handle: {parent_handle}")
                    
                    temp_sim.close()
            except Exception as e:
                print(f"⚠ Could not validate receptacle: {e}")
        
        # If simulator validation failed, try URDF fallback
        if recep_value is None and receptacle_handle and furniture != "floor":
            parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
            parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"
            
            receptacle_mesh_name = find_receptacle_mesh_name(parent_handle)
            if receptacle_mesh_name:
                recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
                print(f"⚠ Using receptacle from URDF (not validated): {recep_value}")
            else:
                # Last resort: construct generic receptacle_mesh name
                parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
                recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
                print(f"⚠ WARNING: Could not validate receptacle, using constructed name: {recep_value}")
        
        # Only use floor if explicitly requested
        if recep_value is None:
            recep_value = "floor"
            print(f"✓ Using 'floor' as receptacle")
        
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


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Episode Visualizer App")
    print("="*80 + "\n")
    print("✓ Server is ready!")
    print("Open http://localhost:5002 in your browser")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5002)
