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

# Storage for simulator-discovered receptacles per episode
_episode_receptacle_cache = {}

# Cache for available object template IDs
_available_object_template_ids = None

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


def _base_handle(handle: str) -> str:
    """Normalize handle by removing instance suffix (e.g. _:\\d+)."""
    return handle.split('_:')[0] if '_:' in handle else handle


def _select_sim_receptacle_for_parent(receptacle_data: Dict, parent_handle: str) -> Optional[str]:
    """Select a simulator-discovered receptacle unique_name for a parent handle."""
    if not receptacle_data or not parent_handle:
        return None

    parent_handle = parent_handle.split('|')[0]
    parent_base = _base_handle(parent_handle)

    by_parent_handle = receptacle_data.get('by_parent_handle', {})
    by_parent_base = receptacle_data.get('by_parent_base', {})

    # Try exact and normalized variants against parent-handle map
    parent_candidates = [
        parent_handle,
        parent_base,
        f"{parent_base}_:0000",
    ]

    for key in parent_candidates:
        recs = by_parent_handle.get(key, [])
        if recs:
            return recs[0]

    # Try base-handle map
    recs = by_parent_base.get(parent_base, [])
    if recs:
        return recs[0]

    return None


def load_sim_receptacles(episode_id: str, dataset_path: str = None) -> Dict:
    """Load simulator-discovered receptacles for an episode via helper script."""
    if dataset_path is None:
        dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"

    cache_key = f"{dataset_path}:{episode_id}"
    if cache_key in _episode_receptacle_cache:
        return _episode_receptacle_cache[cache_key]

    try:
        project_root = Path(__file__).parent.parent
        script_path = project_root / "visualization" / "episode_info" / "get_episode_receptacles.py"
        output_file = project_root / "visualization" / f"temp_episode_{episode_id}_receptacles.json"

        cmd = [
            sys.executable,
            str(script_path),
            f"+episode_id={episode_id}",
            f"+dataset_path={dataset_path}",
            f"+output_file={output_file}",
            "hydra.run.dir=.",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            print(f"⚠ Simulator receptacle discovery failed for episode {episode_id}")
            print(result.stderr)
            return {}

        if not output_file.exists():
            print(f"⚠ Simulator receptacle output file missing: {output_file}")
            return {}

        with open(output_file, 'r') as f:
            payload = json.load(f)

        try:
            output_file.unlink()
        except Exception:
            pass

        receptacle_data = payload.get('receptacles', {})
        _episode_receptacle_cache[cache_key] = receptacle_data
        return receptacle_data
    except Exception as e:
        print(f"⚠ Error loading simulator receptacles for episode {episode_id}: {e}")
        return {}


def repair_episode_receptacle_mappings(episode: Dict, episode_id: str, dataset_path: str = None) -> Dict:
    """Repair invalid name_to_receptacle mappings using simulator-discovered active receptacles."""
    if dataset_path is None:
        dataset_path = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz"

    mapping = episode.get("name_to_receptacle", {})
    if not isinstance(mapping, dict) or not mapping:
        return {
            "applied": False,
            "reason": "no_mapping",
            "total": 0,
            "repaired": 0,
            "invalid_before": 0,
            "invalid_after": 0,
        }

    sim_receptacle_data = load_sim_receptacles(episode_id, dataset_path)
    active_receptacles = set(sim_receptacle_data.get("all", []))

    if not active_receptacles:
        return {
            "applied": False,
            "reason": "no_sim_receptacles",
            "total": len(mapping),
            "repaired": 0,
            "invalid_before": 0,
            "invalid_after": 0,
        }

    repaired_mapping = {}
    repaired_count = 0
    invalid_before = 0
    floor_normalized = 0

    for obj_handle, recep_value in mapping.items():
        # Normalize any floor-encoded receptacle variants to literal 'floor'
        # e.g. floor_:0000|receptacle_mesh_floor.0000
        if recep_value == "floor" or str(recep_value).startswith("floor"):
            repaired_mapping[obj_handle] = "floor"
            if recep_value != "floor":
                floor_normalized += 1
            continue

        if recep_value in active_receptacles:
            repaired_mapping[obj_handle] = recep_value
            continue

        invalid_before += 1
        parent_handle = recep_value.split('|')[0] if '|' in recep_value else recep_value
        replacement = _select_sim_receptacle_for_parent(sim_receptacle_data, parent_handle)

        if replacement:
            repaired_mapping[obj_handle] = replacement
            repaired_count += 1
        else:
            repaired_mapping[obj_handle] = recep_value

    invalid_after = sum(
        1
        for value in repaired_mapping.values()
        if value != "floor" and value not in active_receptacles
    )

    episode["name_to_receptacle"] = repaired_mapping
    return {
        "applied": True,
        "reason": "ok",
        "total": len(mapping),
        "repaired": repaired_count,
        "floor_normalized": floor_normalized,
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "active_receptacles": len(active_receptacles),
    }


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


def build_object_category_map() -> Dict[str, str]:
    """Build object-id to category mapping from loaded metadata CSVs."""
    load_object_database()
    category_map = {}
    for row in object_database:
        obj_id = (row.get('id') or '').strip()
        category = (row.get('category') or '').strip()
        if obj_id and category and obj_id not in category_map:
            category_map[obj_id] = category
    return category_map


def normalize_object_id(raw_object_id: str) -> str:
    """Normalize object id by removing path/config/suffix tokens."""
    obj_id = raw_object_id.split('/')[-1]
    obj_id = obj_id.replace('.object_config.json', '')
    if '_:' in obj_id:
        obj_id = obj_id.split('_:')[0]
    return obj_id


def load_available_object_template_ids() -> set:
    """Load all available object template IDs from local object config files."""
    global _available_object_template_ids

    if _available_object_template_ids is not None:
        return _available_object_template_ids

    project_root = Path(__file__).parent.parent
    object_dirs = [
        project_root / "data" / "hssd-hab" / "objects",
        project_root / "data" / "objects",
        project_root / "data" / "objects_ovmm",
    ]

    template_ids = set()
    for obj_dir in object_dirs:
        if not obj_dir.exists():
            continue
        for config_file in obj_dir.rglob("*.object_config.json"):
            template_ids.add(config_file.stem.replace('.object_config', ''))

    _available_object_template_ids = template_ids
    return _available_object_template_ids


def _split_handle_instance(handle: str) -> tuple:
    """Split handle into base and numeric instance suffix."""
    if '_:' not in handle:
        return handle, 0
    base, suffix = handle.split('_:', 1)
    try:
        return base, int(suffix)
    except ValueError:
        return base, 0


def repair_episode_object_templates(episode: Dict) -> Dict:
    """Repair missing object template handles and corresponding mapping keys."""
    available_templates = load_available_object_template_ids()

    # Known problematic categories from visualizer input -> valid template IDs
    fallback_template_map = {
        "alarm_clock": "Alarm_Clock_4",
        "toothbrush": "Sonicare_2_Series_Toothbrush_Plaque_Control",
        "water_bottle": "ce33c1228cfca3da78e22645019258d7a92af3a9",
    }

    rigid_objs = episode.get('rigid_objs', [])
    old_to_new_base = {}
    rigid_repaired = 0

    for rigid_obj in rigid_objs:
        if not rigid_obj:
            continue
        template_path = rigid_obj[0]
        base = template_path.replace('.object_config.json', '')
        if base in available_templates:
            continue

        replacement = fallback_template_map.get(base)
        if replacement and replacement in available_templates:
            rigid_obj[0] = f"{replacement}.object_config.json"
            old_to_new_base[base] = replacement
            rigid_repaired += 1

    # Repair name_to_receptacle keys if object base names changed
    key_repaired = 0
    mapping = episode.get('name_to_receptacle', {})
    if isinstance(mapping, dict) and old_to_new_base:
        remapped = {}

        def next_instance(base_name: str) -> int:
            prefix = f"{base_name}_:"
            nums = []
            for key in list(mapping.keys()) + list(remapped.keys()):
                if key.startswith(prefix):
                    _, inst = _split_handle_instance(key)
                    nums.append(inst)
            return max(nums) + 1 if nums else 0

        for key, value in mapping.items():
            base, inst = _split_handle_instance(key)
            if base not in old_to_new_base:
                remapped[key] = value
                continue

            new_base = old_to_new_base[base]
            new_key = f"{new_base}_:{inst:04d}"
            if new_key in remapped:
                new_key = f"{new_base}_:{next_instance(new_base):04d}"

            remapped[new_key] = value
            key_repaired += 1

        episode['name_to_receptacle'] = remapped

    return {
        "applied": rigid_repaired > 0 or key_repaired > 0,
        "rigid_templates_repaired": rigid_repaired,
        "name_to_receptacle_keys_repaired": key_repaired,
        "replacements": old_to_new_base,
    }


def get_skill_runner_object_nodes_from_episode(episode: Dict) -> List[Dict]:
    """Compute skill-runner-style object node names from episode rigid object ordering."""
    category_map = build_object_category_map()
    object_nodes = []

    for idx, rigid_obj in enumerate(episode.get('rigid_objs', [])):
        raw_id = rigid_obj[0] if rigid_obj else ''
        normalized_id = normalize_object_id(raw_id)
        category = category_map.get(normalized_id, normalized_id.lower())

        object_nodes.append({
            'node_name': f"{category}_{idx}",
            'category': category,
            'object_id': normalized_id,
            'rigid_obj_index': idx,
        })

    return object_nodes


def _build_skill_command_alias_map(object_nodes: List[Dict]) -> Dict[str, str]:
    """Build aliases from category-index references to actual graph node names."""
    alias_map: Dict[str, str] = {}
    by_category: Dict[str, List[Dict]] = {}

    for node in object_nodes:
        node_name = node['node_name']
        alias_map[node_name] = node_name

        if '_' not in node_name:
            continue
        category, idx_str = node_name.rsplit('_', 1)
        if idx_str.isdigit():
            by_category.setdefault(category, []).append({
                'index': int(idx_str),
                'node_name': node_name,
            })

    for category, entries in by_category.items():
        ordered = sorted(entries, key=lambda x: x['index'])

        # one-based alias expected in many handwritten command files
        for one_based, entry in enumerate(ordered, start=1):
            alias_map[f"{category}_{one_based}"] = entry['node_name']

        # zero-based alias fallback
        for zero_based, entry in enumerate(ordered):
            alias_map[f"{category}_{zero_based}"] = entry['node_name']

    return alias_map


def _remap_skill_command_line(line: str, alias_map: Dict[str, str]) -> str:
    """Remap object references in one skill command line."""
    stripped = line.strip()
    if not stripped:
        return line

    parts = stripped.split()
    if len(parts) < 3:
        return line

    command = parts[0]

    def _map_token(token: str) -> str:
        if token in {"None", "none", "null", "NULL", ""}:
            return token
        return alias_map.get(token, token)

    if command in {"Pick", "Navigate", "Open", "Close"}:
        old_entity = parts[2]
        new_entity = _map_token(old_entity)
        if new_entity != old_entity:
            return line.replace(old_entity, new_entity, 1)
        return line

    if command == "Place":
        payload = " ".join(parts[2:])
        chunks = payload.split(',')
        if len(chunks) >= 5:
            chunks[0] = _map_token(chunks[0])
            chunks[2] = _map_token(chunks[2])
            chunks[4] = _map_token(chunks[4])
            new_payload = ','.join(chunks)
            prefix = " ".join(parts[:2])
            suffix = "\n" if line.endswith("\n") else ""
            return f"{prefix} {new_payload}{suffix}"

    return line


def create_mapped_skill_runner_commands_file(
    episode: Dict,
    input_commands_file: Path,
    output_commands_file: Path,
) -> Dict:
    """Create mapped skill_runner command file for current episode object names."""
    object_nodes = get_skill_runner_object_nodes_from_episode(episode)
    alias_map = _build_skill_command_alias_map(object_nodes)

    if not input_commands_file.exists():
        return {
            'created': False,
            'reason': f'input_not_found:{input_commands_file}',
            'mapped_lines': 0,
            'output_file': str(output_commands_file),
        }

    input_lines = input_commands_file.read_text(encoding='utf-8').splitlines(keepends=True)
    output_lines = []
    mapped_lines = 0

    for line in input_lines:
        new_line = _remap_skill_command_line(line, alias_map)
        if new_line != line:
            mapped_lines += 1
        output_lines.append(new_line)

    output_commands_file.parent.mkdir(parents=True, exist_ok=True)
    output_commands_file.write_text(''.join(output_lines), encoding='utf-8')

    return {
        'created': True,
        'reason': 'ok',
        'mapped_lines': mapped_lines,
        'output_file': str(output_commands_file),
        'input_file': str(input_commands_file),
        'object_nodes': len(object_nodes),
        'alias_count': len(alias_map),
    }


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

        # Discover active simulator receptacles for this episode (source of truth)
        dataset_path_for_episode = request.args.get('dataset', 'data/datasets/partnr_episodes/v0_0/val_mini.json.gz')
        sim_receptacle_data = load_sim_receptacles(episode_id, dataset_path_for_episode)
        if sim_receptacle_data:
            print(f"✓ Loaded {sim_receptacle_data.get('count', 0)} simulator-discovered receptacles")
        else:
            print("⚠ No simulator receptacles available; falling back to heuristic receptacle construction")
        
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

            # Prefer simulator-discovered receptacle unique_name for this parent
            discovered_receptacle = _select_sim_receptacle_for_parent(sim_receptacle_data, parent_handle)
            if discovered_receptacle:
                recep_value = discovered_receptacle
                print(f"✓ Using simulator-discovered receptacle: {recep_value}")
            
            # Fallback: legacy heuristic receptacle construction
            if recep_value is None:
                receptacle_mesh_name = find_receptacle_mesh_name(parent_handle)
                if receptacle_mesh_name:
                    recep_value = f"{parent_handle_with_suffix}|{receptacle_mesh_name}.0000"
                    print(f"✓ Using receptacle from furniture handle: {recep_value}")
                else:
                    parent_handle_base = parent_handle.split('_:')[0] if '_:' in parent_handle else parent_handle
                    recep_value = f"{parent_handle_with_suffix}|receptacle_mesh_{parent_handle_base}.0000"
                    print(f"✓ Using constructed receptacle from furniture handle: {recep_value}")
        
        # PRIORITY 2: If receptacle_handle was explicitly provided
        elif receptacle_handle and furniture != "floor":
            parent_handle = receptacle_handle.split('|')[0] if '|' in receptacle_handle else receptacle_handle
            parent_handle_with_suffix = parent_handle if "_:" in parent_handle else f"{parent_handle}_:0000"

            # Prefer simulator-discovered receptacle unique_name for this parent
            discovered_receptacle = _select_sim_receptacle_for_parent(sim_receptacle_data, parent_handle)
            if discovered_receptacle:
                recep_value = discovered_receptacle
                print(f"✓ Using simulator-discovered receptacle: {recep_value}")
            
            # Fallback: legacy heuristic receptacle construction
            if recep_value is None:
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
        repair_receptacles: 'true' (default) or 'false' - auto-repair mappings to active simulator receptacles
        repair_templates: 'true' (default) or 'false' - auto-repair missing object template handles
        map_skill_commands: 'true' (default) or 'false' - auto-generate mapped skill_runner commands for current object names
        skill_commands_file: input command file path (default: skill_runner_commands.txt)
    
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
        repair_receptacles = request.args.get('repair_receptacles', 'true').lower() == 'true'
        repair_templates = request.args.get('repair_templates', 'true').lower() == 'true'
        map_skill_commands = request.args.get('map_skill_commands', 'true').lower() == 'true'
        skill_commands_file = request.args.get('skill_commands_file', 'skill_runner_commands.txt')
        
        # Load the modified episode from cache
        cache_key = f"data/datasets/partnr_episodes/v0_0/val_mini.json.gz:{episode_id}"
        
        if cache_key not in _episode_json_cache:
            return jsonify({
                "error": "No modifications found. Please add objects first."
            }), 404
        
        modified_ep = _episode_json_cache[cache_key]
        num_added = len(_added_objects.get(episode_id, []))

        # Optionally repair invalid receptacle mappings before export
        template_repair_summary = {
            "applied": False,
            "rigid_templates_repaired": 0,
            "name_to_receptacle_keys_repaired": 0,
            "replacements": {},
        }
        repair_summary = {
            "applied": False,
            "reason": "disabled",
            "total": 0,
            "repaired": 0,
            "invalid_before": 0,
            "invalid_after": 0,
        }
        if repair_templates or repair_receptacles:
            modified_ep = json.loads(json.dumps(modified_ep))

        if repair_templates:
            template_repair_summary = repair_episode_object_templates(modified_ep)
            print(
                "✓ Template repair summary: "
                f"templates={template_repair_summary.get('rigid_templates_repaired', 0)}, "
                f"keys={template_repair_summary.get('name_to_receptacle_keys_repaired', 0)}"
            )

        if repair_receptacles:
            repair_summary = repair_episode_receptacle_mappings(
                modified_ep,
                episode_id,
                dataset_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
            )
            print(
                "✓ Receptacle repair summary: "
                f"repaired={repair_summary.get('repaired', 0)}, "
                f"invalid_before={repair_summary.get('invalid_before', 0)}, "
                f"invalid_after={repair_summary.get('invalid_after', 0)}"
            )
        
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
                    "wrapped_in_dataset_format": wrap_dataset,
                    "template_repair": template_repair_summary,
                    "receptacle_repair": repair_summary,
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

            # Optionally create mapped skill command file for this export
            commands_map_summary = {
                'created': False,
                'reason': 'disabled',
                'mapped_lines': 0,
            }
            mapped_commands_path = None
            if map_skill_commands:
                project_root = Path(__file__).parent.parent
                input_commands_path = Path(skill_commands_file)
                if not input_commands_path.is_absolute():
                    input_commands_path = project_root / input_commands_path

                mapped_commands_path = Path(json_path).with_suffix('').as_posix() + '_skill_runner_commands_mapped.txt'
                commands_map_summary = create_mapped_skill_runner_commands_file(
                    modified_ep,
                    input_commands_file=input_commands_path,
                    output_commands_file=Path(mapped_commands_path),
                )
                print(
                    "✓ Skill command mapping summary: "
                    f"created={commands_map_summary.get('created')}, "
                    f"mapped_lines={commands_map_summary.get('mapped_lines', 0)}"
                )
            
            print(f"✓ Saved modified episode {episode_id} to server:")
            print(f"  JSON: {json_path}")
            print(f"  GZ: {gz_path}")
            if mapped_commands_path:
                print(f"  Mapped commands: {mapped_commands_path}")
            
            response_paths = {
                "json": json_path,
                "gz": gz_path,
            }
            if mapped_commands_path:
                response_paths["skill_commands_mapped"] = mapped_commands_path

            return jsonify({
                "success": True,
                "paths": response_paths,
                "metadata": {
                    "total_objects_added": num_added,
                    "episode_id": episode_id,
                    "wrapped_in_dataset_format": wrap_dataset,
                    "template_repair": template_repair_summary,
                    "receptacle_repair": repair_summary,
                    "skill_command_mapping": commands_map_summary,
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


@app.route('/api/episode/<episode_id>/skill-runner-object-names', methods=['GET'])
def get_skill_runner_object_names(episode_id):
    """Return valid object node names for skill runner (e.g. soap_dispenser_25)."""
    try:
        dataset_path = request.args.get('dataset', 'data/datasets/partnr_episodes/v0_0/val_mini.json.gz')
        cache_key = f"{dataset_path}:{episode_id}"

        # Prefer modified in-memory episode if available, else load original episode
        if cache_key in _episode_json_cache:
            episode = _episode_json_cache[cache_key]
            source = 'modified_cache'
        else:
            episode = load_full_episode_json(episode_id, dataset_path=dataset_path)
            source = 'dataset'

        object_nodes = get_skill_runner_object_nodes_from_episode(episode)

        # Optional substring filter for quick lookup
        query = request.args.get('query', '').strip().lower()
        if query:
            object_nodes = [
                node for node in object_nodes
                if query in node['node_name'].lower()
                or query in node['category'].lower()
                or query in node['object_id'].lower()
            ]

        return jsonify({
            'episode_id': str(episode_id),
            'source': source,
            'count': len(object_nodes),
            'objects': object_nodes,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


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
