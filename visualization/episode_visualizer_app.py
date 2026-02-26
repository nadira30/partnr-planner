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
import subprocess
import csv
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from typing import Dict, List
from collections import defaultdict

app = Flask(__name__)

# Cache for episode data
_episode_cache = {}

# Storage for added objects per episode
_added_objects = defaultdict(list)


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
    """API endpoint to add an object to an episode."""
    try:
        data = request.get_json()
        
        object_category = data.get('object_category')
        room = data.get('room')
        furniture = data.get('furniture')
        
        if not all([object_category, room, furniture]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Count existing objects of this category
        existing_count = sum(
            1 for obj in _added_objects.get(episode_id, [])
            if obj['object_category'] == object_category
        )
        
        # Also check original episode data
        try:
            episode_data = get_episode_data(episode_id)
            existing_count += sum(
                1 for obj_name in episode_data.get('object_locations', {}).keys()
                if obj_name.startswith(object_category + '_')
            )
        except:
            pass
        
        # Create object name with counter
        object_name = f"{object_category}_{existing_count}"
        
        # Store the added object
        _added_objects[episode_id].append({
            'object_name': object_name,
            'object_category': object_category,
            'room': room,
            'furniture': furniture
        })
        
        print(f"✓ Added {object_name} to {furniture} in {room} (Episode {episode_id})")
        
        return jsonify({
            'success': True,
            'object_name': object_name,
            'message': f'Added {object_name} to {furniture} in {room}'
        })
        
    except Exception as e:
        import traceback
        print(f"ERROR adding object:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/episode/<episode_id>/export-config', methods=['GET'])
def export_config(episode_id):
    """
    API endpoint to export added objects as a NEW separate config file.
    
    IMPORTANT: This creates a NEW file with ONLY the added objects.
    It does NOT modify the original episode config file.
    The output is an ADDITION that can be used alongside the original.
    """
    try:
        added_objs = _added_objects.get(episode_id, [])
        
        if not added_objs:
            return jsonify({'error': 'No objects added to export'}), 400
        
        # Group objects by room and furniture
        config_items = []
        
        # Create a dict to group by (room, furniture, object_category)
        grouped = {}
        for obj in added_objs:
            key = (obj['room'], obj['furniture'], obj['object_category'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(obj)
        
        # Convert to initial_state format
        for (room, furniture, obj_category), objs in grouped.items():
            config_items.append({
                "number": len(objs),
                "object_classes": [obj_category],
                "allowed_regions": [room],
                "furniture_names": [furniture]
            })
        
        # Create the full config structure for NEW objects ONLY
        config = {
            "_comment": "This is a NEW file containing ONLY the added objects. It does NOT modify the original episode config.",
            "file_type": "added_objects_only",
            "original_episode_id": episode_id,
            "new_episode_id": f"{episode_id}_with_additions",
            "added_objects": {
                "initial_state": config_items
            },
            "metadata": {
                "description": "Contains ONLY objects added via the visualizer UI",
                "original_episode_config": f"See original episode {episode_id} config for base objects",
                "total_objects_added": len(added_objs),
                "timestamp": "2026-02-26",
                "usage": "Append these objects to the original episode's initial_state array",
                "objects_detail": added_objs
            }
        }
        
        print(f"✓ Exported NEW config file for episode {episode_id} with {len(added_objs)} added objects")
        print(f"  (Original episode config remains unchanged)")
        
        return jsonify(config)
        
    except Exception as e:
        import traceback
        print(f"ERROR exporting config:")
        print(traceback.format_exc())
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
