# Scene Object Extraction

This tool extracts comprehensive scene object information from PARTNR episodes and saves it in JSON format.

## Files

- `extract_scene_objects_simple.py` - Main extraction script (recommended)
- `extract_scene_objects.py` - Advanced extraction script using skill_runner (requires Habitat environment)

## Usage

### Extract all unique scenes:
```bash
python extract_scene_objects_simple.py
```

### Extract a specific episode:
```bash
python extract_scene_objects_simple.py --episode-id "334"
```

### Custom dataset and output directory:
```bash
python extract_scene_objects_simple.py \
  --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
  --output my_scene_objects
```

## Output Format

The script generates:

1. **Individual scene files**: `scene_objects/scene_<scene_id>.json`
   - Contains episode metadata and all entities for that scene

2. **Combined file**: `scene_objects/all_scenes_objects.json`
   - Contains all scenes indexed by environment (env_0, env_1, etc.)

### Entity Structure

Each entity (room, furniture, or object) has:

```json
{
  "id": 0,
  "category": "Object|Rooms|Furniture",
  "class_name": "cellphone",
  "prefab_name": "cellphone_0",
  "properties": ["GRABBABLE", "MOVABLE", "SURFACES", "SITTABLE", "LIEABLE"],
  "states": [],
  "room": "living_room_0",
  "on_furniture": "table_1"
}
```

### Properties

- `GRABBABLE` - Object can be picked up
- `MOVABLE` - Object can be moved
- `SURFACES` - Has surfaces (tables, counters, shelves, etc.)
- `SITTABLE` - Can be sat on (chairs, couches, beds, etc.)
- `LIEABLE` - Can lie down on (beds, couches)

## Example Output

```json
{
  "env_0": [
    {
      "id": 0,
      "category": "Object",
      "class_name": "candle",
      "prefab_name": "candle_0",
      "properties": ["GRABBABLE", "MOVABLE"],
      "states": [],
      "room": "hallway_0",
      "on_furniture": "table_5"
    },
    {
      "id": 1,
      "category": "Rooms",
      "class_name": "hallway",
      "prefab_name": "hallway_0",
      "properties": [],
      "states": []
    },
    {
      "id": 2,
      "category": "Furniture",
      "class_name": "table",
      "prefab_name": "table_5",
      "properties": ["GRABBABLE", "MOVABLE", "SURFACES"],
      "states": []
    }
  ]
}
```

## Notes

- The simple script (`extract_scene_objects_simple.py`) works directly with episode data and doesn't require running the Habitat simulator
- Objects, rooms, and furniture are extracted from the episode's `initial_state` configuration
- Object placement (room and furniture) is determined from the episode metadata
- Properties are inferred based on object category
