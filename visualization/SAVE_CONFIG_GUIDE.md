# 💾 Save Configuration Guide

## Overview

The Episode Visualizer now supports saving added objects to a configuration file in the same format as episode initial state configs. This allows you to:

- Export your object additions as JSON config files
- Share configurations with others
- Re-use object placement patterns
- Document object arrangements

---

## How It Works

### 1. **Add Objects to Episode**
Use the sidebar form to add objects to different rooms and furniture:
- Select object category (e.g., book, laptop, vase)
- Choose room and furniture
- Click "Add Object"

### 2. **Save Configuration**
Click the **"Download Config File"** button in the sidebar's "Save Configuration" section.

This will:
- Export all added objects to a JSON file
- Download as: `episode_<ID>_added_objects_config.json`
- Show confirmation with object count

---

## Configuration Format

The exported JSON follows the episode `initial_state` format:

```json
{
  "episode_id": "100_modified",
  "added_objects_config": {
    "initial_state": [
      {
        "number": 1,
        "object_classes": ["book"],
        "allowed_regions": ["office_1"],
        "furniture_names": ["table_36"]
      },
      {
        "number": 1,
        "object_classes": ["laptop"],
        "allowed_regions": ["office_1"],
        "furniture_names": ["table_36"]
      }
    ]
  },
  "metadata": {
    "original_episode_id": "100",
    "total_objects_added": 2,
    "timestamp": "{\"date\": \"2026-02-26\"}",
    "objects_detail": [
      {
        "object_name": "book_0",
        "object_category": "book",
        "room": "office_1",
        "furniture": "table_36"
      },
      {
        "object_name": "laptop_1",
        "object_category": "laptop",
        "room": "office_1",
        "furniture": "table_36"
      }
    ]
  }
}
```

### Structure Breakdown

#### `episode_id`
- Format: `{original_id}_modified`
- Indicates this is a modified version of an episode

#### `added_objects_config.initial_state`
- **Array of object placement configs**
- Each entry specifies:
  - `number`: How many objects of this type
  - `object_classes`: Array with object category
  - `allowed_regions`: Array with room name
  - `furniture_names`: Array with furniture name

#### `metadata`
- `original_episode_id`: The base episode ID
- `total_objects_added`: Count of added objects
- `timestamp`: When the config was created
- `objects_detail`: Full list of individual objects with their placements

---

## Use Cases

### 1. **Testing Object Placements**
Add objects through UI, export config, use in episode generation:
```bash
# Add objects via UI, then download config
# Use the config in your episode generator
```

### 2. **Sharing Configurations**
Export and share object placement patterns:
```bash
# Person A: Add objects and export config
# Person B: Use config as template for similar episodes
```

### 3. **Documentation**
Keep records of object arrangements:
```bash
# Export configs for different scenarios
# Store in version control
# Reference for future episodes
```

### 4. **Batch Object Addition**
Create template configs for common patterns:
```json
{
  "initial_state": [
    {
      "number": 3,
      "object_classes": ["book"],
      "allowed_regions": ["office_1"],
      "furniture_names": ["table_36"]
    }
  ]
}
```

---

## API Usage

### Export Configuration Endpoint

```bash
GET /api/episode/<episode_id>/export-config
```

**Example:**
```bash
curl http://localhost:5002/api/episode/100/export-config > config.json
```

**Response:**
```json
{
  "episode_id": "100_modified",
  "added_objects_config": { ... },
  "metadata": { ... }
}
```

**Error Cases:**
- No objects added: Returns 400 with error message
- Invalid episode: Returns appropriate error code

---

## Workflow Example

### Step-by-Step: Adding and Saving Objects

1. **Start the app:**
   ```bash
   cd /home/nadira/partnr-planner/visualization
   conda activate habitat
   python episode_visualizer_app.py
   ```

2. **Load Episode 100** in browser (http://localhost:5002)

3. **Add objects via UI:**
   - Search "book" → Select "book"
   - Room: "office_1"
   - Furniture: "table_36"
   - Click "Add Object"
   
   - Search "laptop" → Select "laptop"
   - Room: "office_1"
   - Furniture: "table_36"
   - Click "Add Object"

4. **Click "Download Config File"**
   - File downloads: `episode_100_added_objects_config.json`
   - Contains both objects in initial_state format

5. **Use the config:**
   ```bash
   # View the exported config
   cat episode_100_added_objects_config.json
   
   # Use in episode generation
   # Copy relevant sections to your episode generator configs
   ```

---

## Integration with Episode Generation

The exported `initial_state` format matches the episode generator input:

```python
# Example: Using the exported config in episode generation
import json

# Load exported config
with open('episode_100_added_objects_config.json') as f:
    config = json.load(f)

# Extract initial_state
initial_state = config['added_objects_config']['initial_state']

# Use in episode generator
episode_config = {
    "scene_id": "102817140",
    "episode_id": "new_episode_001",
    "initial_state": initial_state  # Add your exported objects
}
```

---

## Practical Examples

### Example 1: Office Setup
```json
{
  "initial_state": [
    {"number": 1, "object_classes": ["laptop"], "allowed_regions": ["office_1"], "furniture_names": ["table_36"]},
    {"number": 2, "object_classes": ["book"], "allowed_regions": ["office_1"], "furniture_names": ["table_36"]},
    {"number": 1, "object_classes": ["monitor_stand"], "allowed_regions": ["office_1"], "furniture_names": ["table_36"]},
    {"number": 1, "object_classes": ["lamp"], "allowed_regions": ["office_1"], "furniture_names": ["table_8"]}
  ]
}
```

### Example 2: Living Room Decoration
```json
{
  "initial_state": [
    {"number": 1, "object_classes": ["vase"], "allowed_regions": ["living_room_1"], "furniture_names": ["table_45"]},
    {"number": 2, "object_classes": ["cushion"], "allowed_regions": ["living_room_1"], "furniture_names": ["couch_5"]},
    {"number": 1, "object_classes": ["book"], "allowed_regions": ["living_room_1"], "furniture_names": ["table_45"]}
  ]
}
```

### Example 3: Kitchen Items
```json
{
  "initial_state": [
    {"number": 2, "object_classes": ["bottle"], "allowed_regions": ["kitchen_1"], "furniture_names": ["counter_32"]},
    {"number": 1, "object_classes": ["bowl"], "allowed_regions": ["kitchen_1"], "furniture_names": ["table_15"]},
    {"number": 1, "object_classes": ["canister"], "allowed_regions": ["kitchen_1"], "furniture_names": ["counter_32"]}
  ]
}
```

---

## File Management

### Default Save Location
When using the UI:
- Browser downloads to your default Downloads folder
- Filename: `episode_<ID>_added_objects_config.json`

### Organizing Config Files
Suggested structure:
```
configs/
├── office_setups/
│   ├── episode_100_office_basic.json
│   └── episode_100_office_full.json
├── living_room_setups/
│   └── episode_100_living_decorated.json
└── kitchen_setups/
    └── episode_100_kitchen_stocked.json
```

---

## Tips & Best Practices

1. **Descriptive Filenames**: Rename downloaded files to describe the setup
   ```bash
   mv episode_100_added_objects_config.json episode_100_office_workspace.json
   ```

2. **Version Control**: Store configs in git for tracking changes
   ```bash
   git add configs/*.json
   git commit -m "Add office workspace configuration"
   ```

3. **Validation**: Check exported configs before using in production
   ```bash
   python -m json.tool episode_100_added_objects_config.json
   ```

4. **Backup**: Keep copies of successful configurations
   ```bash
   cp episode_100_added_objects_config.json backups/
   ```

5. **Documentation**: Add comments in metadata for future reference
   ```json
   "metadata": {
     "description": "Standard office setup with laptop and books",
     "use_case": "Testing workspace scenarios"
   }
   ```

---

## Troubleshooting

**Problem**: "No objects to export" error
- **Solution**: Add at least one object before exporting

**Problem**: Download doesn't start
- **Solution**: Check browser pop-up blocker settings

**Problem**: Config file is empty
- **Solution**: Ensure objects were successfully added (check for success messages)

**Problem**: Can't find downloaded file
- **Solution**: Check browser's Downloads folder or download history

---

## API Testing

Test the export functionality via command line:

```bash
# Add some objects
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "book", "room": "office_1", "furniture": "table_36"}'

curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "laptop", "room": "office_1", "furniture": "table_36"}'

# Export config
curl http://localhost:5002/api/episode/100/export-config | python -m json.tool

# Save to file
curl http://localhost:5002/api/episode/100/export-config > my_config.json
```

---

## Summary

✅ **Features:**
- Export added objects as episode config format
- Download JSON files directly from browser
- Compatible with episode generator input format
- Includes metadata for tracking and documentation

📁 **Output:**
- JSON file with `initial_state` array
- Metadata with timestamps and object details
- Ready to use in episode generation workflows

🎯 **Use For:**
- Testing object placements
- Creating reusable templates
- Sharing configurations
- Documentation and version control

**Server:** http://localhost:5002
**Button Location:** Sidebar → Save Configuration → Download Config File
