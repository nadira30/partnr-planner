# Episode Entity Information Extractor

This folder contains scripts to extract and display all entities, furniture, objects, and receptacles from episodes in the partnr-planner dataset.

## Scripts

### `get_episode_entities.py`

Extract and display all entities from a specific episode.

#### Usage

**Basic usage - Display entities:**
```bash
cd /home/nadira/partnr-planner/visualization/episode_info
python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz
```

**Include entity handles (sim object handles):**
```bash
python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz print_handles=true
```

**Save output to JSON file:**
```bash
python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz output_file=outputs/episode_100_entities.json
```

**Full example with all options:**
```bash
python get_episode_entities.py \
    episode_id=100 \
    dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    print_handles=true \
    output_file=outputs/episode_100_entities.json
```

**Run from project root:**
```bash
cd /home/nadira/partnr-planner
python visualization/episode_info/get_episode_entities.py episode_id=100
```

#### Arguments

- `episode_id` (required): The episode ID to load and extract entities from
- `dataset_path` (optional): Path to the dataset file (default: `data/datasets/partnr_episodes/v0_0/val_mini.json.gz`)
- `print_handles` (optional): Include sim handles for furniture and objects in the output (default: false)
- `output_file` (optional): Save the entity data to a JSON file at the specified path

#### Output

The script displays:
1. **Episode Information**: Episode ID, Scene ID, and episode description
2. **Entities by Category**:
   - Rooms
   - Furniture
   - Objects  
   - Receptacles
3. **Entity Handles** (if `--print-handles` is used): Mapping of entity names to their simulation handles
4. **Summary Statistics**: Count of each entity type

#### JSON Output Format

When using `--output`, the JSON file will have the following structure:

```json
{
  "episode_id": "100",
  "scene_id": "103997895_171031182",
  "episode_info": "Description of the episode...",
  "entities": {
    "rooms": ["kitchen_0", "bedroom_0", ...],
    "furniture": ["table_50", "chair_123", ...],
    "objects": ["apple_1", "book_2", ...],
    "receptacles": ["cabinet_10", "drawer_20", ...]
  },
  "handles": {
    "furniture_handles": {
      "table_50": "frl_apartment_table_01_:0000",
      "chair_123": "chair_:0001"
    },
    "object_handles": {
      "apple_1": "apple_:0000",
      "book_2": "book_:0001"
    }
  }
}
```

## Examples

### Example 1: Quick entity check
```bash
cd /home/nadira/partnr-planner/visualization/episode_info
python get_episode_entities.py episode_id=100
```

### Example 2: Specific dataset
```bash
python get_episode_entities.py episode_id=100 dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz
```

### Example 3: Full entity data with JSON export
```bash
python get_episode_entities.py \
    episode_id=100 \
    dataset_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
    print_handles=true \
  Uses Hydra for configuration management (CLI arguments use `key=value` format, not `--key value`)
- Visual sensors are disabled for faster loading
- Entity names can be used directly with the skill runner or planning tools
- The script automatically creates output directories if they don't exist
- Run `python get_episode_entities.py --help` to see all available Hydra options
### Example 4: From project root
```bash
cd /home/nadira/partnr-planner
python -m visualization.episode_info.get_episode_entities episode_id=100
```

## Notes

- The script uses the same configuration and environment setup as `skill_runner.py`
- Visual sensors are disabled for faster loading
- Entity names can be used directly with the skill runner or planning tools
- The script automatically creates output directories if they don't exist
