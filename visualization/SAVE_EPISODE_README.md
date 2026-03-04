# Episode Save Script

## Overview

The `save_episode_with_additions.py` script allows you to save episode data with any objects that have been added through the visualizer API to a timestamped JSON file.

## Features

- **Load Original Episodes**: Reads episodes from the gzipped dataset (`val_mini.json.gz`)
- **Fetch API Additions**: Retrieves any objects added via the Flask visualizer UI
- **Merge Data**: Combines original episode data with added objects
- **Timestamped Output**: Saves files with current date/time in the filename
- **Flexible Output**: Can save single episodes or entire modified datasets

## Usage

### Basic Usage

Save a single episode with any added objects:

```bash
cd /home/nadira/partnr-planner
python3 visualization/save_episode_with_additions.py --episode 100
```

This will create a file like:
- `visualization/data/episode_100_modified_2026-02-27_11-20-02.json`

### Save Full Dataset

To save the entire dataset (all episodes) with your modifications:

```bash
python3 visualization/save_episode_with_additions.py --episode 100 --save-full-dataset
```

This creates:
- Single episode: `visualization/data/episode_100_modified_TIMESTAMP.json`
- Full dataset: `visualization/data/val_mini_modified_TIMESTAMP.json.gz`

### Custom Options

```bash
# Custom dataset path
python3 visualization/save_episode_with_additions.py \
  --episode 100 \
  --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz

# Custom API URL (if Flask is running on different port)
python3 visualization/save_episode_with_additions.py \
  --episode 100 \
  --api-url http://localhost:5002

# Custom output directory
python3 visualization/save_episode_with_additions.py \
  --episode 100 \
  --output-dir my_custom_output
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--episode` | Yes | - | Episode ID to save |
| `--dataset` | No | `data/datasets/partnr_episodes/v0_0/val_mini.json.gz` | Path to dataset file |
| `--api-url` | No | `http://localhost:5002` | Flask API base URL |
| `--output-dir` | No | `visualization/data` | Output directory for saved files |
| `--save-full-dataset` | No | False | Save entire dataset with modifications |

## Output Files

### Single Episode JSON

When saving a single episode, the output file contains:

```json
{
  "episode_id": "100",
  "scene_id": "106366410_174226806",
  "additional_obj_config_paths": [
    "...",
    {
      "type": "added_via_visualizer",
      "timestamp": "2026-02-27 11:20:02",
      "initial_state": [
        {
          "number": 1,
          "object_classes": ["apple"],
          "allowed_regions": ["kitchen"],
          "furniture_names": ["kitchen_counter_0"]
        }
      ],
      "objects_detail": [
        {
          "object_name": "apple_0",
          "object_category": "apple",
          "room": "kitchen",
          "furniture": "kitchen_counter_0"
        }
      ]
    }
  ],
  ...
}
```

### Full Dataset (gzipped)

When using `--save-full-dataset`, the script creates a complete dataset file with the modified episode included.

## Workflow

1. **Start Flask Server**: 
   ```bash
   cd visualization
   python3 episode_visualizer_app.py
   ```

2. **Add Objects**: 
   - Open http://localhost:5002 in browser
   - Load episode 100
   - Add objects through the UI

3. **Save Changes**:
   ```bash
   python3 visualization/save_episode_with_additions.py --episode 100
   ```

4. **Use Modified Episode**:
   - The saved JSON can be used to create a new episode
   - Or reference it in your habitat-llm configuration

## Notes

- Files are saved with timestamps to avoid overwriting
- If no objects have been added via the API, the original episode is saved
- The Flask API must be running to fetch added objects
- Output directory is created automatically if it doesn't exist
- Single episode files are uncompressed JSON for easy inspection
- Full dataset files are gzipped to match the original format

## Dependencies

- Python 3.x
- `requests` library for API calls
- Standard library: `gzip`, `json`, `pathlib`, `datetime`

## Troubleshooting

**Error: Dataset file not found**
- Check that the dataset path is correct
- Verify the file exists: `ls data/datasets/partnr_episodes/v0_0/val_mini.json.gz`

**Warning: Could not connect to API**
- Make sure the Flask server is running
- Check the API URL matches your Flask server port

**No objects added**
- This is normal if you haven't added any objects through the visualizer
- The script will still save the original episode data
