# ⚠️ IMPORTANT: Original Config Files Are Never Modified

## Overview

The Episode Visualizer's save feature creates **NEW separate files** containing **ONLY the objects you added**. 

**The original episode config files are NEVER modified or touched.**

---

## How It Works

### What Gets Saved

When you click **"Download New Objects File"**, the system:

✅ **Creates a NEW file** with only your added objects  
✅ **Names it clearly**: `episode_{ID}_NEW_objects_only.json`  
✅ **Leaves original untouched**: Original episode config remains unchanged  
✅ **Marks file type**: `"file_type": "added_objects_only"`  

❌ **Does NOT modify** the original episode config  
❌ **Does NOT overwrite** any existing files  
❌ **Does NOT merge** with the original automatically  

---

## File Format

### NEW Objects File (What Gets Saved)

**Filename:** `episode_100_NEW_objects_only.json`

```json
{
  "_comment": "This is a NEW file containing ONLY the added objects. It does NOT modify the original episode config.",
  "file_type": "added_objects_only",
  "original_episode_id": "100",
  "new_episode_id": "100_with_additions",
  "added_objects": {
    "initial_state": [
      {
        "number": 1,
        "object_classes": ["book"],
        "allowed_regions": ["office_1"],
        "furniture_names": ["table_36"]
      },
      {
        "number": 1,
        "object_classes": ["lamp"],
        "allowed_regions": ["bedroom_1"],
        "furniture_names": ["table_8"]
      }
    ]
  },
  "metadata": {
    "description": "Contains ONLY objects added via the visualizer UI",
    "original_episode_config": "See original episode 100 config for base objects",
    "usage": "Append these objects to the original episode's initial_state array",
    "total_objects_added": 2,
    "timestamp": "2026-02-26",
    "objects_detail": [...]
  }
}
```

### Original Episode Config (Remains Unchanged)

**Location:** `data/datasets/partnr_episodes/v0_0/val_mini.json.gz`

This file is **NEVER modified** by the visualizer. It always contains:
- Original scene configuration
- Original objects from the episode
- Original furniture and room setup

---

## Using Both Files Together

### Option 1: Manual Merge

You can manually combine both configs when needed:

```python
import json

# Load original episode config
with open('original_episode_100_config.json') as f:
    original = json.load(f)

# Load NEW objects file
with open('episode_100_NEW_objects_only.json') as f:
    additions = json.load(f)

# Combine initial_state arrays
combined_initial_state = (
    original['initial_state'] + 
    additions['added_objects']['initial_state']
)

# Create new combined config
combined_config = original.copy()
combined_config['initial_state'] = combined_initial_state
combined_config['episode_id'] = additions['new_episode_id']
```

### Option 2: Reference Both Files

Keep them separate and reference both:

```python
# Use original for base episode
base_episode = load_episode(original_config)

# Add objects from NEW file
additional_objects = load_additions(new_objects_file)
base_episode.add_objects(additional_objects)
```

### Option 3: Use NEW File Standalone

Use just the additions for object placement patterns:

```python
# Load only the NEW objects
with open('episode_100_NEW_objects_only.json') as f:
    additions = json.load(f)

# Use in a different episode
new_episode = create_episode(different_scene)
new_episode.add_objects(additions['added_objects']['initial_state'])
```

---

## Safety Features

### 1. Clear File Naming
- Original: `val_mini.json.gz` (in data folder)
- New File: `episode_100_NEW_objects_only.json` (in downloads)
- **Different names = No accidental overwrites**

### 2. File Type Marker
```json
"file_type": "added_objects_only"
```
Clearly identifies this as an addition file, not a full config.

### 3. Comment Header
```json
"_comment": "This is a NEW file containing ONLY the added objects. It does NOT modify the original episode config."
```
Explicit warning at the top of the file.

### 4. Metadata References
```json
"original_episode_config": "See original episode 100 config for base objects"
```
Reminds you to check the original for complete information.

---

## Workflow Example

### Safe Workflow for Adding Objects

1. **Load Original Episode**
   ```
   Episode 100 loads with:
   - 4 original objects (from dataset)
   - 43 furniture items
   - 10 rooms
   ```

2. **Add New Objects via UI**
   ```
   You add:
   - 1 book to office table
   - 1 lamp to bedroom table
   ```

3. **Save NEW Objects File**
   ```
   Click "Download New Objects File"
   → Downloads: episode_100_NEW_objects_only.json
   → Contains: ONLY your 2 added objects
   → Original config: UNCHANGED
   ```

4. **View Both Separately**
   ```
   Original episode config: Still has 4 original objects
   NEW objects file: Has your 2 additions
   Total if combined: 6 objects
   ```

5. **Use as Needed**
   ```
   - Keep separate for testing
   - Merge manually if needed
   - Use NEW file as template
   - Share NEW file with others
   ```

---

## What This Means

### ✅ Safe Operations

- **Adding objects via UI**: Safe, stores in memory only
- **Downloading NEW file**: Safe, creates new file only
- **Testing different setups**: Safe, each download is separate
- **Sharing additions**: Safe, share NEW files only

### ⚠️ Be Aware

- **NEW file is separate**: Not automatically merged with original
- **Need both files**: For complete picture, reference both
- **Manual merge**: If you want combined config, merge manually
- **Session-based**: Added objects reset when server restarts

### ❌ Never Happens

- **Original config modified**: NEVER touched by visualizer
- **Dataset files changed**: NEVER altered
- **Automatic overwrites**: NEVER occurs
- **Data loss**: Original always preserved

---

## File Locations

### Original Files (Never Modified)
```
/home/nadira/partnr-planner/data/datasets/partnr_episodes/v0_0/val_mini.json.gz
└── Contains original episode configs
    └── Episode 100 with original objects
```

### NEW Added Objects Files (Your Downloads)
```
~/Downloads/episode_100_NEW_objects_only.json
└── Contains ONLY objects you added
    └── Created when you click "Download New Objects File"
```

### Visualizer Working Files (Temporary)
```
/home/nadira/partnr-planner/visualization/
├── episode_100_NEW_objects_only.json (if saved via curl)
└── temp_episode_100.json (temporary, auto-deleted)
```

---

## FAQ

**Q: Will this modify my original episode configs?**  
A: **NO.** Original configs in the dataset are never touched.

**Q: Where does the downloaded file go?**  
A: Your browser's Downloads folder with name `episode_{ID}_NEW_objects_only.json`

**Q: Can I use the NEW file alone?**  
A: Yes, as a template for object placements or to add to other episodes.

**Q: How do I combine with original?**  
A: Manually merge the `initial_state` arrays or reference both files in your code.

**Q: What happens when I restart the server?**  
A: Added objects in memory are cleared. Download your NEW file before restarting to save them.

**Q: Can I have multiple NEW files?**  
A: Yes, each download creates a separate file. Rename them for organization.

---

## Best Practices

1. **Download Immediately**: Click "Download New Objects File" after adding objects
2. **Rename Files**: Give descriptive names like `episode_100_office_setup.json`
3. **Keep Both**: Store both original and NEW files for reference
4. **Document Changes**: Use metadata to describe what you added
5. **Version Control**: Track NEW files in git, not originals
6. **Test Separately**: Test NEW object additions before merging with original

---

## Summary

🔒 **Original episode configs are protected and never modified**  
📁 **NEW files contain only your added objects**  
🎯 **Clear naming prevents confusion and accidents**  
✅ **Safe to experiment without risk of data loss**  
🔄 **Easy to share, test, and merge as needed**

**Remember:** The button says "Download New Objects File" for a reason - it's creating a NEW separate file with your additions!
