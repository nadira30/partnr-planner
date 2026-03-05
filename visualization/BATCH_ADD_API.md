# Batch Object Addition API

This document explains how to add multiple objects to an episode using YAML configuration files.

## Quick Start

### 1. Create a YAML file with objects to add

```yaml
objects:
  - object_category: "laptop"
    room: "office_1"
    furniture: "table_36"
  
  - object_category: "book"
    room: "bedroom_1"
    furniture: "table_25"
  
  - object_category: "mug"
    room: "kitchen_1"
    furniture: "counter_19"
```

### 2. Submit via API

**Method 1: File Upload**
```bash
curl -X POST http://localhost:5002/api/episode/100/add-objects-batch \
  -F "file=@my_objects.yaml"
```

**Method 2: Raw YAML Content**
```bash
curl -X POST http://localhost:5002/api/episode/100/add-objects-batch \
  -H "Content-Type: application/x-yaml" \
  --data-binary @my_objects.yaml
```

**Method 3: JSON with YAML Content**
```bash
curl -X POST http://localhost:5002/api/episode/100/add-objects-batch \
  -H "Content-Type: application/json" \
  -d '{
    "yaml_content": "objects:\n  - object_category: laptop\n    room: office_1\n    furniture: table_36\n"
  }'
```

## YAML Format

### Required Fields

Each object must have:
- **object_category**: The category/type of object (e.g., "laptop", "book", "mug")
- **room**: The room where the object should be placed (e.g., "office_1", "bedroom_1")
- **furniture**: The furniture piece to place the object on (e.g., "table_36", "counter_19", or "floor")

### Optional Fields

- **object_id**: Specific object ID from the object database (e.g., "Laptop_10")
  - If not provided, will be looked up based on object_category
- **position**: Manual position override `{x: 0.0, y: 0.85, z: 0.0}`
  - If not provided, position will be calculated automatically based on furniture location
- **receptacle_handle**: Specific receptacle handle for advanced use cases
  - Usually not needed; calculated automatically

### Example with All Fields

```yaml
objects:
  - object_category: "laptop"
    room: "office_1"
    furniture: "table_36"
    object_id: "Laptop_10"
    position: {x: -5.4, y: 0.93, z: -6.5}
    
  - object_category: "book"
    room: "bedroom_1"
    furniture: "table_25"
    # Only required fields - position calculated automatically
```

## API Response

```json
{
  "success": true,
  "total": 3,
  "success_count": 3,
  "failure_count": 0,
  "results": [
    {
      "index": 0,
      "success": true,
      "object_category": "laptop",
      "object_name": "laptop_0",
      "message": "Added laptop_0 to table_36 in office_1"
    },
    {
      "index": 1,
      "success": true,
      "object_category": "book",
      "object_name": "book_1",
      "message": "Added book_1 to table_25 in bedroom_1"
    },
    {
      "index": 2,
      "success": false,
      "object_category": "invalid_object",
      "error": "Object not found in database"
    }
  ]
}
```

## Features

### Automatic Position Calculation

The Y-coordinate (height) is automatically calculated using:

1. **Existing Objects**: If other objects exist on the same furniture, their Y-coordinate is used
2. **Furniture Type Heuristics**: Based on furniture type (tables: 0.85m, counters: 0.90m, beds: 0.55m, etc.)
3. **Generic Fallback**: 0.85m default for unknown furniture types

This ensures objects are placed at the correct surface height for proper physics simulation.

### Smart Relationship Building

The system automatically:
- Creates proper object-to-receptacle mappings
- Establishes kinematic relationships
- Ensures objects are pickable and moveable
- Handles both rigid and articulated furniture

## Comparison: Single vs Batch API

### Single Object API
```bash
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{
    "object_category": "laptop",
    "room": "office_1",
    "furniture": "table_36"
  }'
```

### Batch API (YAML)
```bash
curl -X POST http://localhost:5002/api/episode/100/add-objects-batch \
  -F "file=@objects.yaml"
```

**Advantages of Batch API:**
- Add multiple objects in one request
- Version control friendly (YAML files)
- Easy to share and reproduce episode modifications
- Cleaner for complex scenarios

## Example Use Cases

### Setting up a Complete Office
```yaml
objects:
  # Desk setup
  - object_category: "laptop"
    room: "office_1"
    furniture: "table_5"
  - object_category: "monitor_stand"
    room: "office_1"
    furniture: "table_5"
  - object_category: "keyboard"
    room: "office_1"
    furniture: "table_5"
  
  # Bookshelf
  - object_category: "book"
    room: "office_1"
    furniture: "shelves_26"
  - object_category: "book"
    room: "office_1"
    furniture: "shelves_26"
  
  # Floor items
  - object_category: "cushion"
    room: "office_1"
    furniture: "floor"
```

### Kitchen Setup
```yaml
objects:
  # Counter items
  - object_category: "kettle"
    room: "kitchen_1"
    furniture: "counter_19"
  - object_category: "mug"
    room: "kitchen_1"
    furniture: "counter_19"
  
  # Cabinet storage
  - object_category: "bowl"
    room: "kitchen_1"
    furniture: "cabinet_40"
  - object_category: "plate"
    room: "kitchen_1"
    furniture: "cabinet_40"
  
  # Fridge
  - object_category: "bottle"
    room: "kitchen_1"
    furniture: "fridge_58"
```

## Error Handling

The batch API continues processing even if individual objects fail:

```json
{
  "success": false,
  "total": 3,
  "success_count": 2,
  "failure_count": 1,
  "results": [
    {"index": 0, "success": true, ...},
    {"index": 1, "success": false, "error": "Furniture not found"},
    {"index": 2, "success": true, ...}
  ]
}
```

You can check which objects succeeded/failed and retry if needed.

## Tips

1. **Start with required fields only** - let the system calculate positions automatically
2. **Use descriptive object categories** - matches the object database
3. **Check furniture names** - use the visualization UI to see valid furniture names
4. **Test incrementally** - start with a few objects, then add more
5. **Keep YAML files in version control** - easy to reproduce episode setups

## See Also

- [example_objects.yaml](example_objects.yaml) - Example YAML file
- Single object API: `POST /api/episode/<id>/add-object`
- Export episode: `GET /api/episode/<id>/export-config`
