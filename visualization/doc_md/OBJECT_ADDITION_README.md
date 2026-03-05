# Object Addition Feature

## Overview
The Episode Visualizer now supports dynamically adding objects to episodes through the web interface.

## Features

### 1. **Object Category Selection**
- Search through 107+ object categories from `object_categories_one_per_class.csv`
- Searchable dropdown with real-time filtering
- Categories include: laptop, book, vase, bottle, cushion, candle, and many more

### 2. **Room Selection**
- Dropdown populated with all rooms from the loaded episode
- Automatically updates when a new episode is loaded

### 3. **Furniture Selection**
- Dropdown shows furniture available in the selected room
- Dynamically filtered based on room selection

### 4. **Add & Refresh**
- **Add Object** button: Adds the selected object to the specified furniture in the chosen room
- **Refresh View** button: Reloads the episode view to show newly added objects
- Success messages confirm when objects are added

## Usage

1. **Load an Episode**: Enter an episode ID (e.g., 100) and click "Load Episode"

2. **Add an Object**:
   - Type to search for an object category in the search box
   - Select the desired category from the filtered list
   - Choose a room from the dropdown
   - Select furniture within that room
   - Click "Add Object"

3. **View Changes**: Click "Refresh View" to see the newly added objects appear as colored squares on the furniture

## API Endpoints

### Get Object Categories
```bash
GET /api/object-categories
```

Returns all available object categories from the CSV file.

**Example Response:**
```json
{
  "categories": [
    {"id": "B00TS18AEA", "category": "battery_charger"},
    {"id": "Eat_to_Live_Book", "category": "book"},
    ...
  ]
}
```

### Add Object to Episode
```bash
POST /api/episode/<episode_id>/add-object
Content-Type: application/json

{
  "object_category": "laptop",
  "room": "office_1",
  "furniture": "table_36"
}
```

**Example Response:**
```json
{
  "success": true,
  "object_name": "laptop_1",
  "message": "Added laptop_1 to table_36 in office_1"
}
```

### Get Episode Data
```bash
GET /api/episode/<episode_id>
```

Returns episode data including any added objects.

## Technical Details

### Object Naming
- Objects are automatically numbered based on their category
- Format: `{category}_{counter}` (e.g., "laptop_0", "laptop_1")
- Counter increments for each new object of the same category

### Storage
- Added objects are stored in memory during the Flask app session
- Objects persist across refreshes within the same session
- Restarting the Flask app clears all added objects

### Cache Management
- Episode data is cached for performance
- Cache is intelligently bypassed when added objects exist
- Ensures added objects always appear in the latest view

## Example Workflow

```bash
# Start the Flask app
cd /home/nadira/partnr-planner/visualization
conda activate habitat
python episode_visualizer_app.py

# Open browser to http://localhost:5002

# Or use API directly:
# Add a book to office table
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "book", "room": "office_1", "furniture": "table_36"}'

# Add a vase to living room
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "vase", "room": "living_room_1", "furniture": "table_45"}'

# Get updated episode data
curl http://localhost:5002/api/episode/100
```

## Files Modified

- `episode_visualizer_app.py`: Added object category loading, add object endpoint, and in-memory storage
- `templates/episode_visualizer.html`: Added UI form for object addition with searchable dropdowns
- Added `/api/object-categories` endpoint
- Added `/api/episode/<id>/add-object` endpoint

## Future Enhancements

Potential improvements:
- Persist added objects to a database or file
- Support for removing/editing added objects
- Batch object addition
- Object placement validation
- Export modified episode data
