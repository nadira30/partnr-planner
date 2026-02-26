# Episode Visualizer App

A web-based visualization tool for exploring PARTNR episodes, displaying rooms, furniture (with icons), and objects in an interactive interface.

## Features

- 🏠 **Room-based Layout**: View all rooms in an episode with organized cards
- 🪑 **Furniture Icons**: Visual representation of furniture using custom icons
- 🎨 **Colorful Objects**: Objects are represented by colorful squares for easy identification
- 🔍 **Interactive**: Click and explore different episodes
- 📊 **Location Tracking**: See which objects are on which furniture in which room

## Installation

The app uses the existing habitat-llm environment. Make sure you have:

1. Conda environment `habitat` activated
2. Flask installed (if not already): `conda install -c conda-forge flask`

## Usage

### Start the Server

```bash
cd /home/nadira/partnr-planner/visualization
conda activate habitat
python episode_visualizer_app.py
```

The server will start on `http://localhost:5002`

### Access the App

Open your web browser and navigate to:
```
http://localhost:5002
```

### Use the App

1. **Enter Episode ID**: Type an episode ID in the input box (e.g., `100`)
2. **Click "Load Episode"** or press Enter
3. **Explore**: 
   - Each room is shown as a card
   - Furniture items display with icons from the `receptacles_icons` folder
   - Objects appear as colored squares on their furniture
   - Hover over elements for more details

## File Structure

```
visualization/
├── episode_visualizer_app.py      # Flask backend application
├── templates/
│   └── episode_visualizer.html    # Frontend HTML/CSS/JavaScript
├── static/
│   └── receptacles_icons/         # Symlink to furniture icons
└── receptacles_icons/              # Furniture icon images
    ├── table@2x.png
    ├── chair@2x.png
    ├── bed@2x.png
    └── ...
```

## Available Icons

The following furniture types have icons:
- `table`, `chair`, `bed`, `couch`, `bench`
- `cabinet`, `chest_of_drawers`, `wardrobe`, `shelves`
- `fridge`, `oven`, `microwave`, `dishwasher`, `sink`
- `toilet`, `bathtub`, `shower`
- `stand`, `stool`, `counter`, `trashcan`
- `washer_dryer`

## API Endpoints

### GET `/`
Returns the main visualization interface.

### GET `/api/episode/<episode_id>`
Returns JSON data for a specific episode.

**Query Parameters:**
- `dataset` (optional): Path to dataset file (default: `data/datasets/partnr_episodes/v0_0/val_mini.json.gz`)

**Response:**
```json
{
  "episode_id": "100",
  "scene_id": "106366410_174226806",
  "rooms": ["kitchen_1", "bedroom_1", ...],
  "furniture_by_room": {
    "kitchen_1": ["table_5", "chair_9", ...],
    ...
  },
  "object_locations": {
    "laptop_0": {
      "furniture": "table_36",
      "room": "office_1"
    },
    ...
  }
}
```

## Customization

### Change Port

Edit `episode_visualizer_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5002)  # Change port here
```

### Modify Colors

Edit the `objectColors` array in `episode_visualizer.html`:
```javascript
const objectColors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', // Add your colors
];
```

### Add New Furniture Icons

1. Add PNG file to `receptacles_icons/` folder
2. Name it `<furniture_type>@2x.png` (e.g., `lamp@2x.png`)
3. The app will automatically use it for matching furniture

## Troubleshooting

### Port Already in Use
If port 5002 is already in use:
```bash
lsof -ti:5002 | xargs kill -9
```

### Icons Not Loading
Make sure the symbolic link exists:
```bash
ls -la /home/nadira/partnr-planner/visualization/static/receptacles_icons
```

### Episode Not Found
Verify the episode ID exists in your dataset:
```bash
conda run -n habitat python visualization/episode_info/get_episode_entities.py +episode_id=<YOUR_ID>
```

## Notes

- The app caches loaded episodes for faster subsequent access
- Unknown furniture items (starting with "unknown_") are filtered out
- Floor entities (starting with "floor_") are also filtered for cleaner display
- Objects are color-coded consistently based on their names

## Future Enhancements

- [ ] Add search/filter functionality
- [ ] Export visualization as image
- [ ] Add 3D scene preview
- [ ] Support for multiple episodes comparison
- [ ] Real-time updates during simulation
