# 🎨 Quick Start Guide - Adding Objects to Episodes

## Step-by-Step Instructions

### 1. **Start the Application**
```bash
cd /home/nadira/partnr-planner/visualization
conda activate habitat
python episode_visualizer_app.py
```

Open your browser to: **http://localhost:5002**

---

### 2. **Load an Episode**
- Enter an episode ID in the top input field (e.g., `100`)
- Click **"Load Episode"** button
- Wait for the episode to load (shows rooms, furniture, and existing objects)

---

### 3. **Add a New Object**

In the **"➕ Add New Object"** section:

#### **Step A: Choose Object Category**
- Type in the search box to filter categories (e.g., type "book")
- Select from the filtered dropdown (e.g., "book", "laptop", "vase")
- **107 categories available** including:
  - 📚 book, laptop, monitor_stand
  - 🏺 vase, bottle, bowl
  - 🪑 cushion, pillow
  - 💡 lamp, candle
  - 🎮 action_figure, board_game
  - 📦 box, basket, backpack
  - And many more!

#### **Step B: Choose Room**
- Select a room from the dropdown
- Examples: `office_1`, `living_room_1`, `bedroom_1`, `kitchen_1`

#### **Step C: Choose Furniture**
- Dropdown automatically shows furniture in the selected room
- Examples: `table_36`, `couch_5`, `counter_32`

#### **Step D: Add the Object**
- Click **"Add Object"** button
- You'll see a success message: "Added {object} to {furniture} in {room}"

---

### 4. **Refresh to See Changes**
- Click **"Refresh View"** button
- The episode reloads showing your newly added object
- Objects appear as **colored squares** on the furniture

---

## 📸 Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│                   🏠 Episode Visualizer                      │
│     Explore rooms, furniture, and objects in episodes        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  [Episode ID: 100________] [Load Episode]                   │
│                                                              │
│  ➕ Add New Object                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Object Category                                        │ │
│  │ [Search: laptop_________________]                      │ │
│  │ ┌──────────────────────┐                              │ │
│  │ │ laptop               │                              │ │
│  │ │ laptop_cover         │                              │ │
│  │ │ laptop_stand         │                              │ │
│  │ └──────────────────────┘                              │ │
│  │                                                        │ │
│  │ Room              Furniture                           │ │
│  │ [office_1▼]       [table_36▼]                         │ │
│  │                                                        │ │
│  │ [Add Object] [Refresh View]                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Episode 100 | Scene: 106366410_174226806                    │
│ Rooms: 10 | Objects: 9                                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   office_1       │ │  living_room_1   │ │   bedroom_1      │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ 🪑 table_36  🟦🟧│ │ 🛋️ couch_5   🟨  │ │ 🪑 table_8   🟩  │
│ 🪑 chair_35      │ │ 🪑 table_45  🟪  │ │ 🛏️ bed_6         │
│                  │ │                  │ │                  │
│ Objects:         │ │ Objects:         │ │ Objects:         │
│ 🟦 laptop_0      │ │ 🟨 cushion_0     │ │ 🟩 lamp_0        │
│ 🟧 book_0        │ │ 🟪 vase_0        │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 🎯 Example Scenarios

### Scenario 1: Adding a Laptop to Office
```
1. Search: "laptop"
2. Select: laptop
3. Room: office_1
4. Furniture: table_36
5. Click: Add Object
✅ Result: laptop_1 added to table_36
```

### Scenario 2: Adding Decoration to Living Room
```
1. Search: "vase"
2. Select: vase
3. Room: living_room_1
4. Furniture: table_45
5. Click: Add Object
✅ Result: vase_0 added to table_45
```

### Scenario 3: Adding Kitchen Items
```
1. Search: "bottle"
2. Select: bottle
3. Room: kitchen_1
4. Furniture: counter_32
5. Click: Add Object
✅ Result: bottle_0 added to counter_32
```

---

## 💡 Tips & Tricks

1. **Search is Your Friend**: Type part of the object name to quickly filter
   - Type "la" → shows: lamp, laptop, laptop_cover, laptop_stand
   - Type "bo" → shows: book, bottle, board_game, bowl, box

2. **Object Naming**: Objects are auto-numbered
   - First laptop: `laptop_0`
   - Second laptop: `laptop_1`
   - First book: `book_0`

3. **Multiple Objects**: You can add as many objects as you want
   - Add multiple items to the same furniture
   - Add different objects to different rooms

4. **Visual Feedback**: 
   - Success message appears after adding
   - Colored squares show on furniture after refresh
   - Each object type gets a unique color

5. **Persistence**: Objects stay during your session
   - Survive episode refreshes
   - Reset when Flask app restarts

---

## 🚀 Advanced Usage (API)

### Using cURL to Add Objects
```bash
# Add a book
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "book", "room": "office_1", "furniture": "table_36"}'

# Add a lamp
curl -X POST http://localhost:5002/api/episode/100/add-object \
  -H "Content-Type: application/json" \
  -d '{"object_category": "lamp", "room": "bedroom_1", "furniture": "table_8"}'

# Refresh to see changes
curl http://localhost:5002/api/episode/100
```

---

## ❓ Troubleshooting

**Problem**: Dropdown is empty for furniture
- **Solution**: Make sure you selected a room first

**Problem**: Object not appearing after refresh
- **Solution**: Click "Refresh View" button to reload the episode

**Problem**: Search not working
- **Solution**: Wait for the page to fully load (categories need to load first)

**Problem**: Category list shows "Loading categories..."
- **Solution**: Check that the CSV file exists at `visualization/objects/object_categories_one_per_class.csv`

---

## 📚 Available Object Categories (Sample)

```
action_figure        backpack            basket
bath_towel           battery_charger     board_game
book                 bottle              bowl
box                  bundt_pan           candle_holder
canister             carrying_case       cushion
dumbbell             folder              handbag
lamp                 laptop              laptop_cover
laptop_stand         monitor_stand       mouse_pad
multiport_hub        phone_stand         pillow
soap_dispenser       tape                vase
... and 80+ more!
```

---

**Need Help?** Check the full documentation:
- [OBJECT_ADDITION_README.md](OBJECT_ADDITION_README.md)
- [OBJECT_ADDITION_SUMMARY.md](OBJECT_ADDITION_SUMMARY.md)

**Server URL**: http://localhost:5002
