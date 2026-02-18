# Smart Home Motion Sensor Simulation System

This directory contains three Python scripts that simulate a smart home with motion sensors distributed across rooms. Each room has three strategically placed sensors that detect person movement.

## Overview

The system simulates motion sensors in a smart home environment where:
- **Three sensors per room** are positioned at strategic locations (corners and center)
- **Sensors trigger based on proximity** when a person enters their detection radius (default 2.0-2.5 meters)
- **Multiple sensors can trigger simultaneously** depending on the person's location
- **Real-time tracking** of person movements and sensor activations

## Files

### 1. `smart_home_motion_sensors.py` - Basic System
The foundational implementation with core motion sensor functionality.

**Features:**
- Basic motion sensor simulation with 3 sensors per room
- Person position tracking
- Sensor trigger detection based on proximity
- Detection history and statistics
- Support for loading room layouts from episode data

**Usage:**
```bash
# Run demo with simulated rooms
python smart_home_motion_sensors.py --demo

# Use with episode data
python smart_home_motion_sensors.py --episode 0 --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz
```

**Example Output:**
```
🏠 Initializing Smart Home Motion Sensor System...
✓ Added room 'living_room_1' with 3 motion sensors
✓ Added room 'kitchen_1' with 3 motion sensors

📍 Room: living_room_1
   Sensors Active: 2/3
   🔴 TRIGGERED | living_room_1_sensor_1 (corner_a) at [1.50, 1.50, 1.25]
   ⚪ IDLE | living_room_1_sensor_2 (corner_b) at [4.50, 1.50, 1.25]
   🔴 TRIGGERED | living_room_1_sensor_3 (center) at [3.00, 1.50, 2.50]
```

### 2. `smart_home_advanced.py` - Advanced Features
Extended system with analytics, automation, and multi-person tracking.

**Features:**
- **Multi-person tracking**: Track multiple people simultaneously across rooms
- **Activity zone analyzer**: Identify hotspots and analyze movement patterns
- **Room transition tracking**: Monitor which rooms people move between
- **Automation controller**: Trigger automated actions based on sensor events
  - Auto lights on/off
  - Extended activity alerts
  - Energy-saving modes
- **Dwell time analysis**: Calculate time spent in each room
- **Performance metrics**: Sensor uptime, detection rates, and statistics

**Usage:**
```bash
# Demo multi-person tracking
python smart_home_advanced.py --track-multiple-people

# Demo activity pattern analysis
python smart_home_advanced.py --analyze-patterns

# Run all demos
python smart_home_advanced.py
```

**Example Output:**
```
👥 PERSON TRACKING SUMMARY
Person: person_1
   Current Room: kitchen
   Distance Traveled: 4.06m
   Rooms Visited: 2

🔥 TOP ACTIVITY HOTSPOTS:
   1. living_room_center: 3.0 activity units
   2. kitchen_corner_a: 3.0 activity units

🤖 AUTOMATION CONTROLLER
   💡 [AUTOMATION] Turning on lights in living_room
   🔔 [AUTOMATION] Alert: Extended activity in kitchen
```

### 3. `smart_home_habitat_integration.py` - Habitat Integration
Integration with the partnr-planner Habitat-LLM environment for realistic simulations.

**Features:**
- Load room layouts from Habitat episode datasets
- Extract room boundaries from furniture positions
- Map furniture to rooms automatically
- Track agent movements through the environment
- Generate comprehensive activity reports

**Usage:**
```bash
# Show room information for an episode
python smart_home_habitat_integration.py --episode 0 --show-rooms

# Run simulation with episode data
python smart_home_habitat_integration.py --episode 0

# Use custom dataset
python smart_home_habitat_integration.py --episode 5 --dataset path/to/dataset.json.gz
```

**Example Output:**
```
✓ Initialized motion sensor system for episode 0
  Scene: 102817140
  Rooms detected: 7
  Total sensors: 21

📊 MOTION SENSOR SUMMARY
Total Steps: 7
Total Sensor Triggers: 18
Detection Rate: 100.0%
Most Active Room: kitchen (6 triggers)

Activity by Room:
   kitchen              | ████████████████████████████████████████ 6
   living_room          | ██████████████████████████ 4
   bedroom              | ████████████████ 3
```

## Architecture

### Sensor System Components

#### MotionSensor (Basic)
- `sensor_id`: Unique identifier
- `room_name`: Parent room
- `position`: 3D coordinates [x, y, z]
- `zone`: Location zone (corner_a, corner_b, center)
- `detection_radius`: Range for triggering (meters)
- `trigger_count`: Total times triggered

#### RoomSensorArray
- Manages 3 sensors per room
- Calculates sensor positions based on room bounds
- Updates sensor states based on person position
- Provides room-level status summaries

#### SmartHomeSystem
- Central controller for all rooms and sensors
- Tracks detection history
- Provides system-wide statistics
- Manages timestamps and events

### Advanced Components

#### Person (Advanced)
- Tracks individual person movement
- Records path history with timestamps
- Calculates distance traveled
- Maintains list of visited rooms

#### ActivityZoneAnalyzer
- Analyzes activity patterns over time
- Identifies hotspots and high-traffic areas
- Tracks room transitions
- Calculates average dwell times

#### AutomationController
- Rule-based automation system
- Condition checking (e.g., motion detected, extended dwell time)
- Action execution (e.g., lights on/off, alerts)
- Maintains trigger statistics

## Sensor Placement Strategy

Each room has three sensors strategically positioned:

1. **Sensor 1 - Corner A (Northwest)**: 
   - Position: 25% from min_x, 25% from min_z
   - Purpose: Covers entrance and corner areas

2. **Sensor 2 - Corner B (Northeast)**:
   - Position: 25% from max_x, 25% from min_z
   - Purpose: Covers opposite corner and perimeter

3. **Sensor 3 - Center**:
   - Position: Room center [center_x, center_y, center_z]
   - Purpose: Covers central area with larger radius

### Detection Logic

A sensor triggers when:
```python
distance = ||sensor_position - person_position|| ≤ detection_radius
```

Multiple sensors can trigger simultaneously, providing:
- **Single sensor**: Person in specific zone
- **Two sensors**: Person between zones or moving
- **Three sensors**: Person in center of room

## Example Scenarios

### Scenario 1: Person Entering Living Room
```
Position: [1.5, 1.5, 1.5] (near corner A)
Result: Sensor 1 ✓ TRIGGERED, Sensor 2 ✗, Sensor 3 ✓ TRIGGERED
```

### Scenario 2: Person in Room Center
```
Position: [3.0, 1.5, 2.5] (room center)
Result: All 3 sensors ✓ TRIGGERED (within range)
```

### Scenario 3: Person Transitioning Between Rooms
```
Position: [5.5, 1.5, 2.0] (room boundary)
Result: Living room sensor 2 ✓, Kitchen sensor 1 ✓
```

## Integration with Partnr-Planner

The system integrates with the existing partnr-planner environment:

1. **Episode Loading**: Extracts room layouts from episode datasets
2. **Room Detection**: Infers room boundaries from furniture positions
3. **Agent Tracking**: Monitors agent movements through the environment
4. **Real-time Updates**: Updates sensor states as agents navigate

### Room Extraction Algorithm

```python
1. Parse name_to_receptacle from episode
2. Group furniture by room prefix (e.g., "bedroom_1_chair_16" → "bedroom_1")
3. Calculate bounds: [min/max x, y, z] from furniture positions
4. Add margins (±1.5m) to bounds
5. Create 3 sensors per detected room
```

## Performance Metrics

The system tracks several metrics:

- **Detection Rate**: Percentage of time steps with active sensors
- **Sensor Trigger Count**: Times each sensor has been activated
- **Room Activity**: Total triggers per room
- **Person Distance**: Total distance traveled by each person
- **Dwell Time**: Average time spent in each room
- **Transition Patterns**: Common paths between rooms

## Dependencies

- `numpy`: Numerical operations and distance calculations
- `argparse`: Command-line interface
- `json`/`gzip`: Episode data loading
- `dataclasses`: Data structure definitions
- `typing`: Type hints for clarity

## Running with Conda Environment

All scripts should be run with the habitat conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Then run any script
python smart_home_motion_sensors.py --demo
python smart_home_advanced.py --track-multiple-people
python smart_home_habitat_integration.py --episode 0
```

## Future Enhancements

Potential extensions to the system:

1. **Visualization**
   - 2D/3D floor plan visualization
   - Real-time sensor state animation
   - Heat maps of activity zones

2. **Machine Learning**
   - Occupancy prediction
   - Anomaly detection
   - Activity recognition

3. **Advanced Sensors**
   - Different sensor types (PIR, ultrasonic, camera)
   - Adjustable sensitivity
   - Battery life simulation

4. **Smart Home Integration**
   - HVAC control based on occupancy
   - Security system integration
   - Energy optimization

5. **Privacy Features**
   - Anonymized tracking
   - Configurable data retention
   - Privacy zones

## Troubleshooting

### No sensors triggered
- Check detection radius (default 2.0-2.5m)
- Verify person position is within room bounds
- Ensure sensor positions are calculated correctly

### Wrong room detection
- Review room boundary extraction
- Check furniture position data quality
- Adjust boundary margins if needed

### Module not found errors
- Activate conda environment: `conda activate habitat`
- Install dependencies: `pip install numpy`

## License

This code is part of the partnr-planner project and follows the same license.

## Author

Created as an extension to the partnr-planner smart home simulation system.
