# Smart Home Motion Sensor Simulation System

A comprehensive smart home simulation system with motion sensors that detect person movement across rooms. Each room has three strategically placed sensors that trigger based on proximity.

## 📁 Files in this Directory

- **`smart_home_motion_sensors.py`** - Core motion sensor system implementation
- **`smart_home_advanced.py`** - Advanced features with analytics and automation
- **`smart_home_habitat_integration.py`** - Integration with Habitat environment
- **`test_motion_sensors.py`** - Comprehensive test suite with multiple scenarios
- **`demo_smart_home.sh`** - Quick demo script to run all features
- **`SMART_HOME_SENSORS_README.md`** - Full documentation
- **`SMART_HOME_FILES_SUMMARY.md`** - File summary and quick reference

## 🚀 Quick Start

```bash
# Navigate to this directory
cd smart_home_sensors

# Activate the habitat environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Run basic demo
python smart_home_motion_sensors.py --demo

# Run advanced multi-person tracking
python smart_home_advanced.py --track-multiple-people

# Run test suite
python test_motion_sensors.py --scenario boundary

# Or run the full demo script
./demo_smart_home.sh
```

## 📖 Documentation

See **`SMART_HOME_SENSORS_README.md`** for comprehensive documentation including:
- System architecture
- Usage instructions
- Sensor placement strategy
- Integration details
- Performance metrics
- Troubleshooting guide

## ✨ Key Features

- ✅ 3 sensors per room (strategic placement)
- ✅ Proximity-based detection
- ✅ Multi-person tracking
- ✅ Activity pattern analysis
- ✅ Automated actions (lights, alerts)
- ✅ Room transition tracking
- ✅ Comprehensive statistics

## 📊 Example Output

```
🏠 SMART HOME MOTION SENSOR SYSTEM STATUS
Total Rooms: 3
Total Sensors: 9

📍 Room: living_room_1
   Sensors Active: 2/3
   🔴 TRIGGERED | living_room_1_sensor_1 (corner_a)
   ⚪ IDLE | living_room_1_sensor_2 (corner_b)
   🔴 TRIGGERED | living_room_1_sensor_3 (center)
```

For more details, see the full documentation files in this directory.
