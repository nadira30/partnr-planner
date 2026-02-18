# Smart Home Motion Sensor Integration with PARTNR Planners

## Overview

The motion sensor system can now be integrated with the PARTNR planner execution to track agent movements in real-time during episodes.

## Quick Start

### Option 1: Using the Integration Module (Recommended for Future Development)

The integration module (`planner_integration.py`) provides hooks to connect the sensor system with the planner, but requires modifications to the main `play_planner_demo.py` file.

### Option 2: Standalone Monitoring (Current Working Approach)

Run the motion sensor system separately to analyze specific episodes:

```bash
cd smart_home_sensors

# Monitor specific episode
python smart_home_habitat_integration.py --episode 0
```

## Integration Steps (For Developers)

To fully integrate motion sensors into the planner execution:

### 1. Add Configuration Support

Add to your config YAML (e.g., `habitat_llm/conf/baselines/heuristic_full_obs.yaml`):

```yaml
# Smart home motion sensor configuration
smart_home_sensors:
  enabled: True                    # Enable sensor tracking
  output_dir: "outputs/sensor_logs"  # Where to save logs
  log_frequency: 10                # Log status every N steps
```

### 2. Import the Integration Module

In `habitat_llm/examples/play_planner_demo.py`, add:

```python
# Add to imports section
import sys
sys.path.insert(0, 'smart_home_sensors')
from planner_integration import create_sensor_integration, extract_agent_positions
```

### 3. Initialize Sensors at Episode Start

In the `run_planner` function, after environment setup:

```python
# After env_interface is created
sensor_integration = None
if hasattr(config, 'smart_home_sensors'):
    episode_data = env_interface.env.env.env._env.current_episode.__dict__
    sensor_integration = create_sensor_integration(config, episode_data)
```

### 4. Update Sensors During Execution

In the evaluation runner's step loop, add:

```python
# After each agent action/movement
if sensor_integration:
    sim = env_interface.env.env.env._env.sim
    agent_positions = extract_agent_positions(sim, num_agents=2)
    sensor_integration.update_agent_positions(agent_positions)
```

### 5. Save Results at Episode End

After episode completion:

```python
# At episode end
if sensor_integration:
    sensor_integration.print_final_report()
    sensor_integration.save_results()
```

## Example Output When Integrated

When running with sensors enabled, you'll see:

```
✓ Motion sensor integration initialized for episode 0
  Tracking 3 rooms

📡 Motion Sensors - Step 10:
  🟢 living_room_1: 2/3 sensors active
  
📡 Motion Sensors - Step 20:
  🟢 kitchen_1: 3/3 sensors active

================================================================================
📊 MOTION SENSOR FINAL REPORT - Episode 0
================================================================================
Total Steps: 150
Total Sensor Triggers: 342
Detection Events: 145/150
Detection Rate: 96.7%

Top 3 Most Active Sensors:
   1. kitchen_1_sensor_3: 45 triggers
   2. living_room_1_sensor_2: 38 triggers
   3. bedroom_1_sensor_1: 32 triggers
================================================================================

✓ Sensor results saved to outputs/sensor_logs/sensor_log_episode_0.json
```

## Standalone Usage (No Integration Needed)

You can also use the sensor system standalone without modifying the planner:

### Basic Demo
```bash
cd smart_home_sensors
python smart_home_motion_sensors.py --demo
```

### Advanced Multi-Person Tracking
```bash
python smart_home_advanced.py --track-multiple-people
```

### Test Scenarios
```bash
python test_motion_sensors.py --scenario boundary
python test_motion_sensors.py --list-scenarios  # See all scenarios
```

### Episode Analysis
```bash
python smart_home_habitat_integration.py --episode 0 --show-rooms
```

## Benefits of Integration

When integrated with the planner:

1. **Real-Time Monitoring**: See which sensors trigger as agents move
2. **Activity Analysis**: Understand agent movement patterns across rooms
3. **Performance Insights**: Identify which areas agents visit most
4. **Automation Testing**: Test smart home automations with realistic movement
5. **Data Collection**: Gather sensor data for ML training or analysis

## Files

- `planner_integration.py` - Integration layer for connecting sensors to planner
- `smart_home_motion_sensors.py` - Core sensor system
- `smart_home_advanced.py` - Advanced analytics
- `smart_home_habitat_integration.py` - Episode data integration
- `test_motion_sensors.py` - Test suite

## Current Status

✅ **Standalone system**: Fully functional
✅ **Integration module**: Ready for use
⏳ **Automatic integration**: Requires planner code modifications

The integration module is ready to use, but requires adding the hooks listed above to the main planner code. This keeps the sensor system modular and doesn't modify existing PARTNR code unless explicitly enabled in configuration.

## Notes

- The sensor system is designed to be non-intrusive
- It only activates when explicitly enabled in config
- Failed sensor initialization won't crash the planner
- All sensor output is logged separately from planner metrics
- Sensor data is saved to JSON for post-analysis

For questions or issues, see the main documentation in `SMART_HOME_SENSORS_README.md`.
