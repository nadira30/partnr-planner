#!/usr/bin/env python3
"""
Integration module for smart home motion sensors with PARTNR planners.

This module provides hooks to integrate the motion sensor system with the 
habitat-llm evaluation runners, tracking agent movements in real-time during 
episode execution.

Usage:
    Add to your config file:
    
    smart_home_sensors:
        enabled: True
        output_dir: "outputs/sensor_logs"
        log_frequency: 10  # Log sensor status every N steps
"""

import numpy as np
import os
import json
from typing import Dict, Optional, List
from pathlib import Path

try:
    from smart_home_motion_sensors import SmartHomeSystem
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    print("Warning: smart_home_motion_sensors not available for import")


class PlannerMotionSensorIntegration:
    """
    Integration layer between PARTNR planners and motion sensor system.
    Tracks agent movements and updates sensors in real-time during episode execution.
    """
    
    def __init__(self, config: Dict, episode_data: Dict):
        """
        Initialize the motion sensor integration.
        
        Args:
            config: Configuration dictionary with sensor settings
            episode_data: Episode data containing room and furniture information
        """
        if not SENSORS_AVAILABLE:
            raise ImportError("smart_home_motion_sensors module not available")
        
        self.enabled = config.get('enabled', False)
        self.output_dir = config.get('output_dir', 'outputs/sensor_logs')
        self.log_frequency = config.get('log_frequency', 10)
        
        self.episode_id = episode_data.get('episode_id', 'unknown')
        self.scene_id = episode_data.get('scene_id', 'unknown')
        
        # Initialize motion sensor system
        self.sensor_system = SmartHomeSystem()
        self.step_count = 0
        self.detection_log = []
        
        # Extract and setup rooms
        self._setup_rooms(episode_data)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Motion sensor integration initialized for episode {self.episode_id}")
        print(f"  Tracking {len(self.sensor_system.room_sensor_arrays)} rooms")
    
    def _setup_rooms(self, episode_data: Dict):
        """Extract room information from episode and setup sensors"""
        # Try to extract rooms from episode data
        name_to_receptacle = episode_data.get('name_to_receptacle', {})
        
        if not name_to_receptacle:
            print("Warning: No room data found in episode, using default test rooms")
            self._setup_default_rooms()
            return
        
        # Group furniture by room to infer room bounds
        room_positions = {}
        
        for name in name_to_receptacle.keys():
            # Parse room name from furniture name
            # Format could be: "bedroom_1_chair_16" or similar
            parts = name.split('_')
            if len(parts) >= 2:
                # Try to identify room name
                room_candidate = None
                if len(parts) >= 3 and parts[1].isdigit():
                    room_candidate = f"{parts[0]}_{parts[1]}"
                else:
                    room_candidate = parts[0]
                
                # Check if this looks like a room name
                room_keywords = ['bedroom', 'bathroom', 'kitchen', 'living', 'dining', 
                               'hallway', 'entryway', 'laundry', 'office', 'room']
                
                if any(keyword in room_candidate.lower() for keyword in room_keywords):
                    if room_candidate not in room_positions:
                        room_positions[room_candidate] = []
        
        # For this simplified version, create generic rooms if we can't extract them
        if not room_positions:
            self._setup_default_rooms()
        else:
            # Setup rooms with estimated bounds
            for room_name in room_positions.keys():
                # Create generic room bounds
                room_bounds = {
                    'min_x': 0.0, 'max_x': 5.0,
                    'min_y': 0.0, 'max_y': 3.0,
                    'min_z': 0.0, 'max_z': 5.0
                }
                self.sensor_system.add_room(room_name, room_bounds)
    
    def _setup_default_rooms(self):
        """Setup default room layout for testing"""
        default_rooms = {
            'living_room_1': {
                'min_x': 0.0, 'max_x': 6.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 0.0, 'max_z': 5.0
            },
            'kitchen_1': {
                'min_x': 6.0, 'max_x': 10.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 0.0, 'max_z': 4.0
            },
            'bedroom_1': {
                'min_x': 0.0, 'max_x': 5.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 5.0, 'max_z': 9.0
            }
        }
        
        for room_name, bounds in default_rooms.items():
            self.sensor_system.add_room(room_name, bounds)
    
    def update_agent_positions(self, agent_positions: Dict[int, np.ndarray], 
                              agent_rooms: Optional[Dict[int, str]] = None):
        """
        Update sensors based on current agent positions.
        
        Args:
            agent_positions: Dictionary mapping agent_id to position array [x, y, z]
            agent_rooms: Optional dictionary mapping agent_id to room name
        """
        if not self.enabled:
            return
        
        self.step_count += 1
        
        for agent_id, position in agent_positions.items():
            # Determine room (use provided or infer)
            room = None
            if agent_rooms and agent_id in agent_rooms:
                room = agent_rooms[agent_id]
            elif len(self.sensor_system.room_sensor_arrays) > 0:
                # Use first room as default
                room = list(self.sensor_system.room_sensor_arrays.keys())[0]
            
            if room:
                event = self.sensor_system.update_person_location(position, room)
                self.detection_log.append(event)
                
                # Log status periodically
                if self.step_count % self.log_frequency == 0:
                    self._log_status()
    
    def _log_status(self):
        """Log current sensor status"""
        print(f"\n📡 Motion Sensors - Step {self.step_count}:")
        for room_name, sensor_array in self.sensor_system.room_sensor_arrays.items():
            active_count = sum(1 for s in sensor_array.sensors if s.is_triggered)
            if active_count > 0:
                print(f"  🟢 {room_name}: {active_count}/3 sensors active")
    
    def get_current_status(self) -> str:
        """Get current sensor system status"""
        return self.sensor_system.get_system_status()
    
    def save_results(self):
        """Save sensor activity results to file"""
        if not self.enabled:
            return
        
        output_file = os.path.join(
            self.output_dir, 
            f"sensor_log_episode_{self.episode_id}.json"
        )
        
        summary = self.sensor_system.get_detection_summary()
        
        results = {
            'episode_id': self.episode_id,
            'scene_id': self.scene_id,
            'total_steps': self.step_count,
            'detection_summary': summary,
            'detection_history': self.detection_log[:100]  # Save first 100 events
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Sensor results saved to {output_file}")
        
        # Print summary
        print(f"\n📊 Motion Sensor Summary:")
        print(f"   Total steps: {self.step_count}")
        print(f"   Total triggers: {summary.get('total_triggers', 0)}")
        print(f"   Detection rate: {summary.get('detection_rate', 0):.1%}")
    
    def print_final_report(self):
        """Print final sensor activity report"""
        if not self.enabled:
            return
        
        print(self.sensor_system.get_system_status())
        
        summary = self.sensor_system.get_detection_summary()
        print(f"\n{'='*80}")
        print(f"📊 MOTION SENSOR FINAL REPORT - Episode {self.episode_id}")
        print(f"{'='*80}")
        print(f"Total Steps: {self.step_count}")
        print(f"Total Sensor Triggers: {summary.get('total_triggers', 0)}")
        print(f"Detection Events: {summary.get('events_with_detections', 0)}/{summary.get('total_events', 0)}")
        print(f"Detection Rate: {summary.get('detection_rate', 0):.1%}")
        
        if summary.get('sensor_trigger_counts'):
            print(f"\nTop 3 Most Active Sensors:")
            sorted_sensors = sorted(
                summary['sensor_trigger_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (sensor_id, count) in enumerate(sorted_sensors[:3], 1):
                print(f"   {i}. {sensor_id}: {count} triggers")
        
        print(f"{'='*80}\n")


def create_sensor_integration(config: Dict, episode_data: Dict) -> Optional[PlannerMotionSensorIntegration]:
    """
    Factory function to create sensor integration if enabled and available.
    
    Args:
        config: Configuration dictionary
        episode_data: Episode data
        
    Returns:
        PlannerMotionSensorIntegration instance or None if disabled/unavailable
    """
    if not SENSORS_AVAILABLE:
        return None
    
    sensor_config = config.get('smart_home_sensors', {})
    
    if not sensor_config.get('enabled', False):
        return None
    
    try:
        return PlannerMotionSensorIntegration(sensor_config, episode_data)
    except Exception as e:
        print(f"Warning: Failed to initialize motion sensor integration: {e}")
        return None


def extract_agent_positions(sim, num_agents: int = 2) -> Dict[int, np.ndarray]:
    """
    Extract current positions of all agents from the simulator.
    
    Args:
        sim: Habitat simulator instance
        num_agents: Number of agents to track
        
    Returns:
        Dictionary mapping agent_id to position array
    """
    positions = {}
    
    try:
        for agent_id in range(num_agents):
            agent_pos = sim.agents_mgr[agent_id].articulated_agent.base_pos
            positions[agent_id] = np.array(agent_pos)
    except Exception as e:
        print(f"Warning: Could not extract agent positions: {e}")
    
    return positions
