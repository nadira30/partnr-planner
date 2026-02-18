#!/usr/bin/env python3
"""
Smart Home Motion Sensor Integration with Habitat-LLM Environment

This module integrates motion sensor simulation with the existing partnr-planner
habitat environment. It tracks agent movements and triggers sensors in real-time
as agents navigate through the simulated house.

Usage:
    python smart_home_habitat_integration.py --episode 0
    python smart_home_habitat_integration.py --episode 0 --config baselines/heuristic_full_obs.yaml
"""

import numpy as np
import argparse
import json
import gzip
import sys
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class HabitatMotionSensorIntegration:
    """
    Integrates motion sensors with Habitat environment to track agent movements
    """
    
    def __init__(self, episode_data: Dict):
        self.episode_data = episode_data
        self.episode_id = episode_data.get('episode_id', 'unknown')
        self.scene_id = episode_data.get('scene_id', 'unknown')
        
        # Motion sensor system
        self.sensors_by_room: Dict[str, List[Dict]] = {}
        self.sensor_history: List[Dict] = []
        self.current_step = 0
        
        # Extract room and furniture information
        self.rooms = self._extract_rooms_from_episode()
        self.furniture_to_room = self._map_furniture_to_rooms()
        
        # Initialize sensors
        self._initialize_sensors()
        
        print(f"✓ Initialized motion sensor system for episode {self.episode_id}")
        print(f"  Scene: {self.scene_id}")
        print(f"  Rooms detected: {len(self.rooms)}")
        print(f"  Total sensors: {sum(len(sensors) for sensors in self.sensors_by_room.values())}")
    
    def _extract_rooms_from_episode(self) -> Dict[str, Dict]:
        """Extract room information from episode data"""
        rooms = {}
        
        # Extract positions from rigid_objs (furniture and objects)
        rigid_objs = self.episode_data.get('rigid_objs', [])
        
        # Collect all object positions
        all_positions = []
        for rigid_obj in rigid_objs:
            if isinstance(rigid_obj, list) and len(rigid_obj) >= 2:
                # rigid_obj is [object_name, transformation_matrix]
                transform_matrix = rigid_obj[1]
                if isinstance(transform_matrix, list) and len(transform_matrix) >= 3:
                    # Extract translation from transformation matrix
                    # Matrix format: [[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz], [0, 0, 0, 1]]
                    position = [
                        transform_matrix[0][3],  # tx
                        transform_matrix[1][3],  # ty
                        transform_matrix[2][3]   # tz
                    ]
                    all_positions.append(position)
        
        if not all_positions:
            # No positions found, return empty
            return rooms
        
        # Convert to numpy array for easier processing
        positions_array = np.array(all_positions)
        
        # Create rooms by clustering positions spatially
        # For simplicity, divide the space into regions based on x and z coordinates
        x_coords = positions_array[:, 0]
        z_coords = positions_array[:, 2]
        
        # Calculate overall bounds
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(positions_array[:, 1]), np.max(positions_array[:, 1])
        min_z, max_z = np.min(z_coords), np.max(z_coords)
        
        # Calculate range and divide into regions
        x_range = max_x - min_x
        z_range = max_z - min_z
        
        # If the space is large enough, divide it into multiple rooms
        if x_range > 8 or z_range > 8:
            # Create a grid of rooms
            num_x_divisions = max(2, int(x_range / 5))
            num_z_divisions = max(2, int(z_range / 5))
            
            x_step = x_range / num_x_divisions
            z_step = z_range / num_z_divisions
            
            room_id = 1
            for i in range(num_x_divisions):
                for j in range(num_z_divisions):
                    room_min_x = min_x + i * x_step
                    room_max_x = min_x + (i + 1) * x_step
                    room_min_z = min_z + j * z_step
                    room_max_z = min_z + (j + 1) * z_step
                    
                    # Count objects in this region
                    objects_in_region = np.sum(
                        (x_coords >= room_min_x) & (x_coords < room_max_x) &
                        (z_coords >= room_min_z) & (z_coords < room_max_z)
                    )
                    
                    if objects_in_region > 0:
                        room_name = f"room_{room_id}"
                        rooms[room_name] = {
                            'name': room_name,
                            'bounds': {
                                'min_x': float(room_min_x) - 0.5,
                                'max_x': float(room_max_x) + 0.5,
                                'min_y': float(min_y) - 0.5,
                                'max_y': float(max_y) + 0.5,
                                'min_z': float(room_min_z) - 0.5,
                                'max_z': float(room_max_z) + 0.5,
                            },
                            'furniture_count': int(objects_in_region),
                            'center': [
                                float((room_min_x + room_max_x) / 2),
                                float((min_y + max_y) / 2),
                                float((room_min_z + room_max_z) / 2)
                            ]
                        }
                        room_id += 1
        else:
            # Small space, treat as single room
            rooms['main_room'] = {
                'name': 'main_room',
                'bounds': {
                    'min_x': float(min_x) - 1.0,
                    'max_x': float(max_x) + 1.0,
                    'min_y': float(min_y) - 0.5,
                    'max_y': float(max_y) + 2.5,
                    'min_z': float(min_z) - 1.0,
                    'max_z': float(max_z) + 1.0,
                },
                'furniture_count': len(all_positions),
                'center': [
                    float(np.mean(x_coords)),
                    float(np.mean(positions_array[:, 1])),
                    float(np.mean(z_coords))
                ]
            }
        
        return rooms
    
    def _map_furniture_to_rooms(self) -> Dict[str, str]:
        """Create mapping of furniture to rooms"""
        furniture_to_room = {}
        name_to_receptacle = self.episode_data.get('name_to_receptacle', {})
        
        for furniture_name in name_to_receptacle.keys():
            parts = furniture_name.split('_')
            if len(parts) >= 2:
                if len(parts) >= 3 and parts[1].isdigit():
                    room_name = f"{parts[0]}_{parts[1]}"
                else:
                    room_name = parts[0]
                
                if room_name in self.rooms:
                    furniture_to_room[furniture_name] = room_name
        
        return furniture_to_room
    
    def _initialize_sensors(self):
        """Create three motion sensors for each room"""
        for room_name, room_data in self.rooms.items():
            bounds = room_data['bounds']
            center = room_data['center']
            
            # Calculate sensor positions
            x_offset = (bounds['max_x'] - bounds['min_x']) * 0.3
            z_offset = (bounds['max_z'] - bounds['min_z']) * 0.3
            
            sensors = [
                {
                    'id': f"{room_name}_sensor_1",
                    'zone': 'corner_northwest',
                    'position': np.array([
                        bounds['min_x'] + x_offset,
                        center[1],
                        bounds['min_z'] + z_offset
                    ]),
                    'radius': 2.0,
                    'active': False,
                    'trigger_count': 0
                },
                {
                    'id': f"{room_name}_sensor_2",
                    'zone': 'corner_northeast',
                    'position': np.array([
                        bounds['max_x'] - x_offset,
                        center[1],
                        bounds['min_z'] + z_offset
                    ]),
                    'radius': 2.0,
                    'active': False,
                    'trigger_count': 0
                },
                {
                    'id': f"{room_name}_sensor_3",
                    'zone': 'center',
                    'position': np.array(center),
                    'radius': 2.5,
                    'active': False,
                    'trigger_count': 0
                }
            ]
            
            self.sensors_by_room[room_name] = sensors
    
    def update_agent_position(self, agent_name: str, position: np.ndarray) -> Dict:
        """
        Update sensors based on agent position
        
        Args:
            agent_name: Name of the agent (e.g., 'agent_0', 'agent_1')
            position: 3D position array [x, y, z]
        
        Returns:
            Dictionary with trigger information
        """
        self.current_step += 1
        triggered_sensors = []
        affected_rooms = []
        
        # Check all rooms
        for room_name, sensors in self.sensors_by_room.items():
            room_bounds = self.rooms[room_name]['bounds']
            
            # Check if agent is in this room's bounds
            in_room = (
                room_bounds['min_x'] <= position[0] <= room_bounds['max_x'] and
                room_bounds['min_z'] <= position[2] <= room_bounds['max_z']
            )
            
            room_has_trigger = False
            
            # Check each sensor in the room
            for sensor in sensors:
                distance = np.linalg.norm(sensor['position'] - position)
                
                if distance <= sensor['radius']:
                    sensor['active'] = True
                    sensor['trigger_count'] += 1
                    triggered_sensors.append(sensor['id'])
                    room_has_trigger = True
                else:
                    sensor['active'] = False
            
            if room_has_trigger:
                affected_rooms.append(room_name)
        
        # Record event
        event = {
            'step': self.current_step,
            'agent': agent_name,
            'position': position.tolist(),
            'triggered_sensors': triggered_sensors,
            'affected_rooms': affected_rooms,
            'sensor_count': len(triggered_sensors)
        }
        self.sensor_history.append(event)
        
        return event
    
    def get_room_at_position(self, position: np.ndarray) -> Optional[str]:
        """Determine which room a position is in"""
        for room_name, room_data in self.rooms.items():
            bounds = room_data['bounds']
            if (bounds['min_x'] <= position[0] <= bounds['max_x'] and
                bounds['min_z'] <= position[2] <= bounds['max_z']):
                return room_name
        return None
    
    def print_sensor_status(self):
        """Print current status of all sensors"""
        print(f"\n{'='*80}")
        print(f"📡 MOTION SENSOR STATUS - Step {self.current_step}")
        print(f"{'='*80}")
        
        for room_name, sensors in self.sensors_by_room.items():
            active_count = sum(1 for s in sensors if s['active'])
            status_icon = "🟢" if active_count > 0 else "⚪"
            
            print(f"\n{status_icon} {room_name.upper()} ({active_count}/3 active):")
            for sensor in sensors:
                status = "🔴 TRIGGERED" if sensor['active'] else "⚪ IDLE"
                pos = sensor['position']
                print(f"   {status} | {sensor['id']} [{sensor['zone']}]")
                print(f"            Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                print(f"            Total triggers: {sensor['trigger_count']}")
        
        print(f"\n{'='*80}\n")
    
    def get_statistics(self) -> Dict:
        """Get statistics about sensor activity"""
        total_triggers = sum(
            sensor['trigger_count']
            for sensors in self.sensors_by_room.values()
            for sensor in sensors
        )
        
        steps_with_triggers = sum(1 for event in self.sensor_history if event['sensor_count'] > 0)
        
        room_activity = {}
        for room_name, sensors in self.sensors_by_room.items():
            room_activity[room_name] = sum(sensor['trigger_count'] for sensor in sensors)
        
        most_active_room = max(room_activity.items(), key=lambda x: x[1]) if room_activity else (None, 0)
        
        return {
            'total_steps': self.current_step,
            'total_triggers': total_triggers,
            'steps_with_triggers': steps_with_triggers,
            'detection_rate': steps_with_triggers / self.current_step if self.current_step > 0 else 0,
            'room_activity': room_activity,
            'most_active_room': most_active_room[0],
            'most_active_room_triggers': most_active_room[1]
        }
    
    def print_summary(self):
        """Print summary of sensor activity"""
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print(f"📊 MOTION SENSOR SUMMARY - Episode {self.episode_id}")
        print(f"{'='*80}")
        print(f"Scene: {self.scene_id}")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Total Sensor Triggers: {stats['total_triggers']}")
        print(f"Steps with Detection: {stats['steps_with_triggers']}")
        print(f"Detection Rate: {stats['detection_rate']:.1%}")
        print(f"\nMost Active Room: {stats['most_active_room']} ({stats['most_active_room_triggers']} triggers)")
        
        print(f"\nActivity by Room:")
        for room, count in sorted(stats['room_activity'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                bar_length = min(40, int(count / max(stats['room_activity'].values()) * 40))
                bar = '█' * bar_length
                print(f"   {room:20s} | {bar} {count}")
        
        print(f"\n{'='*80}\n")


def load_episode(dataset_path: str, episode_id: int) -> Optional[Dict]:
    """Load episode data from dataset"""
    try:
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
        
        for episode in data['episodes']:
            if str(episode['episode_id']) == str(episode_id):
                return episode
        
        print(f"Episode {episode_id} not found in dataset")
        return None
    except Exception as e:
        print(f"Error loading episode: {e}")
        return None


def demo_with_simulated_movement(motion_system: HabitatMotionSensorIntegration):
    """Demonstrate with simulated agent movement"""
    print("\n🚶 Simulating agent movement through the house...\n")
    
    # Get list of rooms
    room_list = list(motion_system.rooms.keys())
    
    if not room_list:
        print("No rooms found in episode")
        return
    
    # Simulate movement through rooms
    for i, room_name in enumerate(room_list[:7]):  # Visit up to 7 rooms
        room_center = motion_system.rooms[room_name]['center']
        
        # Add some variation to position
        position = np.array(room_center) + np.random.uniform(-0.5, 0.5, 3)
        
        print(f"Step {i+1}: Agent moves to {room_name}")
        event = motion_system.update_agent_position('agent_0', position)
        
        if event['sensor_count'] > 0:
            print(f"   🔔 {event['sensor_count']} sensor(s) triggered: {', '.join(event['triggered_sensors'])}")
        else:
            print(f"   ⚪ No sensors triggered")
        print()
    
    # Show final status
    motion_system.print_sensor_status()
    motion_system.print_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Smart Home Motion Sensor Integration with Habitat'
    )
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode ID to load')
    parser.add_argument('--dataset', type=str,
                       default='data/datasets/partnr_episodes/v0_0/val_mini.json.gz',
                       help='Path to dataset file')
    parser.add_argument('--show-rooms', action='store_true',
                       help='Show room information and exit')
    
    args = parser.parse_args()
    
    # Resolve dataset path relative to project root
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        # If running from smart_home_sensors directory, go up one level
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        dataset_path = os.path.join(project_root, dataset_path)
    
    # Load episode
    print(f"Loading episode {args.episode} from {dataset_path}...")
    episode_data = load_episode(dataset_path, args.episode)
    
    if not episode_data:
        print("Failed to load episode")
        return 1
    
    # Create motion sensor system
    motion_system = HabitatMotionSensorIntegration(episode_data)
    
    if args.show_rooms:
        print(f"\n{'='*80}")
        print(f"ROOM INFORMATION - Episode {args.episode}")
        print(f"{'='*80}")
        for room_name, room_data in motion_system.rooms.items():
            print(f"\n{room_name}:")
            print(f"   Furniture count: {room_data['furniture_count']}")
            print(f"   Center: {room_data['center']}")
            print(f"   Bounds: X[{room_data['bounds']['min_x']:.1f}, {room_data['bounds']['max_x']:.1f}] "
                  f"Z[{room_data['bounds']['min_z']:.1f}, {room_data['bounds']['max_z']:.1f}]")
            print(f"   Sensors: 3")
        return 0
    
    # Run demonstration
    demo_with_simulated_movement(motion_system)
    
    print("✅ Integration demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
