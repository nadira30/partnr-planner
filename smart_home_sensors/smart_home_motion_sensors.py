#!/usr/bin/env python3
"""
Smart Home Motion Sensor Simulation

This module simulates a smart home system with motion sensors distributed across rooms.
Each room has three motion sensors positioned at different locations to detect person movement.
Sensors can be triggered individually or in combination depending on the person's position.

Usage:
    python smart_home_motion_sensors.py
    python smart_home_motion_sensors.py --episode 0 --dataset data/datasets/partnr_episodes/v0_0/val_mini.json.gz
"""

import numpy as np
import subprocess
import sys
import json
import gzip
import argparse
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SensorZone(Enum):
    """Enum for sensor placement zones within a room"""
    CORNER_A = "corner_a"  # Top-left corner area
    CORNER_B = "corner_b"  # Top-right corner area
    CENTER = "center"       # Center of the room
    ENTRANCE = "entrance"   # Near the entrance/door


@dataclass
class MotionSensor:
    """Represents a single motion sensor with position and detection range"""
    sensor_id: str
    room_name: str
    position: np.ndarray  # 3D position [x, y, z]
    zone: SensorZone
    detection_radius: float = 2.5  # Detection radius in meters
    is_triggered: bool = False
    trigger_count: int = 0
    last_trigger_time: float = 0.0
    
    def __str__(self):
        status = "🔴 TRIGGERED" if self.is_triggered else "⚪ IDLE"
        return f"{status} | {self.sensor_id} ({self.zone.value}) at [{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}]"
    
    def check_person_in_range(self, person_position: np.ndarray) -> bool:
        """Check if a person is within detection range"""
        distance = np.linalg.norm(self.position - person_position)
        return distance <= self.detection_radius
    
    def trigger(self, timestamp: float = 0.0):
        """Activate the sensor"""
        self.is_triggered = True
        self.trigger_count += 1
        self.last_trigger_time = timestamp
    
    def reset(self):
        """Reset sensor to idle state"""
        self.is_triggered = False


@dataclass
class RoomSensorArray:
    """Collection of three motion sensors for a room"""
    room_name: str
    room_bounds: Dict[str, float]  # min/max x, y, z coordinates
    sensors: List[MotionSensor] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize three sensors at strategic positions"""
        if not self.sensors:
            self.sensors = self._create_sensor_array()
    
    def _create_sensor_array(self) -> List[MotionSensor]:
        """Create three motion sensors at strategic positions in the room"""
        sensors = []
        
        # Calculate room dimensions and key positions
        center_x = (self.room_bounds['min_x'] + self.room_bounds['max_x']) / 2
        center_y = (self.room_bounds['min_y'] + self.room_bounds['max_y']) / 2
        center_z = (self.room_bounds['min_z'] + self.room_bounds['max_z']) / 2
        
        # Offset for corner sensors
        x_offset = (self.room_bounds['max_x'] - self.room_bounds['min_x']) * 0.25
        z_offset = (self.room_bounds['max_z'] - self.room_bounds['min_z']) * 0.25
        
        # Sensor 1: Corner A (front-left area)
        sensor1_pos = np.array([
            self.room_bounds['min_x'] + x_offset,
            center_y,
            self.room_bounds['min_z'] + z_offset
        ])
        sensors.append(MotionSensor(
            sensor_id=f"{self.room_name}_sensor_1",
            room_name=self.room_name,
            position=sensor1_pos,
            zone=SensorZone.CORNER_A
        ))
        
        # Sensor 2: Corner B (front-right area)
        sensor2_pos = np.array([
            self.room_bounds['max_x'] - x_offset,
            center_y,
            self.room_bounds['min_z'] + z_offset
        ])
        sensors.append(MotionSensor(
            sensor_id=f"{self.room_name}_sensor_2",
            room_name=self.room_name,
            position=sensor2_pos,
            zone=SensorZone.CORNER_B
        ))
        
        # Sensor 3: Center (room center)
        sensor3_pos = np.array([center_x, center_y, center_z])
        sensors.append(MotionSensor(
            sensor_id=f"{self.room_name}_sensor_3",
            room_name=self.room_name,
            position=sensor3_pos,
            zone=SensorZone.CENTER
        ))
        
        return sensors
    
    def update_sensors(self, person_position: np.ndarray, timestamp: float = 0.0) -> Set[str]:
        """Update all sensors based on person position, returns triggered sensor IDs"""
        triggered = set()
        
        # Reset all sensors first
        for sensor in self.sensors:
            sensor.reset()
        
        # Check each sensor for person detection
        for sensor in self.sensors:
            if sensor.check_person_in_range(person_position):
                sensor.trigger(timestamp)
                triggered.add(sensor.sensor_id)
        
        return triggered
    
    def get_status_summary(self) -> str:
        """Get a summary of all sensor states in this room"""
        triggered_count = sum(1 for s in self.sensors if s.is_triggered)
        lines = [
            f"\n📍 Room: {self.room_name}",
            f"   Sensors Active: {triggered_count}/3"
        ]
        for sensor in self.sensors:
            lines.append(f"   {sensor}")
        return "\n".join(lines)


class SmartHomeSystem:
    """Main smart home system managing all motion sensors across rooms"""
    
    def __init__(self):
        self.room_sensor_arrays: Dict[str, RoomSensorArray] = {}
        self.detection_history: List[Dict] = []
        self.current_timestamp: float = 0.0
    
    def add_room(self, room_name: str, room_bounds: Dict[str, float]):
        """Add a room with its motion sensor array"""
        sensor_array = RoomSensorArray(room_name=room_name, room_bounds=room_bounds)
        self.room_sensor_arrays[room_name] = sensor_array
        print(f"✓ Added room '{room_name}' with 3 motion sensors")
    
    def update_person_location(self, person_position: np.ndarray, room_name: str = None):
        """Update sensors based on person's current position"""
        self.current_timestamp += 1.0
        
        all_triggered = set()
        active_rooms = []
        
        # Update all rooms or specific room
        rooms_to_check = [room_name] if room_name else list(self.room_sensor_arrays.keys())
        
        for room in rooms_to_check:
            if room in self.room_sensor_arrays:
                triggered = self.room_sensor_arrays[room].update_sensors(
                    person_position, 
                    self.current_timestamp
                )
                if triggered:
                    all_triggered.update(triggered)
                    active_rooms.append(room)
        
        # Record detection event
        detection_event = {
            'timestamp': self.current_timestamp,
            'position': person_position.tolist(),
            'triggered_sensors': list(all_triggered),
            'active_rooms': active_rooms,
            'sensor_count': len(all_triggered)
        }
        self.detection_history.append(detection_event)
        
        return detection_event
    
    def get_system_status(self) -> str:
        """Get complete system status"""
        lines = [
            "\n" + "="*80,
            "🏠 SMART HOME MOTION SENSOR SYSTEM STATUS",
            "="*80,
            f"Total Rooms: {len(self.room_sensor_arrays)}",
            f"Total Sensors: {len(self.room_sensor_arrays) * 3}",
            f"Timestamp: {self.current_timestamp:.1f}s",
        ]
        
        for room_name, sensor_array in self.room_sensor_arrays.items():
            lines.append(sensor_array.get_status_summary())
        
        lines.append("\n" + "="*80)
        return "\n".join(lines)
    
    def get_detection_summary(self) -> Dict:
        """Get summary statistics of all detections"""
        if not self.detection_history:
            return {"total_events": 0}
        
        total_events = len(self.detection_history)
        events_with_detections = sum(1 for e in self.detection_history if e['sensor_count'] > 0)
        
        sensor_trigger_counts = {}
        for event in self.detection_history:
            for sensor_id in event['triggered_sensors']:
                sensor_trigger_counts[sensor_id] = sensor_trigger_counts.get(sensor_id, 0) + 1
        
        return {
            'total_events': total_events,
            'events_with_detections': events_with_detections,
            'detection_rate': events_with_detections / total_events if total_events > 0 else 0,
            'sensor_trigger_counts': sensor_trigger_counts,
            'most_active_sensor': max(sensor_trigger_counts.items(), key=lambda x: x[1])[0] if sensor_trigger_counts else None
        }


def load_episode_rooms(dataset_path: str, episode_id: int) -> Dict[str, Dict]:
    """Load room information from episode dataset"""
    try:
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
        
        for episode in data['episodes']:
            if str(episode['episode_id']) == str(episode_id):
                return extract_room_bounds_from_episode(episode)
        
        return {}
    except Exception as e:
        print(f"Warning: Could not load episode {episode_id}: {e}")
        return {}


def extract_room_bounds_from_episode(episode: Dict) -> Dict[str, Dict]:
    """Extract room bounds from episode data"""
    rooms = {}
    
    # Try to extract room information from receptacles and their positions
    name_to_receptacle = episode.get('name_to_receptacle', {})
    
    # Group receptacles by room (inferred from name prefix)
    room_objects = {}
    
    for name, receptacle in name_to_receptacle.items():
        # Extract room name from object name (e.g., "bedroom_1_chair_16")
        parts = name.split('_')
        if len(parts) >= 2:
            # Assume format: roomtype_number_furniture_id
            potential_room = f"{parts[0]}_{parts[1]}"
            if 'room' in parts[0] or parts[0] in ['kitchen', 'bedroom', 'bathroom', 'living', 'hallway', 'laundryroom', 'entryway']:
                if potential_room not in room_objects:
                    room_objects[potential_room] = []
                
                # Get position from receptacle
                if 'translation' in receptacle:
                    pos = receptacle['translation']
                    room_objects[potential_room].append(pos)
    
    # Calculate bounds for each room
    for room_name, positions in room_objects.items():
        if positions:
            positions = np.array(positions)
            rooms[room_name] = {
                'min_x': float(np.min(positions[:, 0])) - 1.0,
                'max_x': float(np.max(positions[:, 0])) + 1.0,
                'min_y': float(np.min(positions[:, 1])) - 0.5,
                'max_y': float(np.max(positions[:, 1])) + 2.5,
                'min_z': float(np.min(positions[:, 2])) - 1.0,
                'max_z': float(np.max(positions[:, 2])) + 1.0,
            }
    
    return rooms


def simulate_person_movement_demo():
    """Demonstrate the smart home system with simulated person movement"""
    print("\n🏠 Initializing Smart Home Motion Sensor System...")
    
    # Create smart home system
    smart_home = SmartHomeSystem()
    
    # Define example rooms with bounds
    example_rooms = {
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
    
    # Add rooms to the system
    for room_name, bounds in example_rooms.items():
        smart_home.add_room(room_name, bounds)
    
    print(smart_home.get_system_status())
    
    # Simulate person movement through the house
    print("\n🚶 Simulating person movement through the house...\n")
    
    movement_sequence = [
        (np.array([1.5, 1.5, 1.5]), "living_room_1", "Person enters living room (corner area)"),
        (np.array([3.0, 1.5, 2.5]), "living_room_1", "Person moves to center of living room"),
        (np.array([5.5, 1.5, 2.0]), "living_room_1", "Person approaches kitchen entrance"),
        (np.array([7.0, 1.5, 2.0]), "kitchen_1", "Person enters kitchen"),
        (np.array([8.0, 1.5, 2.0]), "kitchen_1", "Person at kitchen center"),
        (np.array([2.5, 1.5, 7.0]), "bedroom_1", "Person enters bedroom"),
        (np.array([2.5, 1.5, 7.0]), "bedroom_1", "Person stays in bedroom center"),
    ]
    
    for position, room, description in movement_sequence:
        print(f"⏱️  Time {smart_home.current_timestamp:.1f}s: {description}")
        event = smart_home.update_person_location(position, room)
        
        if event['sensor_count'] > 0:
            print(f"   🔔 {event['sensor_count']} sensor(s) triggered: {', '.join(event['triggered_sensors'])}")
        else:
            print(f"   ⚪ No sensors triggered")
        print()
    
    # Show final status
    print(smart_home.get_system_status())
    
    # Show detection summary
    summary = smart_home.get_detection_summary()
    print(f"\n📊 DETECTION SUMMARY")
    print(f"   Total events: {summary['total_events']}")
    print(f"   Events with detections: {summary['events_with_detections']}")
    print(f"   Detection rate: {summary['detection_rate']:.1%}")
    if summary.get('most_active_sensor'):
        print(f"   Most active sensor: {summary['most_active_sensor']}")
    
    return smart_home


def simulate_with_episode_data(episode_id: int, dataset_path: str):
    """Simulate smart home using actual episode room data"""
    print(f"\n🏠 Loading episode {episode_id} from dataset...")
    
    rooms = load_episode_rooms(dataset_path, episode_id)
    
    if not rooms:
        print("⚠️  No rooms found in episode data. Using demo simulation.")
        return simulate_person_movement_demo()
    
    print(f"✓ Found {len(rooms)} rooms in episode")
    
    # Create smart home system with episode rooms
    smart_home = SmartHomeSystem()
    for room_name, bounds in rooms.items():
        smart_home.add_room(room_name, bounds)
    
    print(smart_home.get_system_status())
    
    # Simulate random movement through detected rooms
    print("\n🚶 Simulating person movement through episode rooms...\n")
    
    room_list = list(rooms.keys())
    for i, room_name in enumerate(room_list[:5]):  # Check first 5 rooms
        bounds = rooms[room_name]
        # Generate random position within room bounds
        position = np.array([
            (bounds['min_x'] + bounds['max_x']) / 2 + np.random.uniform(-0.5, 0.5),
            (bounds['min_y'] + bounds['max_y']) / 2,
            (bounds['min_z'] + bounds['max_z']) / 2 + np.random.uniform(-0.5, 0.5)
        ])
        
        print(f"⏱️  Time {smart_home.current_timestamp:.1f}s: Person in {room_name}")
        event = smart_home.update_person_location(position, room_name)
        
        if event['sensor_count'] > 0:
            print(f"   🔔 {event['sensor_count']} sensor(s) triggered")
        print()
    
    print(smart_home.get_system_status())
    
    summary = smart_home.get_detection_summary()
    print(f"\n📊 DETECTION SUMMARY")
    print(f"   Total events: {summary['total_events']}")
    print(f"   Events with detections: {summary['events_with_detections']}")
    print(f"   Detection rate: {summary['detection_rate']:.1%}")
    
    return smart_home


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Home Motion Sensor Simulation')
    parser.add_argument('--episode', type=int, default=None,
                       help='Episode ID to load room data from')
    parser.add_argument('--dataset', type=str,
                       default='data/datasets/partnr_episodes/v0_0/val_mini.json.gz',
                       help='Path to dataset file')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with simulated rooms')
    
    args = parser.parse_args()
    
    if args.demo or args.episode is None:
        smart_home = simulate_person_movement_demo()
    else:
        smart_home = simulate_with_episode_data(args.episode, args.dataset)
    
    print("\n✓ Simulation complete!")
    
    return smart_home


if __name__ == "__main__":
    main()
