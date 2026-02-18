#!/usr/bin/env python3
"""
Smart Home Motion Sensor Test Suite

This script provides various test scenarios to demonstrate the motion sensor system.
Run different tests to see how sensors respond to various movement patterns.

Usage:
    python test_motion_sensors.py
    python test_motion_sensors.py --scenario entrance
    python test_motion_sensors.py --scenario cross-room
    python test_motion_sensors.py --list-scenarios
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from smart_home_motion_sensors import SmartHomeSystem


class MotionSensorTester:
    """Test suite for motion sensor system"""
    
    def __init__(self):
        self.smart_home = SmartHomeSystem()
        self._setup_test_house()
    
    def _setup_test_house(self):
        """Setup a standard test house layout"""
        test_rooms = {
            'entryway': {
                'min_x': 0.0, 'max_x': 3.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 0.0, 'max_z': 3.0
            },
            'living_room': {
                'min_x': 3.0, 'max_x': 8.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 0.0, 'max_z': 6.0
            },
            'kitchen': {
                'min_x': 8.0, 'max_x': 12.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 0.0, 'max_z': 4.0
            },
            'hallway': {
                'min_x': 3.0, 'max_x': 8.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 6.0, 'max_z': 8.0
            },
            'bedroom': {
                'min_x': 0.0, 'max_x': 5.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 8.0, 'max_z': 12.0
            },
            'bathroom': {
                'min_x': 5.0, 'max_x': 8.0,
                'min_y': 0.0, 'max_y': 3.0,
                'min_z': 8.0, 'max_z': 11.0
            }
        }
        
        for room_name, bounds in test_rooms.items():
            self.smart_home.add_room(room_name, bounds)
    
    def test_entrance_scenario(self):
        """Test: Person entering through entryway"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Person Entering Through Front Door")
        print("="*80)
        print("\nThis simulates someone entering the house and moving to living room.\n")
        
        path = [
            (np.array([1.0, 1.5, 1.0]), "entryway", "Opening front door"),
            (np.array([1.5, 1.5, 1.5]), "entryway", "Standing in entryway"),
            (np.array([2.5, 1.5, 2.0]), "entryway", "Moving through entryway"),
            (np.array([4.0, 1.5, 3.0]), "living_room", "Entering living room"),
            (np.array([5.5, 1.5, 3.0]), "living_room", "Walking into living room"),
        ]
        
        for position, room, description in path:
            print(f"⏱️  {description}")
            event = self.smart_home.update_person_location(position, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_cross_room_scenario(self):
        """Test: Person moving through multiple rooms"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Cross-Room Movement")
        print("="*80)
        print("\nThis simulates someone walking through multiple connected rooms.\n")
        
        path = [
            (np.array([5.5, 1.5, 3.0]), "living_room", "Starting in living room"),
            (np.array([7.5, 1.5, 3.0]), "living_room", "Walking across living room"),
            (np.array([9.0, 1.5, 2.0]), "kitchen", "Entering kitchen"),
            (np.array([10.0, 1.5, 2.0]), "kitchen", "Moving in kitchen"),
            (np.array([6.5, 1.5, 7.0]), "hallway", "Entering hallway"),
            (np.array([2.5, 1.5, 10.0]), "bedroom", "Entering bedroom"),
        ]
        
        for position, room, description in path:
            print(f"⏱️  {description}")
            event = self.smart_home.update_person_location(position, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_stationary_scenario(self):
        """Test: Person staying in one location"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Stationary Person")
        print("="*80)
        print("\nThis tests how sensors respond when person stays in one place.\n")
        
        position = np.array([5.5, 1.5, 3.0])
        room = "living_room"
        
        for i in range(5):
            # Add small random jitter to simulate natural movement
            jittered_pos = position + np.random.uniform(-0.1, 0.1, 3)
            print(f"⏱️  Time {i+1}: Person stationary in living room (minor movement)")
            event = self.smart_home.update_person_location(jittered_pos, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_boundary_scenario(self):
        """Test: Person at room boundaries"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Room Boundary Detection")
        print("="*80)
        print("\nThis tests sensor triggering at room boundaries.\n")
        
        # Position exactly at the boundary between living room and kitchen
        boundary_positions = [
            (np.array([7.8, 1.5, 2.0]), "living_room", "Near living room/kitchen boundary (living room side)"),
            (np.array([8.0, 1.5, 2.0]), "kitchen", "Exactly at living room/kitchen boundary"),
            (np.array([8.2, 1.5, 2.0]), "kitchen", "Just inside kitchen"),
            (np.array([7.5, 1.5, 5.8]), "living_room", "Near living room/hallway boundary"),
            (np.array([5.5, 1.5, 6.0]), "hallway", "At hallway entrance"),
        ]
        
        for position, room, description in boundary_positions:
            print(f"⏱️  {description}")
            event = self.smart_home.update_person_location(position, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_corner_detection_scenario(self):
        """Test: Person in room corners"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Corner Detection")
        print("="*80)
        print("\nThis tests how corner sensors detect person in different corners.\n")
        
        # Test each corner of the living room
        living_room_corners = [
            (np.array([3.5, 1.5, 0.5]), "living_room", "Southwest corner (corner A zone)"),
            (np.array([7.5, 1.5, 0.5]), "living_room", "Southeast corner (corner B zone)"),
            (np.array([5.5, 1.5, 3.0]), "living_room", "Center of room"),
            (np.array([3.5, 1.5, 5.5]), "living_room", "Northwest corner"),
            (np.array([7.5, 1.5, 5.5]), "living_room", "Northeast corner"),
        ]
        
        for position, room, description in living_room_corners:
            print(f"⏱️  {description}")
            event = self.smart_home.update_person_location(position, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_rapid_movement_scenario(self):
        """Test: Rapid movement through house"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Rapid Movement")
        print("="*80)
        print("\nThis simulates someone quickly moving through the house.\n")
        
        rapid_path = [
            (np.array([1.5, 1.5, 1.5]), "entryway", "Rush through entryway"),
            (np.array([5.5, 1.5, 3.0]), "living_room", "Quick pass through living room"),
            (np.array([10.0, 1.5, 2.0]), "kitchen", "Brief stop in kitchen"),
            (np.array([5.5, 1.5, 7.0]), "hallway", "Rush through hallway"),
            (np.array([2.5, 1.5, 10.0]), "bedroom", "Arrive in bedroom"),
        ]
        
        for position, room, description in rapid_path:
            print(f"⏱️  {description}")
            event = self.smart_home.update_person_location(position, room)
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def test_edge_detection_scenario(self):
        """Test: Person just outside sensor range"""
        print("\n" + "="*80)
        print("TEST SCENARIO: Edge Detection (Testing Sensor Range)")
        print("="*80)
        print("\nThis tests the detection radius limits of sensors.\n")
        
        # Position person at various distances from a sensor
        # Living room center sensor is at approximately [5.5, 1.5, 3.0]
        center_sensor_pos = np.array([5.5, 1.5, 3.0])
        
        test_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        
        for distance in test_distances:
            # Position to the right of center sensor
            position = center_sensor_pos + np.array([distance, 0, 0])
            print(f"⏱️  Person {distance:.1f}m from center sensor")
            event = self.smart_home.update_person_location(position, "living_room")
            self._print_event_summary(event)
        
        self._print_final_status()
    
    def _print_event_summary(self, event: dict):
        """Print summary of a detection event"""
        if event['sensor_count'] > 0:
            print(f"   🔔 {event['sensor_count']} sensor(s) triggered")
        else:
            print(f"   ⚪ No sensors triggered")
    
    def _print_final_status(self):
        """Print final status and statistics"""
        print(self.smart_home.get_system_status())
        
        summary = self.smart_home.get_detection_summary()
        print(f"\n📊 TEST RESULTS")
        print(f"   Total events: {summary['total_events']}")
        print(f"   Events with detections: {summary['events_with_detections']}")
        print(f"   Detection rate: {summary['detection_rate']:.1%}")
        
        if summary.get('sensor_trigger_counts'):
            print(f"\n   Top 5 Most Active Sensors:")
            sorted_sensors = sorted(
                summary['sensor_trigger_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (sensor_id, count) in enumerate(sorted_sensors[:5], 1):
                print(f"      {i}. {sensor_id}: {count} triggers")
        print()


def list_scenarios():
    """List all available test scenarios"""
    scenarios = {
        'entrance': 'Person entering through front door',
        'cross-room': 'Person moving through multiple rooms',
        'stationary': 'Person staying in one location',
        'boundary': 'Person at room boundaries',
        'corner': 'Person in different corners of room',
        'rapid': 'Rapid movement through house',
        'edge': 'Testing sensor detection range limits',
    }
    
    print("\n" + "="*80)
    print("AVAILABLE TEST SCENARIOS")
    print("="*80)
    for scenario, description in scenarios.items():
        print(f"  {scenario:15s} - {description}")
    print("\nUsage: python test_motion_sensors.py --scenario <scenario_name>")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test suite for smart home motion sensor system'
    )
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['entrance', 'cross-room', 'stationary', 'boundary',
                               'corner', 'rapid', 'edge', 'all'],
                       help='Test scenario to run')
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List all available scenarios')
    
    args = parser.parse_args()
    
    if args.list_scenarios:
        list_scenarios()
        return 0
    
    print("\n🏠 Smart Home Motion Sensor Test Suite")
    print("="*80)
    
    tester = MotionSensorTester()
    
    # Run selected scenario(s)
    scenario_map = {
        'entrance': tester.test_entrance_scenario,
        'cross-room': tester.test_cross_room_scenario,
        'stationary': tester.test_stationary_scenario,
        'boundary': tester.test_boundary_scenario,
        'corner': tester.test_corner_detection_scenario,
        'rapid': tester.test_rapid_movement_scenario,
        'edge': tester.test_edge_detection_scenario,
    }
    
    if args.scenario == 'all':
        # Run all scenarios
        for scenario_name, scenario_func in scenario_map.items():
            scenario_func()
            print("\n" + "="*80 + "\n")
            input("Press Enter to continue to next scenario...")
    else:
        # Run specific scenario
        if args.scenario in scenario_map:
            scenario_map[args.scenario]()
        else:
            print(f"Unknown scenario: {args.scenario}")
            return 1
    
    print("\n✅ Test suite complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
