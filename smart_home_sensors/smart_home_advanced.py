#!/usr/bin/env python3
"""
Advanced Smart Home Motion Sensor System with Visualization and Analytics

This module extends the basic motion sensor system with:
- Real-time sensor visualization
- Heat maps of activity zones
- Multi-person tracking
- Zone-based automation triggers
- Sensor network topology analysis
- Activity pattern recognition

Usage:
    python smart_home_advanced.py --visualize
    python smart_home_advanced.py --track-multiple-people
    python smart_home_advanced.py --analyze-patterns
"""

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import time


@dataclass
class Person:
    """Represents a person being tracked in the smart home"""
    person_id: str
    current_position: np.ndarray
    current_room: Optional[str] = None
    path_history: List[Tuple[float, np.ndarray, str]] = field(default_factory=list)
    total_distance_traveled: float = 0.0
    rooms_visited: Set[str] = field(default_factory=set)
    
    def update_position(self, new_position: np.ndarray, room: str, timestamp: float):
        """Update person's position and track movement"""
        if len(self.path_history) > 0:
            distance = np.linalg.norm(new_position - self.current_position)
            self.total_distance_traveled += distance
        
        self.path_history.append((timestamp, new_position.copy(), room))
        self.current_position = new_position
        self.current_room = room
        self.rooms_visited.add(room)


@dataclass
class SensorEvent:
    """Detailed sensor event with context"""
    timestamp: float
    sensor_id: str
    room_name: str
    zone: str
    event_type: str  # "triggered", "cleared", "timeout"
    person_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


class ActivityZoneAnalyzer:
    """Analyzes activity patterns across different zones"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.zone_activities: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.room_dwell_times: Dict[str, List[float]] = defaultdict(list)
        self.transition_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self.last_room: Optional[str] = None
    
    def record_activity(self, room: str, zone: str, timestamp: float, duration: float = 1.0):
        """Record activity in a specific zone"""
        key = f"{room}_{zone}"
        self.zone_activities[key].append((timestamp, duration))
        self.room_dwell_times[room].append(duration)
        
        # Track room transitions
        if self.last_room and self.last_room != room:
            self.transition_matrix[(self.last_room, room)] += 1
        self.last_room = room
    
    def get_hotspots(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Identify most active zones"""
        zone_scores = {}
        for zone, activities in self.zone_activities.items():
            total_duration = sum(duration for _, duration in activities)
            zone_scores[zone] = total_duration
        
        sorted_zones = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_zones[:top_n]
    
    def get_room_transition_patterns(self) -> List[Tuple[Tuple[str, str], int]]:
        """Get common room transition patterns"""
        sorted_transitions = sorted(
            self.transition_matrix.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_transitions[:10]
    
    def get_average_dwell_time(self, room: str) -> float:
        """Get average time spent in a room"""
        if room not in self.room_dwell_times or not self.room_dwell_times[room]:
            return 0.0
        return np.mean(self.room_dwell_times[room])
    
    def generate_activity_report(self) -> str:
        """Generate comprehensive activity report"""
        lines = [
            "\n" + "="*80,
            "📊 ACTIVITY ZONE ANALYSIS REPORT",
            "="*80,
            ""
        ]
        
        # Hotspots
        hotspots = self.get_hotspots()
        if hotspots:
            lines.append("🔥 TOP ACTIVITY HOTSPOTS:")
            for i, (zone, score) in enumerate(hotspots, 1):
                lines.append(f"   {i}. {zone}: {score:.1f} activity units")
            lines.append("")
        
        # Room transitions
        transitions = self.get_room_transition_patterns()
        if transitions:
            lines.append("🚪 COMMON ROOM TRANSITIONS:")
            for (from_room, to_room), count in transitions[:5]:
                lines.append(f"   {from_room} → {to_room}: {count} times")
            lines.append("")
        
        # Dwell times
        if self.room_dwell_times:
            lines.append("⏱️  AVERAGE DWELL TIMES BY ROOM:")
            for room in sorted(self.room_dwell_times.keys()):
                avg_time = self.get_average_dwell_time(room)
                lines.append(f"   {room}: {avg_time:.2f}s")
            lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)


class AutomationController:
    """Controls automated actions based on sensor triggers"""
    
    def __init__(self):
        self.rules: List[Dict] = []
        self.triggered_automations: List[Dict] = []
        self.automation_enabled = True
    
    def add_rule(self, rule_name: str, condition: callable, action: callable, 
                 description: str = ""):
        """Add an automation rule"""
        rule = {
            'name': rule_name,
            'condition': condition,
            'action': action,
            'description': description,
            'trigger_count': 0
        }
        self.rules.append(rule)
        print(f"✓ Added automation rule: {rule_name}")
    
    def evaluate_rules(self, context: Dict) -> List[str]:
        """Evaluate all rules and trigger actions"""
        triggered = []
        
        if not self.automation_enabled:
            return triggered
        
        for rule in self.rules:
            try:
                if rule['condition'](context):
                    rule['action'](context)
                    rule['trigger_count'] += 1
                    triggered.append(rule['name'])
                    
                    self.triggered_automations.append({
                        'timestamp': context.get('timestamp', 0),
                        'rule': rule['name'],
                        'context': context.copy()
                    })
            except Exception as e:
                print(f"⚠️  Error evaluating rule {rule['name']}: {e}")
        
        return triggered
    
    def get_automation_summary(self) -> str:
        """Get summary of automation activity"""
        lines = [
            "\n" + "="*80,
            "🤖 AUTOMATION CONTROLLER SUMMARY",
            "="*80,
            f"Status: {'ENABLED' if self.automation_enabled else 'DISABLED'}",
            f"Total Rules: {len(self.rules)}",
            f"Total Triggers: {sum(r['trigger_count'] for r in self.rules)}",
            ""
        ]
        
        if self.rules:
            lines.append("AUTOMATION RULES:")
            for rule in self.rules:
                status = "✓" if rule['trigger_count'] > 0 else "○"
                lines.append(f"   {status} {rule['name']}: {rule['trigger_count']} triggers")
                if rule['description']:
                    lines.append(f"      {rule['description']}")
            lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)


class AdvancedSmartHome:
    """Advanced smart home system with multiple tracking and analysis features"""
    
    def __init__(self, enable_analytics: bool = True):
        self.rooms: Dict[str, Dict] = {}
        self.sensors: Dict[str, Dict] = {}  # sensor_id -> sensor data
        self.people: Dict[str, Person] = {}
        self.events: List[SensorEvent] = []
        self.current_time: float = 0.0
        
        # Advanced features
        self.enable_analytics = enable_analytics
        self.activity_analyzer = ActivityZoneAnalyzer() if enable_analytics else None
        self.automation_controller = AutomationController()
        
        # Performance metrics
        self.total_detections = 0
        self.false_positives = 0
        self.sensor_uptime: Dict[str, float] = {}
        
        self._setup_default_automations()
    
    def _setup_default_automations(self):
        """Setup some example automation rules"""
        
        # Rule 1: Turn on lights when motion detected
        def lights_condition(ctx):
            return ctx.get('sensor_count', 0) > 0
        
        def lights_action(ctx):
            room = ctx.get('room', 'unknown')
            print(f"   💡 [AUTOMATION] Turning on lights in {room}")
        
        self.automation_controller.add_rule(
            "lights_on_motion",
            lights_condition,
            lights_action,
            "Turn on lights when motion is detected"
        )
        
        # Rule 2: Alert if person in kitchen for extended period
        def kitchen_alert_condition(ctx):
            room = ctx.get('room', '')
            duration = ctx.get('duration', 0)
            return 'kitchen' in room.lower() and duration > 5.0
        
        def kitchen_alert_action(ctx):
            print(f"   🔔 [AUTOMATION] Alert: Extended activity in kitchen")
        
        self.automation_controller.add_rule(
            "kitchen_extended_activity",
            kitchen_alert_condition,
            kitchen_alert_action,
            "Alert when person stays in kitchen > 5 seconds"
        )
        
        # Rule 3: Energy saving - turn off lights when no motion
        def energy_save_condition(ctx):
            return ctx.get('sensor_count', 0) == 0 and ctx.get('last_motion', 999) > 3.0
        
        def energy_save_action(ctx):
            room = ctx.get('room', 'unknown')
            print(f"   💡 [AUTOMATION] Energy save: Turning off lights in {room}")
        
        self.automation_controller.add_rule(
            "energy_save_mode",
            energy_save_condition,
            energy_save_action,
            "Turn off lights after 3s of no motion"
        )
    
    def add_room_with_sensors(self, room_name: str, bounds: Dict[str, float], 
                              num_sensors: int = 3):
        """Add a room and create sensor network"""
        self.rooms[room_name] = {
            'bounds': bounds,
            'sensor_ids': [],
            'last_activity': 0.0,
            'total_activity_time': 0.0
        }
        
        # Create sensors
        center_x = (bounds['min_x'] + bounds['max_x']) / 2
        center_y = (bounds['min_y'] + bounds['max_y']) / 2
        center_z = (bounds['min_z'] + bounds['max_z']) / 2
        
        x_range = bounds['max_x'] - bounds['min_x']
        z_range = bounds['max_z'] - bounds['min_z']
        
        zones = ['corner_a', 'corner_b', 'center']
        positions = [
            np.array([bounds['min_x'] + x_range * 0.25, center_y, bounds['min_z'] + z_range * 0.25]),
            np.array([bounds['max_x'] - x_range * 0.25, center_y, bounds['min_z'] + z_range * 0.25]),
            np.array([center_x, center_y, center_z])
        ]
        
        for i in range(num_sensors):
            sensor_id = f"{room_name}_sensor_{i+1}"
            self.sensors[sensor_id] = {
                'id': sensor_id,
                'room': room_name,
                'zone': zones[i] if i < len(zones) else f'zone_{i}',
                'position': positions[i] if i < len(positions) else positions[-1],
                'radius': 2.5,
                'active': False,
                'trigger_count': 0,
                'last_trigger': 0.0
            }
            self.rooms[room_name]['sensor_ids'].append(sensor_id)
            self.sensor_uptime[sensor_id] = 0.0
    
    def add_person(self, person_id: str, initial_position: np.ndarray, room: str):
        """Add a person to track"""
        self.people[person_id] = Person(
            person_id=person_id,
            current_position=initial_position,
            current_room=room
        )
        print(f"✓ Now tracking person: {person_id}")
    
    def update_person(self, person_id: str, new_position: np.ndarray, room: str):
        """Update person position and trigger sensors"""
        if person_id not in self.people:
            self.add_person(person_id, new_position, room)
            return
        
        person = self.people[person_id]
        person.update_position(new_position, room, self.current_time)
        
        # Check all sensors in the room
        triggered_sensors = []
        if room in self.rooms:
            for sensor_id in self.rooms[room]['sensor_ids']:
                sensor = self.sensors[sensor_id]
                distance = np.linalg.norm(sensor['position'] - new_position)
                
                if distance <= sensor['radius']:
                    sensor['active'] = True
                    sensor['trigger_count'] += 1
                    sensor['last_trigger'] = self.current_time
                    triggered_sensors.append(sensor_id)
                    
                    # Create event
                    event = SensorEvent(
                        timestamp=self.current_time,
                        sensor_id=sensor_id,
                        room_name=room,
                        zone=sensor['zone'],
                        event_type='triggered',
                        person_id=person_id,
                        metadata={'distance': distance}
                    )
                    self.events.append(event)
                    self.total_detections += 1
                else:
                    sensor['active'] = False
            
            # Update room activity
            self.rooms[room]['last_activity'] = self.current_time
            
            # Record in activity analyzer
            if self.activity_analyzer:
                for sensor_id in triggered_sensors:
                    sensor = self.sensors[sensor_id]
                    self.activity_analyzer.record_activity(
                        room, sensor['zone'], self.current_time
                    )
            
            # Evaluate automation rules
            context = {
                'timestamp': self.current_time,
                'room': room,
                'sensor_count': len(triggered_sensors),
                'triggered_sensors': triggered_sensors,
                'person_id': person_id,
                'position': new_position
            }
            self.automation_controller.evaluate_rules(context)
        
        self.current_time += 1.0
    
    def get_sensor_network_topology(self) -> str:
        """Visualize sensor network topology"""
        lines = [
            "\n" + "="*80,
            "🕸️  SENSOR NETWORK TOPOLOGY",
            "="*80,
            ""
        ]
        
        for room_name, room_data in self.rooms.items():
            lines.append(f"📍 {room_name}:")
            for sensor_id in room_data['sensor_ids']:
                sensor = self.sensors[sensor_id]
                status = "🟢" if sensor['active'] else "⚪"
                pos = sensor['position']
                lines.append(
                    f"   {status} {sensor_id} [{sensor['zone']}] "
                    f"at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                    f"- {sensor['trigger_count']} triggers"
                )
            lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)
    
    def get_person_tracking_summary(self) -> str:
        """Get summary of all tracked people"""
        lines = [
            "\n" + "="*80,
            "👥 PERSON TRACKING SUMMARY",
            "="*80,
            f"Total People Tracked: {len(self.people)}",
            ""
        ]
        
        for person_id, person in self.people.items():
            lines.append(f"Person: {person_id}")
            lines.append(f"   Current Room: {person.current_room}")
            lines.append(f"   Position: [{person.current_position[0]:.2f}, "
                        f"{person.current_position[1]:.2f}, "
                        f"{person.current_position[2]:.2f}]")
            lines.append(f"   Distance Traveled: {person.total_distance_traveled:.2f}m")
            lines.append(f"   Rooms Visited: {len(person.rooms_visited)}")
            lines.append(f"   Path Length: {len(person.path_history)} waypoints")
            lines.append("")
        
        lines.append("="*80)
        return "\n".join(lines)
    
    def get_system_dashboard(self) -> str:
        """Get comprehensive system dashboard"""
        lines = [
            "\n" + "="*80,
            "🏠 SMART HOME SYSTEM DASHBOARD",
            "="*80,
            f"⏰ Current Time: {self.current_time:.1f}s",
            f"🏘️  Total Rooms: {len(self.rooms)}",
            f"📡 Total Sensors: {len(self.sensors)}",
            f"👥 People Tracked: {len(self.people)}",
            f"🔔 Total Detections: {self.total_detections}",
            f"📝 Total Events: {len(self.events)}",
            "",
            "ROOM STATUS:",
        ]
        
        for room_name, room_data in self.rooms.items():
            active_sensors = sum(1 for sid in room_data['sensor_ids'] 
                               if self.sensors[sid]['active'])
            status = "🟢 ACTIVE" if active_sensors > 0 else "⚪ IDLE"
            lines.append(f"   {status} {room_name}: {active_sensors}/{len(room_data['sensor_ids'])} sensors")
        
        lines.append("")
        lines.append("="*80)
        return "\n".join(lines)


def demo_multi_person_tracking():
    """Demonstrate multi-person tracking"""
    print("\n🏠 Advanced Smart Home - Multi-Person Tracking Demo\n")
    
    smart_home = AdvancedSmartHome(enable_analytics=True)
    
    # Setup rooms
    rooms = {
        'living_room': {
            'min_x': 0.0, 'max_x': 6.0,
            'min_y': 0.0, 'max_y': 3.0,
            'min_z': 0.0, 'max_z': 5.0
        },
        'kitchen': {
            'min_x': 6.0, 'max_x': 10.0,
            'min_y': 0.0, 'max_y': 3.0,
            'min_z': 0.0, 'max_z': 4.0
        },
        'bedroom': {
            'min_x': 0.0, 'max_x': 5.0,
            'min_y': 0.0, 'max_y': 3.0,
            'min_z': 5.0, 'max_z': 9.0
        }
    }
    
    for room_name, bounds in rooms.items():
        smart_home.add_room_with_sensors(room_name, bounds)
    
    # Add multiple people
    smart_home.add_person('person_1', np.array([3.0, 1.5, 2.5]), 'living_room')
    smart_home.add_person('person_2', np.array([8.0, 1.5, 2.0]), 'kitchen')
    
    print(smart_home.get_sensor_network_topology())
    
    # Simulate movement
    print("\n🚶 Simulating multi-person movement...\n")
    
    movements = [
        ('person_1', np.array([3.0, 1.5, 2.5]), 'living_room', "Person 1 in living room"),
        ('person_2', np.array([8.0, 1.5, 2.0]), 'kitchen', "Person 2 in kitchen"),
        ('person_1', np.array([5.0, 1.5, 2.5]), 'living_room', "Person 1 moves"),
        ('person_2', np.array([7.5, 1.5, 2.5]), 'kitchen', "Person 2 moves"),
        ('person_1', np.array([7.0, 1.5, 2.0]), 'kitchen', "Person 1 enters kitchen"),
        ('person_2', np.array([3.0, 1.5, 2.5]), 'living_room', "Person 2 moves to living room"),
    ]
    
    for person_id, position, room, description in movements:
        print(f"⏱️  {description}")
        smart_home.update_person(person_id, position, room)
        time.sleep(0.1)  # Small delay for visual effect
    
    # Display results
    print(smart_home.get_system_dashboard())
    print(smart_home.get_person_tracking_summary())
    
    if smart_home.activity_analyzer:
        print(smart_home.activity_analyzer.generate_activity_report())
    
    print(smart_home.automation_controller.get_automation_summary())
    
    return smart_home


def demo_pattern_analysis():
    """Demonstrate activity pattern analysis"""
    print("\n🏠 Advanced Smart Home - Pattern Analysis Demo\n")
    
    smart_home = AdvancedSmartHome(enable_analytics=True)
    
    # Setup a typical home layout
    rooms = {
        'bedroom': {'min_x': 0, 'max_x': 4, 'min_y': 0, 'max_y': 3, 'min_z': 0, 'max_z': 4},
        'bathroom': {'min_x': 4, 'max_x': 6, 'min_y': 0, 'max_y': 3, 'min_z': 0, 'max_z': 3},
        'kitchen': {'min_x': 0, 'max_x': 5, 'min_y': 0, 'max_y': 3, 'min_z': 4, 'max_z': 8},
        'living_room': {'min_x': 5, 'max_x': 10, 'min_y': 0, 'max_y': 3, 'min_z': 4, 'max_z': 8},
    }
    
    for room_name, bounds in rooms.items():
        smart_home.add_room_with_sensors(room_name, bounds)
    
    # Simulate typical morning routine
    print("☀️  Simulating morning routine pattern...\n")
    
    routine = [
        ('bedroom', np.array([2, 1.5, 2]), "Wake up"),
        ('bedroom', np.array([2, 1.5, 2]), "Getting ready"),
        ('bathroom', np.array([5, 1.5, 1.5]), "Morning bathroom"),
        ('bathroom', np.array([5, 1.5, 1.5]), "Showering"),
        ('bedroom', np.array([2, 1.5, 2]), "Getting dressed"),
        ('kitchen', np.array([2.5, 1.5, 6]), "Making breakfast"),
        ('kitchen', np.array([2.5, 1.5, 6]), "Eating breakfast"),
        ('living_room', np.array([7.5, 1.5, 6]), "Reading news"),
        ('kitchen', np.array([2.5, 1.5, 6]), "Cleaning dishes"),
        ('bedroom', np.array([2, 1.5, 2]), "Getting ready to leave"),
    ]
    
    for room, position, activity in routine:
        print(f"   {activity} in {room}")
        smart_home.update_person('person_1', position, room)
    
    # Display analysis
    print(smart_home.get_system_dashboard())
    
    if smart_home.activity_analyzer:
        print(smart_home.activity_analyzer.generate_activity_report())
    
    print(smart_home.automation_controller.get_automation_summary())
    
    return smart_home


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced Smart Home Motion Sensor System'
    )
    parser.add_argument('--track-multiple-people', action='store_true',
                       help='Demo multi-person tracking')
    parser.add_argument('--analyze-patterns', action='store_true',
                       help='Demo activity pattern analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization features')
    
    args = parser.parse_args()
    
    if args.track_multiple_people:
        smart_home = demo_multi_person_tracking()
    elif args.analyze_patterns:
        smart_home = demo_pattern_analysis()
    else:
        # Run both demos
        print("Running all demos...\n")
        demo_multi_person_tracking()
        print("\n" + "="*80 + "\n")
        demo_pattern_analysis()
    
    print("\n✅ Advanced simulation complete!")


if __name__ == "__main__":
    main()
