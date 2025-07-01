import threading
from collections import defaultdict
from datetime import datetime

class ParkingState:
    """
    Enhanced parking state management with:
    - Thread-safe operations
    - Per-zone occupancy tracking
    - Vehicle-level timestamps
    - Historical event logging
    - Capacity enforcement
    """
    
    def __init__(self, initial_capacity=50):
        self.lock = threading.Lock()
        self.occupancy = defaultdict(int)  # {zone_id: count}
        self.vehicle_history = defaultdict(dict)  # {plate: {zone: timestamp}}
        self.capacity = defaultdict(lambda: initial_capacity)
        self.last_events = {}  # {zone_id: {event_type: str, timestamp: datetime}}
        
    def update_occupancy(self, plate: str, zone_id: str, event_type: str):
        """
        Update occupancy based on vehicle entry/exit
        Args:
            plate: License plate number
            zone_id: Parking zone identifier
            event_type: 'entry' or 'exit'
        """
        with self.lock:
            current_count = self.occupancy[zone_id]
            capacity = self.capacity[zone_id]
            
            if event_type == 'entry':
                if current_count >= capacity:
                    raise ValueError(f"Zone {zone_id} at full capacity")
                self.occupancy[zone_id] += 1
                # Track vehicle entry
                self.vehicle_history[plate][zone_id] = datetime.utcnow()
                
            elif event_type == 'exit':
                if current_count <= 0:
                    raise ValueError(f"Zone {zone_id} already empty")
                self.occupancy[zone_id] -= 1
                # Remove vehicle tracking
                if plate in self.vehicle_history and zone_id in self.vehicle_history[plate]:
                    del self.vehicle_history[plate][zone_id]
                    
            # Record last event
            self.last_events[zone_id] = {
                "event_type": event_type,
                "timestamp": datetime.utcnow(),
                "plate": plate
            }

    def set_capacity(self, zone_id: str, capacity: int):
        """Set maximum capacity for a zone"""
        with self.lock:
            if capacity < 0:
                raise ValueError("Capacity cannot be negative")
            current = self.occupancy[zone_id]
            if current > capacity:
                raise ValueError(f"Cannot set capacity below current occupancy ({current} vehicles)")
            self.capacity[zone_id] = capacity

    def get_occupancy(self, zone_id: str = None) -> dict:
        """Get current occupancy counts"""
        with self.lock:
            if zone_id:
                return {
                    "zone": zone_id,
                    "occupied": self.occupancy.get(zone_id, 0),
                    "capacity": self.capacity.get(zone_id, 0),
                    "available": self.capacity[zone_id] - self.occupancy.get(zone_id, 0),
                    "last_event": self.last_events.get(zone_id)
                }
            return {
                zone: {
                    "occupied": count,
                    "capacity": self.capacity[zone],
                    "available": self.capacity[zone] - count,
                    "last_event": self.last_events.get(zone)
                } for zone, count in self.occupancy.items()
            }

    def get_vehicle_history(self, plate: str) -> dict:
        """Get parking history for a specific vehicle"""
        with self.lock:
            return self.vehicle_history.get(plate, {})
    
    def reset_zone(self, zone_id: str):
        """Reset occupancy for a zone (for maintenance/testing)"""
        with self.lock:
            self.occupancy[zone_id] = 0
            # Clear vehicle history for this zone
            for plate in list(self.vehicle_history.keys()):
                if zone_id in self.vehicle_history[plate]:
                    del self.vehicle_history[plate][zone_id]
                    if not self.vehicle_history[plate]:
                        del self.vehicle_history[plate]

# Global state instance
parking_state = ParkingState()
