# drone_coordination_system.py
# Main system implementation for FlytBase Drone Coordination System

import time
import math
import uuid
import heapq
import logging
import threading
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DroneCoordinationSystem")

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Layer(Enum):
    """Airspace vertical layer definitions"""
    DOCK = 0       # 0-2m: Takeoff/Landing
    TRANSITION = 1 # 2-4m: Transition
    TRANSIT_1 = 2  # 4-6m: Standard Transit (N/S primary)
    TRANSIT_2 = 3  # 6-8m: High-Speed Transit (E/W primary)
    BEYOND = 4     # 8m+: Beyond grid operations

class Quadrant(Enum):
    """Airspace quadrant definitions"""
    Q1 = 1  # NW quadrant
    Q2 = 2  # NE quadrant
    Q3 = 3  # SW quadrant
    Q4 = 4  # SE quadrant

class Status(Enum):
    """Drone status definitions"""
    DOCKED = 0
    READY = 1
    TAKEOFF = 2
    TRANSITION = 3
    TRANSIT = 4
    LANDING = 5
    EMERGENCY = 6
    MALFUNCTION = 7

class Priority(Enum):
    """Mission priority definitions"""
    LOW = 0      # Standard delivery
    MEDIUM = 1   # Time-sensitive delivery
    HIGH = 2     # Surveillance/monitoring
    URGENT = 3   # Emergency response
    CRITICAL = 4 # System critical operations

class ReservationStatus(Enum):
    """Path reservation status"""
    PENDING = 0
    APPROVED = 1
    DENIED = 2
    MODIFIED = 3
    CANCELLED = 4
    COMPLETED = 5

@dataclass
class Position:
    """3D position in the airspace"""
    x: float  # X coordinate (0-4)
    y: float  # Y coordinate (0-4)
    z: float  # Z coordinate (altitude in meters)
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + 
                         (self.y - other.y)**2 + 
                         (self.z - other.z)**2)
    
    def horizontal_distance_to(self, other: 'Position') -> float:
        """Calculate horizontal distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Waypoint:
    """Flight path waypoint"""
    position: Position
    speed: float      # m/s
    hold_time: float  # seconds to hold at this waypoint (0 for fly-through)
    
    def estimated_time(self, from_pos: Position, current_speed: float) -> float:
        """Estimate time to reach this waypoint from a position at current speed"""
        distance = from_pos.distance_to(self.position)
        avg_speed = (current_speed + self.speed) / 2
        return distance / avg_speed if avg_speed > 0 else float('inf')

@dataclass
class PathSegment:
    """Segment of a flight path between waypoints"""
    start: Waypoint
    end: Waypoint
    start_time: float  # Estimated start time (seconds from reservation)
    end_time: float    # Estimated end time (seconds from reservation)
    layer: Layer       # Assigned airspace layer
    
    def contains_time(self, time_point: float) -> bool:
        """Check if this segment is active at the given time"""
        return self.start_time <= time_point <= self.end_time
    
    def position_at_time(self, time_point: float) -> Position:
        """Interpolate position at a given time"""
        if time_point <= self.start_time:
            return self.start.position
        if time_point >= self.end_time:
            return self.end.position
        
        # Linear interpolation
        progress = (time_point - self.start_time) / (self.end_time - self.start_time)
        return Position(
            x=self.start.position.x + progress * (self.end.position.x - self.start.position.x),
            y=self.start.position.y + progress * (self.end.position.y - self.start.position.y),
            z=self.start.position.z + progress * (self.end.position.z - self.start.position.z)
        )

@dataclass
class FlightPath:
    """Complete flight path for a drone"""
    waypoints: List[Waypoint]
    segments: List[PathSegment]
    total_distance: float
    estimated_duration: float
    start_time: float
    end_time: float

@dataclass
class PathReservation:
    """Path reservation in the airspace"""
    id: str
    drone_id: str
    path: FlightPath
    priority: Priority
    status: ReservationStatus
    request_time: float
    expiry_time: float  # When this reservation expires if not used
    
    def is_active_at(self, time_point: float) -> bool:
        """Check if this reservation is active at the given time"""
        return (self.status == ReservationStatus.APPROVED and
                self.path.start_time <= time_point <= self.path.end_time)

@dataclass
class Drone:
    """Drone state and capabilities"""
    id: str
    position: Position
    velocity: Tuple[float, float, float]  # vx, vy, vz in m/s
    status: Status
    battery_level: float  # 0.0 to 1.0
    max_speed: float      # m/s
    current_reservation_id: Optional[str]
    current_waypoint_index: int
    home_dock_id: str
    last_update_time: float
    priority: Priority
    
    def update_position(self, dt: float) -> None:
        """Update position based on velocity and time delta"""
        self.position.x += self.velocity[0] * dt
        self.position.y += self.velocity[1] * dt
        self.position.z += self.velocity[2] * dt
        self.last_update_time = time.time()
    
    def get_quadrant(self) -> Quadrant:
        """Determine which quadrant the drone is currently in"""
        if self.position.x < 2:
            return Quadrant.Q1 if self.position.y < 2 else Quadrant.Q3
        else:
            return Quadrant.Q2 if self.position.y < 2 else Quadrant.Q4
    
    def get_layer(self) -> Layer:
        """Determine which layer the drone is currently in"""
        if self.position.z < 2:
            return Layer.DOCK
        elif self.position.z < 4:
            return Layer.TRANSITION
        elif self.position.z < 6:
            return Layer.TRANSIT_1
        elif self.position.z < 8:
            return Layer.TRANSIT_2
        else:
            return Layer.BEYOND
    
    def estimate_battery_usage(self, path: FlightPath) -> float:
        """Estimate battery percentage used for a given path"""
        # Simple model: 0.05% per meter of distance + 0.02% per second of flight
        distance_cost = 0.0005 * path.total_distance
        time_cost = 0.0002 * path.estimated_duration
        return distance_cost + time_cost

    def has_sufficient_battery(self, path: FlightPath) -> bool:
        """Check if drone has enough battery for path plus 20% safety margin"""
        required = self.estimate_battery_usage(path) * 1.2
        return self.battery_level >= required

@dataclass
class Dock:
    """Drone dock information"""
    id: str
    position: Position  # Only x,y used; z is always 0
    status: str  # 'available', 'occupied', 'charging', 'offline'
    current_drone_id: Optional[str]
    
    def get_quadrant(self) -> Quadrant:
        """Determine which quadrant this dock is in"""
        if self.position.x < 2:
            return Quadrant.Q1 if self.position.y < 2 else Quadrant.Q3
        else:
            return Quadrant.Q2 if self.position.y < 2 else Quadrant.Q4

# =============================================================================
# CENTRAL TRAFFIC MANAGEMENT SYSTEM
# =============================================================================

class CentralTrafficManagementSystem:
    """Central system for coordinating all drone traffic"""
    
    def __init__(self):
        self.drones: Dict[str, Drone] = {}
        self.docks: Dict[str, Dock] = {}
        self.reservations: Dict[str, PathReservation] = {}
        self.active_reservations: Dict[str, PathReservation] = {}
        self.reservation_queue = []  # Priority queue for pending reservations
        
        # Safety settings
        self.min_horizontal_separation = 2.0  # meters
        self.min_vertical_separation = 1.5    # meters
        
        # Traffic density tracking
        self.layer_density = {layer: 0 for layer in Layer}
        self.quadrant_density = {quadrant: 0 for quadrant in Quadrant}
        
        # Initialize the 4x4 grid with 16 docks
        self._initialize_docks()
        
        # For conflict detection
        self.spatial_index = defaultdict(set)  # Grid-based spatial index
        self.cell_size = 0.5  # meters
        
        # System status
        self.system_start_time = time.time()
        self.is_running = True
        
        # Statistics
        self.stats = {
            "operations_completed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "emergency_events": 0,
            "avg_wait_time": 0.0,
            "path_efficiency": 0.0,
        }
        
        # Start background threads
        self._start_background_processes()
        
        logger.info("Central Traffic Management System initialized")
    
    def _initialize_docks(self) -> None:
        """Initialize the 16 docks in a 4x4 grid"""
        dock_positions = [
            # Row 1
            (0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5),
            # Row 2
            (0.5, 1.5), (1.5, 1.5), (2.5, 1.5), (3.5, 1.5),
            # Row 3
            (0.5, 2.5), (1.5, 2.5), (2.5, 2.5), (3.5, 2.5),
            # Row 4
            (0.5, 3.5), (1.5, 3.5), (2.5, 3.5), (3.5, 3.5),
        ]
        
        for i, pos in enumerate(dock_positions):
            dock_id = f"D{i+1}"
            self.docks[dock_id] = Dock(
                id=dock_id,
                position=Position(x=pos[0], y=pos[1], z=0.0),
                status="available",
                current_drone_id=None
            )
    
    def _start_background_processes(self) -> None:
        """Start background threads for system processes"""
        # Process reservations periodically
        threading.Thread(target=self._reservation_processor, daemon=True).start()
        
        # Update airspace density metrics
        threading.Thread(target=self._update_density_metrics, daemon=True).start()
        
        # Conflict detection
        threading.Thread(target=self._conflict_detection_loop, daemon=True).start()
        
        # Path prediction and optimization
        threading.Thread(target=self._path_optimization_loop, daemon=True).start()
        
        # Cleanup expired reservations
        threading.Thread(target=self._cleanup_expired_reservations, daemon=True).start()
    
    def register_drone(self, drone_id: str, home_dock_id: str, max_speed: float) -> bool:
        """Register a new drone with the system"""
        if drone_id in self.drones:
            logger.warning(f"Drone {drone_id} already registered")
            return False
        
        if home_dock_id not in self.docks:
            logger.error(f"Invalid dock ID: {home_dock_id}")
            return False
        
        dock = self.docks[home_dock_id]
        if dock.status != "available":
            logger.warning(f"Dock {home_dock_id} is not available")
            return False
        
        # Create the drone at the dock position
        self.drones[drone_id] = Drone(
            id=drone_id,
            position=Position(x=dock.position.x, y=dock.position.y, z=0.0),
            velocity=(0.0, 0.0, 0.0),
            status=Status.DOCKED,
            battery_level=1.0,  # Start with full battery
            max_speed=max_speed,
            current_reservation_id=None,
            current_waypoint_index=0,
            home_dock_id=home_dock_id,
            last_update_time=time.time(),
            priority=Priority.LOW
        )
        
        # Update dock status
        dock.status = "occupied"
        dock.current_drone_id = drone_id
        
        logger.info(f"Drone {drone_id} registered at dock {home_dock_id}")
        return True
    
    def unregister_drone(self, drone_id: str) -> bool:
        """Unregister a drone from the system"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        drone = self.drones[drone_id]
        
        # Check if drone is docked
        if drone.status != Status.DOCKED:
            logger.warning(f"Cannot unregister drone {drone_id} while in flight")
            return False
        
        # Update dock status
        dock_id = drone.home_dock_id
        if dock_id in self.docks:
            dock = self.docks[dock_id]
            if dock.current_drone_id == drone_id:
                dock.status = "available"
                dock.current_drone_id = None
        
        # Remove drone
        del self.drones[drone_id]
        
        logger.info(f"Drone {drone_id} unregistered")
        return True
    
    def update_drone_status(self, drone_id: str, 
                           position: Position, 
                           velocity: Tuple[float, float, float],
                           status: Status, 
                           battery_level: float) -> bool:
        """Update drone state information"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        drone = self.drones[drone_id]
        old_position = drone.position
        
        # Update drone state
        drone.position = position
        drone.velocity = velocity
        drone.status = status
        drone.battery_level = battery_level
        drone.last_update_time = time.time()
        
        # Update spatial index for collision detection
        self._update_spatial_index(drone_id, old_position, position)
        
        return True
    
    def request_takeoff(self, drone_id: str, destination_dock_id: str, 
                       priority: Priority = Priority.LOW) -> Tuple[bool, Optional[str]]:
        """Request takeoff and path to destination"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False, "Drone not found"
        
        if destination_dock_id not in self.docks:
            logger.warning(f"Destination dock {destination_dock_id} not found")
            return False, "Destination dock not found"
        
        drone = self.drones[drone_id]
        destination_dock = self.docks[destination_dock_id]
        
        # Check if drone is docked and ready
        if drone.status != Status.DOCKED:
            logger.warning(f"Drone {drone_id} is not docked")
            return False, "Drone is not docked"
        
        # Check if destination dock is available
        if destination_dock.status != "available" and destination_dock.current_drone_id != drone_id:
            logger.warning(f"Destination dock {destination_dock_id} is not available")
            return False, "Destination dock is not available"
        
        # Update drone priority
        drone.priority = priority
        
        # Generate flight path
        path = self._generate_flight_path(drone, destination_dock)
        if not path:
            logger.warning(f"Could not generate valid flight path for drone {drone_id}")
            return False, "Could not generate valid flight path"
        
        # Check battery level
        if not drone.has_sufficient_battery(path):
            logger.warning(f"Drone {drone_id} has insufficient battery for the flight")
            return False, "Insufficient battery for flight"
        
        # Create reservation
        reservation_id = str(uuid.uuid4())
        reservation = PathReservation(
            id=reservation_id,
            drone_id=drone_id,
            path=path,
            priority=priority,
            status=ReservationStatus.PENDING,
            request_time=time.time(),
            expiry_time=time.time() + 300  # 5 minute expiry
        )
        
        # Add to reservation system
        self.reservations[reservation_id] = reservation
        heapq.heappush(
            self.reservation_queue, 
            (-priority.value, reservation.request_time, reservation_id)
        )
        
        logger.info(f"Takeoff request for drone {drone_id} submitted with reservation {reservation_id}")
        return True, reservation_id
    
    def check_reservation_status(self, reservation_id: str) -> ReservationStatus:
        """Check the status of a path reservation"""
        if reservation_id not in self.reservations:
            logger.warning(f"Reservation {reservation_id} not found")
            return ReservationStatus.DENIED
        
        return self.reservations[reservation_id].status
    
    def initiate_takeoff(self, drone_id: str, reservation_id: str) -> bool:
        """Initiate takeoff sequence with approved reservation"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        if reservation_id not in self.reservations:
            logger.warning(f"Reservation {reservation_id} not found")
            return False
        
        reservation = self.reservations[reservation_id]
        if reservation.status != ReservationStatus.APPROVED:
            logger.warning(f"Reservation {reservation_id} is not approved")
            return False
        
        drone = self.drones[drone_id]
        if drone.status != Status.DOCKED and drone.status != Status.READY:
            logger.warning(f"Drone {drone_id} is not ready for takeoff")
            return False
        
        # Update dock status
        dock = self.docks[drone.home_dock_id]
        dock.status = "available"
        dock.current_drone_id = None
        
        # Update drone status
        drone.status = Status.LANDING
        drone.current_reservation_id = reservation_id
        drone.current_waypoint_index = 0
        
        # Move reservation to active
        self.active_reservations[reservation_id] = reservation
        
        logger.info(f"Landing initiated for drone {drone_id} with reservation {reservation_id}")
        return True
    
    def report_completed_landing(self, drone_id: str, dock_id: str) -> bool:
        """Report successful landing at dock"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        if dock_id not in self.docks:
            logger.warning(f"Dock {dock_id} not found")
            return False
        
        drone = self.drones[drone_id]
        dock = self.docks[dock_id]
        
        # Update drone status
        drone.status = Status.DOCKED
        drone.velocity = (0.0, 0.0, 0.0)
        drone.position = Position(x=dock.position.x, y=dock.position.y, z=0.0)
        
        # Update dock status
        dock.status = "occupied"
        dock.current_drone_id = drone_id
        
        # Clear reservation
        if drone.current_reservation_id:
            reservation_id = drone.current_reservation_id
            if reservation_id in self.active_reservations:
                reservation = self.active_reservations[reservation_id]
                reservation.status = ReservationStatus.COMPLETED
                del self.active_reservations[reservation_id]
            drone.current_reservation_id = None
        
        # Update statistics
        self.stats["operations_completed"] += 1
        
        logger.info(f"Drone {drone_id} successfully landed at dock {dock_id}")
        return True
    
    def report_emergency(self, drone_id: str, emergency_type: str) -> bool:
        """Report drone emergency situation"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        drone = self.drones[drone_id]
        old_status = drone.status
        
        # Update drone status
        drone.status = Status.EMERGENCY
        
        # Cancel any active reservation
        if drone.current_reservation_id:
            if drone.current_reservation_id in self.active_reservations:
                reservation = self.active_reservations[drone.current_reservation_id]
                reservation.status = ReservationStatus.CANCELLED
                del self.active_reservations[drone.current_reservation_id]
        
        # Update statistics
        self.stats["emergency_events"] += 1
        
        logger.warning(f"Emergency reported for drone {drone_id}: {emergency_type}")
        
        # Generate emergency path if drone is in flight
        if old_status != Status.DOCKED:
            # Find nearest available dock
            nearest_dock = self._find_nearest_available_dock(drone.position)
            if nearest_dock:
                # Generate emergency landing path
                path = self._generate_emergency_landing_path(drone, nearest_dock)
                if path:
                    # Create emergency reservation with highest priority
                    reservation_id = str(uuid.uuid4())
                    reservation = PathReservation(
                        id=reservation_id,
                        drone_id=drone_id,
                        path=path,
                        priority=Priority.CRITICAL,
                        status=ReservationStatus.APPROVED,  # Auto-approve emergency
                        request_time=time.time(),
                        expiry_time=time.time() + 60  # 1 minute expiry
                    )
                    
                    # Add to active reservations
                    self.reservations[reservation_id] = reservation
                    self.active_reservations[reservation_id] = reservation
                    
                    # Update drone
                    drone.current_reservation_id = reservation_id
                    drone.current_waypoint_index = 0
                    
                    logger.info(f"Emergency landing path generated for drone {drone_id} to dock {nearest_dock.id}")
        
        return True
    
    def _reservation_processor(self) -> None:
        """Background thread to process reservation queue"""
        while self.is_running:
            try:
                # Process up to 5 reservations per cycle
                for _ in range(5):
                    if not self.reservation_queue:
                        break
                    
                    # Get highest priority reservation
                    _, _, reservation_id = heapq.heappop(self.reservation_queue)
                    
                    # Check if reservation still exists and is pending
                    if (reservation_id not in self.reservations or 
                        self.reservations[reservation_id].status != ReservationStatus.PENDING):
                        continue
                    
                    reservation = self.reservations[reservation_id]
                    
                    # Process reservation
                    self._process_reservation(reservation)
                
                # Sleep for a short period
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error in reservation processor: {str(e)}")
                time.sleep(1)
    
    def _process_reservation(self, reservation: PathReservation) -> None:
        """Process a pending reservation"""
        # Check if reservation is expired
        if time.time() > reservation.expiry_time:
            reservation.status = ReservationStatus.CANCELLED
            logger.info(f"Reservation {reservation.id} expired")
            return
        
        # Check if drone still exists
        if reservation.drone_id not in self.drones:
            reservation.status = ReservationStatus.CANCELLED
            logger.info(f"Reservation {reservation.id} cancelled - drone not found")
            return
        
        # Check for conflicts with active reservations
        conflicts = self._detect_path_conflicts(reservation)
        
        if not conflicts:
            # No conflicts, approve the reservation
            reservation.status = ReservationStatus.APPROVED
            logger.info(f"Reservation {reservation.id} approved")
            return
        
        # Try to resolve conflicts by modifying the path
        modified_path = self._resolve_path_conflicts(reservation, conflicts)
        if modified_path:
            # Update reservation with modified path
            reservation.path = modified_path
            reservation.status = ReservationStatus.MODIFIED
            logger.info(f"Reservation {reservation.id} modified to resolve conflicts")
            
            # Re-check for conflicts with modified path
            conflicts = self._detect_path_conflicts(reservation)
            if not conflicts:
                reservation.status = ReservationStatus.APPROVED
                logger.info(f"Modified reservation {reservation.id} approved")
            else:
                # Still have conflicts, deny the reservation
                reservation.status = ReservationStatus.DENIED
                logger.info(f"Reservation {reservation.id} denied due to unresolvable conflicts")
        else:
            # Could not resolve conflicts, deny the reservation
            reservation.status = ReservationStatus.DENIED
            logger.info(f"Reservation {reservation.id} denied due to conflicts")
    
    def _detect_path_conflicts(self, reservation: PathReservation) -> List[Tuple[PathSegment, PathSegment]]:
        """Detect conflicts between a path reservation and active reservations"""
        conflicts = []
        
        # No conflicts if no active reservations
        if not self.active_reservations:
            return conflicts
        
        # Check each segment of the proposed path against active reservations
        for segment in reservation.path.segments:
            for active_id, active_reservation in self.active_reservations.items():
                # Skip if same drone
                if active_reservation.drone_id == reservation.drone_id:
                    continue
                
                # Check for temporal overlap
                for active_segment in active_reservation.path.segments:
                    if (segment.end_time < active_segment.start_time or 
                        segment.start_time > active_segment.end_time):
                        continue  # No temporal overlap
                    
                    # Check for spatial conflict
                    if self._segments_conflict(segment, active_segment):
                        conflicts.append((segment, active_segment))
        
        return conflicts
    
    def _segments_conflict(self, segment1: PathSegment, segment2: PathSegment) -> bool:
        """Check if two path segments have a spatial conflict"""
        # Check if segments are in different layers with sufficient vertical separation
        if segment1.layer != segment2.layer and abs(segment1.layer.value - segment2.layer.value) > 1:
            return False
        
        # Sample points along each segment and check for conflicts
        samples = 10
        for i in range(samples + 1):
            t1 = segment1.start_time + i * (segment1.end_time - segment1.start_time) / samples
            pos1 = segment1.position_at_time(t1)
            
            for j in range(samples + 1):
                t2 = segment2.start_time + j * (segment2.end_time - segment2.start_time) / samples
                pos2 = segment2.position_at_time(t2)
                
                # Check horizontal distance
                if pos1.horizontal_distance_to(pos2) < self.min_horizontal_separation:
                    # Check vertical distance if horizontally too close
                    if abs(pos1.z - pos2.z) < self.min_vertical_separation:
                        return True
        
        return False
    
    def _resolve_path_conflicts(self, reservation: PathReservation, 
                             conflicts: List[Tuple[PathSegment, PathSegment]]) -> Optional[FlightPath]:
        """Try to resolve path conflicts by modifying the path"""
        # Several strategies can be employed:
        # 1. Temporal separation: Delay the path start time
        # 2. Spatial separation: Change the layer
        # 3. Rerouting: Generate a new path
        
        # Try temporal separation first
        delayed_path = self._delay_path_start(reservation.path, conflicts)
        if delayed_path:
            return delayed_path
        
        # Try spatial separation next
        alt_layer_path = self._change_path_layers(reservation.path, conflicts)
        if alt_layer_path:
            return alt_layer_path
        
        # Finally, try rerouting
        drone = self.drones[reservation.drone_id]
        
        # Determine destination dock
        destination_dock_id = None
        last_waypoint = reservation.path.waypoints[-1]
        for dock_id, dock in self.docks.items():
            if (dock.position.x == last_waypoint.position.x and 
                dock.position.y == last_waypoint.position.y):
                destination_dock_id = dock_id
                break
        
        if destination_dock_id:
            destination_dock = self.docks[destination_dock_id]
            new_path = self._generate_alternative_path(drone, destination_dock, conflicts)
            if new_path:
                return new_path
        
        # Could not resolve conflicts
        return None
    
    def _delay_path_start(self, path: FlightPath, 
                        conflicts: List[Tuple[PathSegment, PathSegment]]) -> Optional[FlightPath]:
        """Attempt to delay path start time to resolve conflicts"""
        # Find the latest conflict end time
        latest_conflict_time = 0
        for segment1, segment2 in conflicts:
            latest_conflict_time = max(latest_conflict_time, segment2.end_time)
        
        # Add a safety buffer
        delay_until = latest_conflict_time + 2.0  # 2 second buffer
        
        # Calculate required delay
        delay = delay_until - path.start_time
        if delay > 30:  # Don't delay more than 30 seconds
            return None
        
        # Create a new path with delayed timing
        new_waypoints = path.waypoints.copy()
        new_segments = []
        
        # Shift all segment times
        for segment in path.segments:
            new_segment = PathSegment(
                start=segment.start,
                end=segment.end,
                start_time=segment.start_time + delay,
                end_time=segment.end_time + delay,
                layer=segment.layer
            )
            new_segments.append(new_segment)
        
        return FlightPath(
            waypoints=new_waypoints,
            segments=new_segments,
            total_distance=path.total_distance,
            estimated_duration=path.estimated_duration,
            start_time=path.start_time + delay,
            end_time=path.end_time + delay
        )
    
    def _change_path_layers(self, path: FlightPath, 
                          conflicts: List[Tuple[PathSegment, PathSegment]]) -> Optional[FlightPath]:
        """Attempt to change path layers to resolve conflicts"""
        # Collect all conflicting segments
        conflicting_segments = set()
        for segment1, _ in conflicts:
            conflicting_segments.add(segment1)
        
        # No conflicts, no changes needed
        if not conflicting_segments:
            return path
        
        # Copy path components
        new_waypoints = path.waypoints.copy()
        new_segments = []
        
        # Determine layer usage
        layer_usage = {layer: 0 for layer in [Layer.TRANSIT_1, Layer.TRANSIT_2]}
        for _, active_segment in conflicts:
            if active_segment.layer in layer_usage:
                layer_usage[active_segment.layer] += 1
        
        # Choose the less used transit layer
        target_layer = Layer.TRANSIT_1 if layer_usage[Layer.TRANSIT_1] <= layer_usage[Layer.TRANSIT_2] else Layer.TRANSIT_2
        
        # Modify segments
        for segment in path.segments:
            new_segment = segment
            
            # If this is a conflicting segment in transit layer, change it
            if segment in conflicting_segments and segment.layer in [Layer.TRANSIT_1, Layer.TRANSIT_2]:
                # Create a new segment with different layer
                new_segment = PathSegment(
                    start=segment.start,
                    end=segment.end,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    layer=target_layer
                )
                
                # Update waypoint altitudes
                z_value = 5.0 if target_layer == Layer.TRANSIT_1 else 7.0
                for waypoint_idx, waypoint in enumerate(new_waypoints):
                    if waypoint == segment.start or waypoint == segment.end:
                        new_waypoints[waypoint_idx] = Waypoint(
                            position=Position(x=waypoint.position.x, y=waypoint.position.y, z=z_value),
                            speed=waypoint.speed,
                            hold_time=waypoint.hold_time
                        )
            
            new_segments.append(new_segment)
        
        return FlightPath(
            waypoints=new_waypoints,
            segments=new_segments,
            total_distance=path.total_distance,
            estimated_duration=path.estimated_duration,
            start_time=path.start_time,
            end_time=path.end_time
        )
    
    def _generate_alternative_path(self, drone: Drone, destination_dock: Dock,
                              conflicts: List[Tuple[PathSegment, PathSegment]]) -> Optional[FlightPath]:
        """Generate an alternative path avoiding conflict areas"""
        # This is a more complex implementation using A* with additional constraints
        # For now, we'll create a simple path with a detour
        
        # Extract conflict zones
        conflict_zones = set()
        for segment1, _ in conflicts:
            # Convert segment to grid cells
            for i in range(int(segment1.start.position.x * 2), int(segment1.end.position.x * 2) + 1):
                for j in range(int(segment1.start.position.y * 2), int(segment1.end.position.y * 2) + 1):
                    conflict_zones.add((i/2, j/2))
        
        # Use A* to find a path avoiding conflict zones
        return self._generate_path_a_star(drone, destination_dock, conflict_zones)
    
    def _generate_flight_path(self, drone: Drone, destination_dock: Dock) -> Optional[FlightPath]:
        """Generate a flight path from drone position to destination dock"""
        # Use A* path finding with no specific constraints
        return self._generate_path_a_star(drone, destination_dock, set())
    
    def _generate_landing_path(self, drone: Drone, dock: Dock) -> Optional[FlightPath]:
        """Generate a landing path from current position to dock"""
        # For landing, we prioritize a direct approach
        return self._generate_path_a_star(drone, dock, set(), is_landing=True)
    
    def _generate_emergency_landing_path(self, drone: Drone, dock: Dock) -> Optional[FlightPath]:
        """Generate an emergency landing path to nearest available dock"""
        # Emergency paths are direct and override other constraints
        return self._generate_path_a_star(drone, dock, set(), is_emergency=True)
    
    def _generate_path_a_star(self, drone: Drone, destination_dock: Dock, 
                           avoid_zones: Set[Tuple[float, float]],
                           is_landing: bool = False,
                           is_emergency: bool = False) -> Optional[FlightPath]:
        """Generate path using A* algorithm"""
        # Define start and goal
        start_pos = drone.position
        goal_pos = Position(
            x=destination_dock.position.x,
            y=destination_dock.position.y,
            z=0.0  # Landing height
        )
        
        # Choose appropriate transit layer based on current traffic
        transit_layer = Layer.TRANSIT_1
        if self.layer_density[Layer.TRANSIT_1] > self.layer_density[Layer.TRANSIT_2]:
            transit_layer = Layer.TRANSIT_2
        
        # Define waypoints for the path
        waypoints = []
        
        # Starting point
        waypoints.append(Waypoint(
            position=start_pos,
            speed=0.0 if drone.status == Status.DOCKED else drone.max_speed * 0.5,
            hold_time=0.0
        ))
        
        # If taking off, add transition waypoint
        if drone.status == Status.DOCKED:
            # Takeoff waypoint (directly above dock)
            takeoff_pos = Position(x=start_pos.x, y=start_pos.y, z=3.0)
            waypoints.append(Waypoint(
                position=takeoff_pos,
                speed=drone.max_speed * 0.3,
                hold_time=0.5  # Brief pause in transition layer
            ))
        
        # Transit altitude for the selected layer
        transit_z = 5.0 if transit_layer == Layer.TRANSIT_1 else 7.0
        
        # Define grid points for A* pathfinding
        grid_size = 0.5  # 0.5m grid
        open_set = []
        closed_set = set()
        came_from = {}
        
        # Cost from start to current position
        g_score = {(start_pos.x, start_pos.y): 0}
        
        # Estimated total cost from start to goal through current position
        f_score = {(start_pos.x, start_pos.y): start_pos.horizontal_distance_to(goal_pos)}
        
        # Priority queue with f_score as priority
        heapq.heappush(open_set, (f_score[(start_pos.x, start_pos.y)], (start_pos.x, start_pos.y)))
        
        # A* search
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Goal reached
            if abs(current[0] - goal_pos.x) < grid_size and abs(current[1] - goal_pos.y) < grid_size:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert A* path to waypoints at transit altitude
                for i, (x, y) in enumerate(path):
                    # Skip start position, already added
                    if i == 0 and x == start_pos.x and y == start_pos.y:
                        continue
                    
                    # Skip points that are too close to each other
                    if i > 0 and i < len(path) - 1:
                        prev_x, prev_y = path[i-1]
                        if math.sqrt((x - prev_x)**2 + (y - prev_y)**2) < grid_size:
                            continue
                    
                    # For normal points, use transit altitude
                    waypoints.append(Waypoint(
                        position=Position(x=x, y=y, z=transit_z),
                        speed=drone.max_speed,
                        hold_time=0.0
                    ))
                
                break
            
            closed_set.add(current)
            
            # Check neighbors
            for dx, dy in [(0, grid_size), (grid_size, 0), (0, -grid_size), (-grid_size, 0),
                         (grid_size, grid_size), (grid_size, -grid_size), 
                         (-grid_size, grid_size), (-grid_size, -grid_size)]:
                neighbor = (round(current[0] + dx, 1), round(current[1] + dy, 1))
                
                # Skip if outside grid bounds
                if not (0 <= neighbor[0] <= 4 and 0 <= neighbor[1] <= 4):
                    continue
                
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Skip if in avoid zones (unless emergency)
                if neighbor in avoid_zones and not is_emergency:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + math.sqrt(dx**2 + dy**2)
                
                # Add new node or update if better path found
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + math.sqrt(
                        (neighbor[0] - goal_pos.x)**2 + (neighbor[1] - goal_pos.y)**2
                    )
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If landing, add approach waypoints
        if is_landing or is_emergency:
            # Approach point (above landing spot)
            approach_pos = Position(x=goal_pos.x, y=goal_pos.y, z=3.0)
            waypoints.append(Waypoint(
                position=approach_pos,
                speed=drone.max_speed * 0.3,
                hold_time=0.5  # Brief pause in transition layer
            ))
        
        # Final landing point
        waypoints.append(Waypoint(
            position=goal_pos,
            speed=0.0,
            hold_time=0.0
        ))
        
        # Create path segments
        segments = []
        total_distance = 0.0
        total_time = 0.0
        start_time = time.time()
        
        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            end_wp = waypoints[i+1]
            
            # Calculate distance
            distance = start_wp.position.distance_to(end_wp.position)
            total_distance += distance
            
            # Calculate speed and time
            avg_speed = (start_wp.speed + end_wp.speed) / 2 if avg_speed > 0 else drone.max_speed * 0.5
            segment_time = distance / avg_speed if avg_speed > 0 else 0
            
            # Determine layer
            layer = Layer.DOCK
            if start_wp.position.z < 2.0 and end_wp.position.z < 2.0:
                layer = Layer.DOCK
            elif start_wp.position.z < 4.0 or end_wp.position.z < 4.0:
                layer = Layer.TRANSITION
            elif start_wp.position.z < 6.0:
                layer = Layer.TRANSIT_1
            elif start_wp.position.z < 8.0:
                layer = Layer.TRANSIT_2
            else:
                layer = Layer.BEYOND
            
            # Create segment
            segment = PathSegment(
                start=start_wp,
                end=end_wp,
                start_time=start_time + total_time,
                end_time=start_time + total_time + segment_time,
                layer=layer
            )
            segments.append(segment)
            
            # Update total time
            total_time += segment_time + start_wp.hold_time
        
        # Create flight path
        flight_path = FlightPath(
            waypoints=waypoints,
            segments=segments,
            total_distance=total_distance,
            estimated_duration=total_time,
            start_time=start_time,
            end_time=start_time + total_time
        )
        
        return flight_path
    
    def _find_nearest_available_dock(self, position: Position) -> Optional[Dock]:
        """Find the nearest available dock"""
        nearest_dock = None
        min_distance = float('inf')
        
        for dock_id, dock in self.docks.items():
            if dock.status == "available":
                distance = position.horizontal_distance_to(dock.position)
                if distance < min_distance:
                    min_distance = distance
                    nearest_dock = dock
        
        return nearest_dock
    
    def _update_spatial_index(self, drone_id: str, old_pos: Position, new_pos: Position) -> None:
        """Update spatial index for collision detection"""
        # Remove from old position
        old_cell = (int(old_pos.x / self.cell_size), 
                   int(old_pos.y / self.cell_size),
                   int(old_pos.z / self.cell_size))
        if drone_id in self.spatial_index[old_cell]:
            self.spatial_index[old_cell].remove(drone_id)
        
        # Add to new position
        new_cell = (int(new_pos.x / self.cell_size), 
                   int(new_pos.y / self.cell_size),
                   int(new_pos.z / self.cell_size))
        self.spatial_index[new_cell].add(drone_id)
    
    def _update_density_metrics(self) -> None:
        """Background thread to update airspace density metrics"""
        while self.is_running:
            try:
                # Reset counters
                for layer in Layer:
                    self.layer_density[layer] = 0
                for quadrant in Quadrant:
                    self.quadrant_density[quadrant] = 0
                
                # Count drones in each layer and quadrant
                for drone_id, drone in self.drones.items():
                    if drone.status != Status.DOCKED:
                        layer = drone.get_layer()
                        quadrant = drone.get_quadrant()
                        self.layer_density[layer] += 1
                        self.quadrant_density[quadrant] += 1
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in density metrics update: {str(e)}")
                time.sleep(1)
    
    def _conflict_detection_loop(self) -> None:
        """Background thread for real-time conflict detection"""
        while self.is_running:
            try:
                # Check for potential conflicts
                self._check_potential_conflicts()
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error in conflict detection: {str(e)}")
                time.sleep(1)
    
    def _check_potential_conflicts(self) -> None:
        """Check for potential conflicts between drones"""
        # Get all airborne drones
        airborne_drones = {drone_id: drone for drone_id, drone in self.drones.items() 
                          if drone.status != Status.DOCKED}
        
        if len(airborne_drones) < 2:
            return  # No conflicts possible with fewer than 2 airborne drones
        
        # Check each pair of drones
        checked_pairs = set()
        for drone_id1, drone1 in airborne_drones.items():
            for drone_id2, drone2 in airborne_drones.items():
                if drone_id1 == drone_id2:
                    continue
                
                # Avoid checking the same pair twice
                pair_key = tuple(sorted([drone_id1, drone_id2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                # Check spatial separation
                horizontal_dist = drone1.position.horizontal_distance_to(drone2.position)
                vertical_dist = abs(drone1.position.z - drone2.position.z)
                
                # Check for conflict
                if (horizontal_dist < self.min_horizontal_separation and 
                    vertical_dist < self.min_vertical_separation):
                    self.stats["conflicts_detected"] += 1
                    
                    # Calculate collision time if drones maintain current velocities
                    collision_time = self._calculate_collision_time(drone1, drone2)
                    
                    # Immediate conflict resolution if time is short
                    if collision_time < 5.0:  # Less than 5 seconds
                        self._resolve_immediate_conflict(drone1, drone2)
                        self.stats["conflicts_resolved"] += 1
                    
                    logger.warning(f"Conflict detected between {drone_id1} and {drone_id2}")
    
    def _calculate_collision_time(self, drone1: Drone, drone2: Drone) -> float:
        """Calculate time to potential collision based on current trajectories"""
        # Get velocity vectors
        v1 = np.array(drone1.velocity)
        v2 = np.array(drone2.velocity)
        
        # Get position vectors
        p1 = np.array([drone1.position.x, drone1.position.y, drone1.position.z])
        p2 = np.array([drone2.position.x, drone2.position.y, drone2.position.z])
        
        # Calculate relative velocity and position
        rel_v = v1 - v2
        rel_p = p1 - p2
        
        # Check if relative velocity is zero
        if np.linalg.norm(rel_v) < 0.001:
            return float('inf')  # No collision if not moving relative to each other
        
        # Calculate time of closest approach
        t = -np.dot(rel_p, rel_v) / np.dot(rel_v, rel_v)
        
        # If time is negative, drones are moving away from each other
        if t < 0:
            return float('inf')
        
        # Calculate minimum distance at closest approach
        closest_p = rel_p + t * rel_v
        min_distance = np.linalg.norm(closest_p)
        
        # Check if minimum distance indicates a
        drone.status = Status.TAKEOFF
        drone.current_reservation_id = reservation_id
        drone.current_waypoint_index = 0
        
        # Move reservation to active
        self.active_reservations[reservation_id] = reservation
        
        logger.info(f"Takeoff initiated for drone {drone_id} with reservation {reservation_id}")
        return True
    
    def request_landing(self, drone_id: str, dock_id: str) -> Tuple[bool, Optional[str]]:
        """Request landing at a specific dock"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False, "Drone not found"
        
        if dock_id not in self.docks:
            logger.warning(f"Dock {dock_id} not found")
            return False, "Dock not found"
        
        drone = self.drones[drone_id]
        dock = self.docks[dock_id]
        
        # Check if drone is in flight
        if drone.status == Status.DOCKED:
            logger.warning(f"Drone {drone_id} is already docked")
            return False, "Drone is already docked"
        
        # Check if destination dock is available
        if dock.status != "available" and dock.current_drone_id != drone_id:
            logger.warning(f"Dock {dock_id} is not available")
            return False, "Dock is not available"
        
        # Generate landing path
        path = self._generate_landing_path(drone, dock)
        if not path:
            logger.warning(f"Could not generate valid landing path for drone {drone_id}")
            return False, "Could not generate valid landing path"
        
        # Create reservation with higher priority for landing
        landing_priority = Priority.HIGH if drone.priority.value < Priority.HIGH.value else drone.priority
        reservation_id = str(uuid.uuid4())
        reservation = PathReservation(
            id=reservation_id,
            drone_id=drone_id,
            path=path,
            priority=landing_priority,  # Landings get higher priority
            status=ReservationStatus.PENDING,
            request_time=time.time(),
            expiry_time=time.time() + 300  # 5 minute expiry
        )
        
        # Add to reservation system
        self.reservations[reservation_id] = reservation
        heapq.heappush(
            self.reservation_queue, 
            (-landing_priority.value, reservation.request_time, reservation_id)
        )
        
        logger.info(f"Landing request for drone {drone_id} at dock {dock_id} submitted with reservation {reservation_id}")
        return True, reservation_id
    
    def initiate_landing(self, drone_id: str, reservation_id: str) -> bool:
        """Initiate landing sequence with approved reservation"""
        if drone_id not in self.drones:
            logger.warning(f"Drone {drone_id} not found")
            return False
        
        if reservation_id not in self.reservations:
            logger.warning(f"Reservation {reservation_id} not found")
            return False
        
        reservation = self.reservations[reservation_id]
        if reservation.status != ReservationStatus.APPROVED:
            logger.warning(f"Reservation {reservation_id} is not approved")
            return False
        
        drone = self.drones[drone_id]
        if drone.status == Status.DOCKED:
            logger.warning(f"Drone {drone_id} is already docked")
            return False
        
        # Clear any current reservation
        if drone.current_reservation_id and drone.current_reservation_id in self.active_reservations:
            old_reservation_id = drone.current_reservation_id
            if old_reservation_id in self.active_reservations:
                del self.active_reservations[old_reservation_id]
        
        # Update drone status
        return False