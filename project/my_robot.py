from collections.abc import Mapping, Sequence
from math import atan2, cos, hypot, sin
from typing import NamedTuple

from mujoco import MjData, MjModel

from controller import TILE, wheel_speeds
from plan_a_star import AStarPlanner, Tile

WorldPoint = tuple[float, float]
WheelSpeeds = tuple[float, float]


class Quaternion(NamedTuple):
    """Orientation in MuJoCo's (w, x, y, z) order."""

    w: float
    x: float
    y: float
    z: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "Quaternion":
        """Convert MuJoCo's (w, x, y, z) orientation to a quaternion."""
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    def yaw(self) -> float:
        """Return the heading angle in radians."""
        return atan2(
            2.0 * (self.w * self.z + self.x * self.y),
            1.0 - 2.0 * (self.y * self.y + self.z * self.z),
        )

LOOKAHEAD: float = 0.5  # m
CORNER_TOLERANCE: float = 0.15  # m
FINISH_TOLERANCE: float = 0.35  # m, inside the simulator's finish radius
CRUISE_SPEED: float = 0.6  # m/s
MAX_TURN_RATE: float = 1.2  # rad/s
ALIGN_TOLERANCE: float = 0.12  # rad
REALIGN_THRESHOLD: float = 0.55  # rad


class Controller:
    def __init__(self, m: MjModel, d: MjData, tiles: Mapping[Tile, str]) -> None:
        start_position: WorldPoint = (
            float(d.site("start").xpos[0]),
            float(d.site("start").xpos[1]),
        )
        finish_position: WorldPoint = (
            float(d.site("finish").xpos[0]),
            float(d.site("finish").xpos[1]),
        )
        start: Tile = world_to_tile(start_position)

        finish: Tile = world_to_tile(finish_position)

        planner: AStarPlanner = AStarPlanner(tiles)

        route: list[Tile] = planner.planning(start, finish)

        self.waypoints: list[WorldPoint] = self._route_to_waypoints(route)

        self.target_index: int = 1
        self.aligning: bool = True

    def control(self, m: MjModel, d: MjData) -> WheelSpeeds:
        """Track the planned waypoints and return wheel speeds in rad/s.
        Uses pure pursuit. See: https://existentialrobotics.org/RobotProvingGrounds/algorithms/control/content/pure_pursuit

        Regarding the Algorithm steps in the website the relation is roughly:

        Website Step    | python function
        ---             | --- 
        Steps 1-2       | `pursuit_target`
        Step 3          | `lateral_offset`
        Step 4          | `pursuit_curvature`
        Steps 5-6       | `_pursuit_wheel_speeds` 

        """
        if self.target_index >= len(self.waypoints):
            return 0.0, 0.0

        base = d.body("robot-base_link")
        position: WorldPoint = (float(base.xpos[0]), float(base.xpos[1]))
        orientation: Quaternion = Quaternion.from_sequence(base.xquat)
        yaw: float = orientation.yaw()

        end: WorldPoint = self.waypoints[self.target_index]
        distance: float = hypot(end[0] - position[0], end[1] - position[1])
        at_finish: bool = self.target_index == len(self.waypoints) - 1
        tolerance: float = FINISH_TOLERANCE if at_finish else CORNER_TOLERANCE
        if distance <= tolerance:
            self.target_index += 1
            self.aligning = True
            return 0.0, 0.0

        start: WorldPoint = self.waypoints[self.target_index - 1]
        target: WorldPoint = pursuit_target(position, start, end, LOOKAHEAD)
        offset: WorldPoint = (target[0] - position[0], target[1] - position[1])
        lateral: float = lateral_offset(offset, yaw)
        curvature: float = pursuit_curvature(offset, lateral)
        return self._pursuit_wheel_speeds(offset, yaw, distance, curvature)

    def _pursuit_wheel_speeds(
        self, offset: WorldPoint, yaw: float, distance: float, curvature: float
    ) -> WheelSpeeds:
        """Steps 5–6. Return wheel commands for this simulation step."""
        angle: float = atan2(offset[1], offset[0]) - yaw
        heading_error: float = atan2(sin(angle), cos(angle))
        threshold: float = ALIGN_TOLERANCE if self.aligning else REALIGN_THRESHOLD
        self.aligning = abs(heading_error) > threshold
        if self.aligning:
            return wheel_speeds(0.0, self._bounded_turn_rate(2.0 * heading_error))

        speed: float = min(CRUISE_SPEED, 0.8 * distance)
        return wheel_speeds(speed, self._bounded_turn_rate(speed * curvature))

    @staticmethod
    def _bounded_turn_rate(rate: float) -> float:
        return max(-MAX_TURN_RATE, min(MAX_TURN_RATE, rate))

    @staticmethod
    def _route_to_waypoints(route: Sequence[Tile]) -> list[WorldPoint]:
        """Keep the route endpoints and turns, expressed in meters."""
        if not route:
            return []

        waypoint_tiles: list[Tile] = [route[0]]
        previous_direction: tuple[int, int] | None = None
        for start, end in zip(route, route[1:]):
            direction: tuple[int, int] = (end[0] - start[0], end[1] - start[1])
            if abs(direction[0]) + abs(direction[1]) != 1:
                raise ValueError("Route steps must join cardinally adjacent tiles.")
            if previous_direction is not None and direction != previous_direction:
                waypoint_tiles.append(start)
            previous_direction = direction

        if len(route) > 1:
            waypoint_tiles.append(route[-1])

        return [(i * TILE, j * TILE) for i, j in waypoint_tiles]


# Pure-pursuit geometry adapted from A4/python/pure_pursuit_driver.py.
def pursuit_target(
    position: WorldPoint, start: WorldPoint, end: WorldPoint, lookahead: float
) -> WorldPoint:
    """Steps 1–2. Project onto the segment and look ahead toward its endpoint."""
    dx: float = end[0] - start[0]
    dy: float = end[1] - start[1]
    length: float = hypot(dx, dy)
    if length < 1e-9:
        return end

    projection: float = (
        (position[0] - start[0]) * dx + (position[1] - start[1]) * dy
    ) / length
    progress: float = max(0.0, min(length, projection))
    # this implementation is kinda limited by the capping of the lookahead distance :(
    # A4 carries look-ahead across segments and stores monotonic progress.
    # Cap it at this endpoint and let the controller decide when to turn.
    fraction: float = min(length, progress + lookahead) / length
    return start[0] + fraction * dx, start[1] + fraction * dy


def lateral_offset(offset: WorldPoint, yaw: float) -> float:
    """Step 3. Measure the target's sideways offset, positive to the left."""
    dx, dy = offset
    return -sin(yaw) * dx + cos(yaw) * dy


def pursuit_curvature(offset: WorldPoint, lateral: float) -> float:
    """Step 4. Return signed curvature in inverse meters."""
    dx, dy = offset
    distance_squared: float = dx * dx + dy * dy
    return 2.0 * lateral / max(distance_squared, 1e-9)


def world_to_tile(position: WorldPoint) -> Tile:
    x, y = position
    return round(x / TILE), round(y / TILE)
