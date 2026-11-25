from collections.abc import Mapping, Sequence

from controller import TILE, wheel_speeds
from plan_a_star import AStarPlanner, Tile
from mujoco import MjData, MjModel

WorldPoint = tuple[float, float]


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


def world_to_tile(position: WorldPoint) -> Tile:
    x, y = position
    return round(x / TILE), round(y / TILE)
