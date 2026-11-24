"""A* planning on the dungeon's tile map.

Adapted directly from PythonRobotics/PathPlanning/AStar/a_star.py.
https://github.com/AtsushiSakai/PythonRobotics/blob/ef64608ceb0012f2a0b3d532fddc80243e807cfa/PathPlanning/AStar/a_star.py

"""


# This adaptation omits PythonRobotics' plotting and standalone demo.
from collections.abc import Mapping

Tile = tuple[int, int]


class AStarPlanner:
    """Find shortest paths through floor tiles using four-connected A*."""

    # Use the supplied dungeon tile map in place of a generated obstacle grid.
    def __init__(self, tiles: Mapping[Tile, str]):
        self.tiles = tiles
        # Four unit-cost cardinal moves prevent diagonal corner cutting.
        self.motion = ((1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1))

    # Parent indices are tile coordinates. None marks the start node.
    class Node:
        def __init__(self, x: int, y: int, cost: int, parent_index: Tile | None):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, start: Tile, goal: Tile) -> list[Tile]:
        """Return tiles from start to goal, or an empty list if no route exists."""
        # Endpoints are tile coordinates and must both be floor.
        if self.tiles.get(start) != "." or self.tiles.get(goal) != ".":
            return []

        start_node = self.Node(*start, 0, None)
        goal_node = self.Node(*goal, 0, None)
        # Use (x, y) directly as the key in both search sets.
        open_set = {start: start_node}
        closed_set = {}

        while open_set:
            current_id = min(
                open_set,
                key=lambda tile: open_set[tile].cost
                + self.calc_heuristic(goal_node, open_set[tile]),
            )
            # Remove the selected node and reconstruct from it on arrival.
            current = open_set.pop(current_id)

            if current_id == goal:
                return self.calc_final_path(current, closed_set)

            closed_set[current_id] = current

            # Only floor tiles are traversable, so missing tiles are blocked too.
            for dx, dy, cost in self.motion:
                neighbor = (current.x + dx, current.y + dy)
                if self.tiles.get(neighbor) != "." or neighbor in closed_set:
                    continue

                node = self.Node(*neighbor, current.cost + cost, current_id)
                if neighbor not in open_set or open_set[neighbor].cost > node.cost:
                    open_set[neighbor] = node

        # An empty frontier means there is no route to reconstruct.
        return []

    @staticmethod
    def calc_final_path(goal_node, closed_set) -> list[Tile]:
        path = [(goal_node.x, goal_node.y)]
        parent_index = goal_node.parent_index
        while parent_index is not None:
            node = closed_set[parent_index]
            path.append((node.x, node.y))
            parent_index = node.parent_index

        # The parent chain runs backward. Return the route in driving order.
        path.reverse()
        return path

    # Manhattan distance estimates the remaining number of cardinal steps.
    @staticmethod
    def calc_heuristic(first, second) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)
