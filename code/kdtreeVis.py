import numpy as np
from collections import defaultdict
from typing import Optional, List, Tuple, Generator
from visualizer.main import Visualizer
from rectangle import Rectangle
from operator import itemgetter

class Node:
    __slots__ = ["point", "left", "right", "count"]

    def __init__(
        self,
        point: Tuple[float, float],
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        count: int = 1,
    ):
        self.point = point
        self.left = left
        self.right = right
        self.count = count

    def __repr__(self):
        if self.point is None:
            return "None"
        return f"Node(point={self.point}, count={self.count})"
    
class KdTreeVis:
    def __init__(self, points: np.ndarray):
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a NumPy ndarray.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (n_points, 2).")

        self.vis = Visualizer()
        self.points = points
        self.k = 2
        self.counts = self._count_duplicates()
        unique_points = np.array(list(self.counts.keys()))
        self.boundary = self._calculate_boundary(points)
        
        # Dodanie wszystkich punktów na wizualizacji przed rozpoczęciem budowy drzewa
        self.vis.add_point([tuple(p) for p in self.points], color='black')
        self.vis.add_polygon(self._rectangle_as_polygon(self.boundary), color='#2a2a2a', fill=False)
        
        self.root = self.build(unique_points, self.boundary)

    def _count_duplicates(self) -> defaultdict:
        counts = defaultdict(int)
        for point in self.points:
            key = tuple(point)
            counts[key] += 1
        return counts

    def _calculate_boundary(self, points: np.ndarray) -> Rectangle:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        return Rectangle(xmin, ymin, xmax, ymax)
    
    def build(self, points: np.ndarray, boundary: Rectangle, depth: int = 0) -> Optional[Node]:
        if len(points) == 0:
            return None

        axis = depth % self.k
        idxs = points[:, axis].argsort()
        points = points[idxs]

        median_idx = len(points) // 2
        median_point = tuple(points[median_idx])
        median_coord = points[median_idx, axis]

        self.vis.add_point([median_point], color='red')

        if axis == 0:
            self.vis.add_line_segment(
                [(median_coord, boundary.extreme[1][0]), (median_coord, boundary.extreme[1][1])],
                color='red'
            )
            self.vis.add_line_segment(
                [(median_coord, boundary.extreme[1][0]), (median_coord, boundary.extreme[1][1])],
                color='#d3d3d3'
            )
            left_boundary = Rectangle(boundary.extreme[0][0], boundary.extreme[1][0], median_coord, boundary.extreme[1][1])
            right_boundary = Rectangle(median_coord, boundary.extreme[1][0], boundary.extreme[0][1], boundary.extreme[1][1])
        else:
            self.vis.add_line_segment(
                [(boundary.extreme[0][0], median_coord), (boundary.extreme[0][1], median_coord)],
                color='red'
            )
            self.vis.add_line_segment(
                [(boundary.extreme[0][0], median_coord), (boundary.extreme[0][1], median_coord)],
                color='#d3d3d3'
            )
            left_boundary = Rectangle(boundary.extreme[0][0], boundary.extreme[1][0], boundary.extreme[0][1], median_coord)
            right_boundary = Rectangle(boundary.extreme[0][0], median_coord, boundary.extreme[0][1], boundary.extreme[1][1])

        indices = np.arange(len(points))
        mask_left = (points[:, axis] < median_coord) | (
            (points[:, axis] == median_coord) & (indices != median_idx)
        )
        left_points = points[mask_left]
        self.vis.add_point([tuple(p) for p in left_points], color='pink')
        self.vis.add_point([tuple(p) for p in left_points], color='black')

        left_child = self.build(left_points, left_boundary, depth + 1)

        self.vis.add_point([tuple(p) for p in left_points], color='black')

        right_points = points[points[:, axis] > median_coord]
        self.vis.add_point([tuple(p) for p in right_points], color='pink')
        self.vis.add_point([tuple(p) for p in right_points], color='black')

        right_child = self.build(right_points, right_boundary, depth + 1)

        self.vis.add_point([tuple(p) for p in right_points], color='black')

        self.vis.add_point([median_point], color='black')

        
        return Node(
            point=median_point,
            count=self.counts[median_point],
            left=left_child,
            right=right_child,
        )


    def search_rectangle(self, rectangle: Rectangle) -> Generator[Node, None, None]:
        yield from self._search_rectangle(rectangle, self.root, 0)

    def _search_rectangle(
        self, rectangle: Rectangle, node: Optional[Node], depth: int
    ) -> Generator[Node, None, None]:
        if node is None:
            return

        if rectangle.contains(node.point):
            yield node

        axis = depth % self.k
        coord = node.point[axis]

        if rectangle.extreme[axis][0] <= coord:
            yield from self._search_rectangle(rectangle, node.left, depth + 1)
        if rectangle.extreme[axis][1] > coord:
            yield from self._search_rectangle(rectangle, node.right, depth + 1)

    def _rectangle_as_polygon(self, rectangle: Rectangle) -> List[List[float]]:
        corners = [
            (rectangle.extreme[0][0], rectangle.extreme[1][0]),
            (rectangle.extreme[0][0], rectangle.extreme[1][1]),
            (rectangle.extreme[0][1], rectangle.extreme[1][1]),
            (rectangle.extreme[0][1], rectangle.extreme[1][0]),
        ]
        return corners


# def main():
#     points = np.array([[2, 3], [5, 7], [9, 6], [4, 7], [5, 7], [7, 2], [6, 6], [15, 15], [5, 15], [16, 15], [5, 5]])
#     quadtree = KdTreeVis(points)
#     rectangle = Rectangle(5, 5, 15, 15)
#     quadtree.vis.save_gif("kdtree_build", interval=250)
#     quadtree.search_rectangle(rectangle)


# main()
