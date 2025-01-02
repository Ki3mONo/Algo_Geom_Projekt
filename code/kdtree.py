import numpy as np
from collections import defaultdict
from typing import Optional, Generator, Tuple
from operator import itemgetter
from .rectangle import Rectangle
import sys

sys.setrecursionlimit(10**6)
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


class KdTree:
    def __init__(self, points: np.ndarray):
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a NumPy ndarray.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (n_points, 2).")
        self.k = 2
        self.points = points
        self.counts = self._count_duplicates()
        unique_points = np.array(list(self.counts.keys()))
        self.root = self.build(unique_points)

    def _count_duplicates(self) -> defaultdict:
        counts = defaultdict(int)
        for point in self.points:
            key = tuple(point)
            counts[key] += 1
        return counts

    def build(self, points: np.ndarray, depth: int = 0) -> Optional[Node]:
        if len(points) == 0:
            return None

        axis = depth % self.k
        idxs = points[:, axis].argsort()
        points = points[idxs]

        median_idx = len(points) // 2
        median_point = tuple(points[median_idx])
        median_coord = points[median_idx, axis]

        indices = np.arange(len(points))
        mask_left = (points[:, axis] < median_coord) | (
            (points[:, axis] == median_coord) & (indices != median_idx)
        )
        left_points = points[mask_left]

        right_points = points[points[:, axis] > median_coord]

        left_child = self.build(left_points, depth + 1)
        right_child = self.build(right_points, depth + 1)

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


# def main():
#     points = np.array([[2, 3], [5, 7], [9, 6], [4, 7], [5, 7], [7, 2], [6, 6], [15, 15], [5, 15], [16, 15], [5, 5]])
#     kdtree = KdTree(points)
#     rectangle = Rectangle(5, 5, 15, 15)
#     result = kdtree.search_rectangle(rectangle)
#     for node in result:
#         print(node)


# main()
