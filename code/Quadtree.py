import numpy as np
from typing import Optional, Generator, Tuple, List, Dict
from .rectangle import Rectangle


class QuadTreeNode:
    def __init__(
        self,
        boundary: Rectangle,
        point_counts: Optional[Dict[Tuple[float, float], int]] = None,
        children: Optional[List["QuadTreeNode"]] = None
    ):
        self.boundary = boundary
        self.point_counts = point_counts
        self.children = children if children is not None else []

        if self.point_counts is not None:
            self.count = sum(self.point_counts.values())
        else:
            self.count = sum(child.count for child in self.children)

    def is_leaf(self) -> bool:
        return self.point_counts is not None


class QuadTree:
    def __init__(self, points: np.ndarray, max_capacity: int = 4):
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a NumPy ndarray.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (n_points, 2).")

        self.points = points
        self.max_capacity = max_capacity

        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        self.boundary = Rectangle(xmin, ymin, xmax, ymax)

        all_indices = np.arange(points.shape[0])
        self.root = self._build(all_indices)

    def _build(self, indices: np.ndarray) -> Optional[QuadTreeNode]:
        if len(indices) == 0:
            return None

        sub_points = self.points[indices]
        xmin, ymin = sub_points.min(axis=0)
        xmax, ymax = sub_points.max(axis=0)
        boundary = Rectangle(xmin, ymin, xmax, ymax)

        unique_points, counts = np.unique(sub_points, axis=0, return_counts=True)

        if len(unique_points) <= self.max_capacity:
            point_counts = {
                (p[0], p[1]): c
                for p, c in zip(unique_points, counts)
            }
            return QuadTreeNode(boundary=boundary, point_counts=point_counts)

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        bl_mask = (sub_points[:, 0] <= xmid) & (sub_points[:, 1] <= ymid)
        br_mask = (sub_points[:, 0] >  xmid) & (sub_points[:, 1] <= ymid)
        tl_mask = (sub_points[:, 0] <= xmid) & (sub_points[:, 1] >  ymid)
        tr_mask = (sub_points[:, 0] >  xmid) & (sub_points[:, 1] >  ymid)

        bl_indices = indices[bl_mask]
        br_indices = indices[br_mask]
        tl_indices = indices[tl_mask]
        tr_indices = indices[tr_mask]

        children = []
        for child_indices in [bl_indices, br_indices, tl_indices, tr_indices]:
            child_node = self._build(child_indices)
            if child_node is not None:
                children.append(child_node)

        node = QuadTreeNode(boundary=boundary, children=children, point_counts=None)
        return node

    def search_rectangle(self, rectangle: Rectangle) -> Generator[Tuple[float, float], None, None]:
        if self.root is None:
            return
        yield from self._search_rectangle(self.root, rectangle, seen=set())

    def _search_rectangle(
        self,
        node: QuadTreeNode,
        rectangle: Rectangle,
        seen: set
    ) -> Generator[Tuple[float, float], None, None]:
        if node is None:
            return

        if not rectangle.intersects(node.boundary):
            return

        if node.is_leaf():
            for (px, py), count in (node.point_counts or {}).items():
                if (px, py) not in seen and rectangle.contains((px, py)):
                    seen.add((px, py))
                    yield (px, py)
        else:
            for child in node.children:
                yield from self._search_rectangle(child, rectangle, seen)


    def search_rectangle_with_count(self, rectangle: Rectangle) -> Generator[Tuple[Tuple[float, float], int], None, None]:
        if self.root is None:
            return  
        yield from self._search_rectangle_with_count(self.root, rectangle, seen=set())

    def _search_rectangle_with_count(
        self,
        node: QuadTreeNode,
        rectangle: Rectangle,
        seen: set
    ) -> Generator[Tuple[Tuple[float, float], int], None, None]:
        if node is None:
            return

        if not rectangle.intersects(node.boundary):
            return

        if node.is_leaf():
            for (px, py), count in node.point_counts.items():
                if (px, py) not in seen and rectangle.contains((px, py)):
                    seen.add((px, py))
                    yield ((px, py), count)
        else:
            for child in node.children:
                yield from self._search_rectangle_with_count(child, rectangle, seen)



# def main():
#     points = np.array([[2, 3], [5, 7], [9, 6], [4, 7], [5, 7], [7, 2], [6, 6], [15, 15], [5, 15], [16, 15], [5, 5]])
#     quadtree = QuadTree(points, 1)
#     rectangle = Rectangle(5, 5, 15, 15)
    
#     result_points = quadtree.search_rectangle(rectangle)
#     for (x, y) in result_points:
#         print(f"Point: ({x}, {y})")
        
#     result_all = quadtree.search_rectangle_with_count(rectangle)
#     for (x, y), count in result_all:
#         print(f"Point: ({x}, {y}), Count: {count}")
        
# main()