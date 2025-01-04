import numpy as np
from .visualizer.main import Visualizer
from typing import Optional, Generator, Tuple, List, Dict
from .rectangle import Rectangle

class QuadTreeNode:
    def __init__(
        self,
        boundary: Rectangle,
        points: Optional[List[Tuple[float, float]]] = None,
        children: Optional[List["QuadTreeNode"]] = None
    ):
        self.boundary = boundary
        self.points = points if points is not None else []
        self.children = children if children is not None else []

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class QuadTreeVis:
    def __init__(self, points: np.ndarray, max_capacity: int = 4):
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a NumPy ndarray.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must be of shape (n_points, 2).")

        self.vis = Visualizer()
        self.points = points
        self.max_capacity = max_capacity

        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        self.boundary = Rectangle(xmin, ymin, xmax, ymax)
        
        self.vis.add_polygon(self._rectangle_as_polygon(self.boundary), color='#2a2a2a',fill=False)
        self.vis.add_point(self.points, color='black')
        all_indices = np.arange(points.shape[0])
        self.root = self._build(all_indices, self.boundary)

    def _build(self, indices: np.ndarray, boundary: Rectangle) -> Optional[QuadTreeNode]:
        if len(indices) == 0:
            return None

        sub_points = self.points[indices]
        unique_points = np.unique(sub_points, axis=0)

        xmin, ymin = boundary.extreme[0][0], boundary.extreme[1][0]
        xmax, ymax = boundary.extreme[0][1], boundary.extreme[1][1]
        if xmin == xmax or ymin == ymax:
            return None

        self.vis.add_polygon(self._rectangle_as_polygon(boundary), color='orange', fill=False)

        if len(unique_points) <= self.max_capacity:
            # Węzeł liścia przechowuje wszystkie punkty
            return QuadTreeNode(boundary=boundary, points=[tuple(p) for p in unique_points])

        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        rect_bl = Rectangle(xmin, ymin, xmid, ymid)
        rect_br = Rectangle(xmid, ymin, xmax, ymid)
        rect_tl = Rectangle(xmin, ymid, xmid, ymax)
        rect_tr = Rectangle(xmid, ymid, xmax, ymax)

        bl_mask = (sub_points[:, 0] < xmid) & (sub_points[:, 1] < ymid)
        br_mask = (sub_points[:, 0] >= xmid) & (sub_points[:, 1] < ymid)
        tl_mask = (sub_points[:, 0] < xmid) & (sub_points[:, 1] >= ymid)
        tr_mask = (sub_points[:, 0] >= xmid) & (sub_points[:, 1] >= ymid)

        bl_indices = indices[bl_mask]
        br_indices = indices[br_mask]
        tl_indices = indices[tl_mask]
        tr_indices = indices[tr_mask]

        children = []
        for rect, child_indices in zip([rect_bl, rect_br, rect_tl, rect_tr], [bl_indices, br_indices, tl_indices, tr_indices]):
            child_node = self._build(child_indices, rect)
            if child_node is not None:
                children.append(child_node)

        return QuadTreeNode(boundary=boundary, children=children)


    def search_rectangle(self, rectangle: Rectangle) -> List[Tuple[float, float]]:
        self.vis.clear()
        self.vis.add_point(self.points, color='black')
        self.vis.add_polygon(self._rectangle_as_polygon(self.boundary), color='#2a2a2a', fill=False)
        self.vis.add_polygon(self._rectangle_as_polygon(rectangle), color='blue', fill=False)
        if self.root is None:
            return []

        return self._search_rectangle(self.root, rectangle, seen=set())


    def _search_rectangle(
        self,
        node: QuadTreeNode,
        rectangle: Rectangle,
        seen: set
    ) -> List[Tuple[float, float]]:
        if node is None:
            return []

        results = []
        self.vis.add_polygon(self._rectangle_as_polygon(node.boundary), color='pink', fill=False)

        if not rectangle.intersects(node.boundary):
            return results

        if node.is_leaf():
            for px, py in node.points:
                color = 'green' if rectangle.contains((px, py)) else 'black'
                self.vis.add_point([(px, py)], color=color)

                if rectangle.contains((px, py)) and (px, py) not in seen:
                    seen.add((px, py))
                    results.append((px, py))
        else:
            for child in node.children:
                results.extend(self._search_rectangle(child, rectangle, seen))

        return results



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
#     quadtree = QuadTreeVis(points, 1)
#     rectangle = Rectangle(5, 5, 15, 15)
#     quadtree.vis.save_gif("quadtree_build", interval=5)
#     quadtree.vis.clear()
#     quadtree.search_rectangle(rectangle)
#     quadtree.vis.save_gif("quadtree_search", interval=5)
    

# main()
