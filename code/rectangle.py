from typing import Tuple

class Rectangle:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.extreme = [[xmin, xmax], [ymin, ymax]]
        for p1, p2 in self.extreme:
            if p1 > p2:
                raise ValueError("Invalid rectangle definition")

    def contains(self, point: Tuple[float, float]) -> bool:
        return (
            self.extreme[0][0] <= point[0] <= self.extreme[0][1]
            and self.extreme[1][0] <= point[1] <= self.extreme[1][1]
        )

    def intersects(self, other: "Rectangle") -> bool:
        return not (
            self.extreme[0][1] < other.extreme[0][0]
            or self.extreme[0][0] > other.extreme[0][1]
            or self.extreme[1][1] < other.extreme[1][0]
            or self.extreme[1][0] > other.extreme[1][1]
        )
