import numpy as np
from code.rectangle import Rectangle
import matplotlib.pyplot as plt

def random_uniform_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = np.random.uniform(minval, maxval, (count, dimension))
    rect = Rectangle(0.3*maxval, 0.3*maxval, minval + (maxval - minval) * 0.7, minval + (maxval - minval) * 0.7)
    return points, rect

def random_normal_test(count: int, mean: float = 0, std: float = 1, dimension: int = 2):
    points = np.random.normal(mean, std, (count, dimension))
    minval = mean - std
    maxval = mean + std
    rect = Rectangle(minval, minval, 0.8*minval + (maxval - minval), 0.8*minval + (maxval - minval))
    return points, rect

def random_integer_test(count: int, low: int = 0, high: int = 100, dimension: int = 2):
    points = np.random.randint(low, high, (count, dimension))
    rect = Rectangle(0.2*high, low, 0.9 * high, 0.9 * high)
    return points, rect

def grid_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = np.array(
        [
            [x, y]
            for x in np.linspace(minval, maxval, int(np.sqrt(count)))
            for y in np.linspace(minval, maxval, int(np.sqrt(count)))
        ]
    )
    rect = Rectangle(minval, minval, minval + (maxval - minval) * 0.5, minval + (maxval - minval) * 0.5)
    return points, rect

def circle_test(count: int, radius: float = 1, dimension: int = 2):
    points = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    points = np.array(points)
    rect = Rectangle(-radius * 0.5, -radius, radius * 0.5, radius)
    return points, rect

def line_test(count: int, x1: float = -50, y1: float = -50, x2: float = 50, y2: float = 50):
    points = np.linspace([x1, y1], [x2, y2], count)
    rect = Rectangle(
        min(x1, x2), min(y1, y2), 
        min(x1, x2) + abs(x2 - x1) * 0.7, 
        min(y1, y2) + abs(y2 - y1) * 0.7
    )
    return points, rect

def cross_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = []
    for i in range(count // 2):
        x = np.random.uniform(minval, maxval)
        y = 0
        points.append([x, y])
        x = 0
        y = np.random.uniform(minval, maxval)
        points.append([x, y])
    points = np.array(points)
    rect = Rectangle(-maxval * 0.5, -maxval * 0.5, maxval * 0.5, maxval * 0.5)
    return points, rect

def rectangle_sides_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = []
    for i in range(count // 4):
        x = np.random.uniform(minval, maxval)
        y = minval
        points.append([x, y])
        y = maxval
        points.append([x, y])
        y = np.random.uniform(minval, maxval)
        x = minval
        points.append([x, y])
        x = maxval
        points.append([x, y])
    points = np.array(points)
    rect = Rectangle(minval, minval, maxval * 0.5, maxval * 0.5)
    return points, rect

def rectangle_with_diagonals_test(count: int, a: float = 50, b: float = 50, dimension: int = 2, on_diagonal_ratio: float = 0.5):
    diag_count = int(count * on_diagonal_ratio)
    side_count = count - diag_count

    diag_points = []
    for i in range(diag_count):
        if i % 2 == 0:
            x = np.random.uniform(0, a)
            y = (b / a) * x
        else:
            x = np.random.uniform(0, a)
            y = b - (b / a) * x
        diag_points.append([x, y])

    side_points = []
    for i in range(side_count):
        side = i % 4
        if side == 0:
            x = np.random.uniform(0, a)
            y = 0
        elif side == 1:
            x = np.random.uniform(0, a)
            y = b
        elif side == 2:
            x = 0
            y = np.random.uniform(0, b)
        elif side == 3:
            x = a
            y = np.random.uniform(0, b)
        side_points.append([x, y])

    points = np.array(diag_points + side_points)
    rect = Rectangle(0, 0.1*b, 0.7*a, 0.7*b)
    return points, rect

def test_two_clusters(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = []
    for i in range(count // 2):
        x = np.random.uniform(minval, minval * 0.5)
        y = np.random.uniform(minval, minval * 0.5)
        points.append([x, y])
        x = np.random.uniform(maxval * 0.5, maxval)
        y = np.random.uniform(maxval * 0.5, maxval)
        points.append([x, y])
    points = np.array(points)
    rect = Rectangle(minval * 0.7, minval * 0.7, maxval * 0.7, maxval * 0.7)
    return points, rect

def test_with_outliers(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2):
    points = np.random.uniform(minval, maxval, (count, dimension))
    for i in range(count // 10):
        x = np.random.uniform(minval * 2, maxval * 2)
        y = np.random.uniform(minval * 2, maxval * 2)
        points = np.append(points, [[x, y]], axis=0)
    rect = Rectangle(minval, minval, maxval * 0.5, maxval * 0.5)
    return points, rect


# def main():
#     points, rect = rectangle_with_diagonals_test(1000)
#     plt.scatter(points[:, 0], points[:, 1])
#     plt.gca().add_patch(plt.Rectangle((rect.extreme[0][0], rect.extreme[1][0]), rect.extreme[0][1] - rect.extreme[0][0], rect.extreme[1][1] - rect.extreme[1][0], fill=False))
#     plt.show()
# main()