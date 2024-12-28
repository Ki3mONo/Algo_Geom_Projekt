import numpy as np

def random_uniform_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    return np.random.uniform(minval, maxval, (count, dimension))

def random_normal_test(count: int, mean: float = 0, std: float = 1, dimension: int = 2) -> np.ndarray:
    return np.random.normal(mean, std, (count, dimension))

def random_integer_test(count: int, low: int = 0, high: int = 100, dimension: int = 2) -> np.ndarray:
    return np.random.randint(low, high, (count, dimension))

def grid_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    return np.array(
        [
            [x, y]
            for x in np.linspace(minval, maxval, int(np.sqrt(count)))
            for y in np.linspace(minval, maxval, int(np.sqrt(count)))
        ]
    )

def circle_test(count: int, radius: float = 1, dimension: int = 2) -> np.ndarray:
    points = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def cross_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    points = []
    for i in range(count // 2):
        x = np.random.uniform(minval, maxval)
        y = 0
        points.append([x, y])
        x = 0
        y = np.random.uniform(minval, maxval)
        points.append([x, y])
    return np.array(points)

def line_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    x = np.random.uniform(minval, maxval)
    points = [[x, y] for y in np.linspace(minval, maxval, count)]
    return np.array(points)

def rectangle_sides_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
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
    return np.array(points)

def rectangle_with_diagonals_test(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2, on_diagonal_ratio: float = 0.5) -> np.ndarray:
    points = []
    for i in range((1-count) * on_diagonal_ratio):
        x = np.random.uniform(minval, maxval)
        y = x
        points.append([x, y])
    return np.array(points)

def test_two_clusters(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    points = []
    for i in range(count // 2):
        x = np.random.uniform(minval, 0.2 * minval)
        y = np.random.uniform(minval, 0.2 * minval)
        points.append([x, y])
        x = np.random.uniform(0.2 * maxval, maxval)
        y = np.random.uniform(0.2 * maxval, maxval)
        points.append([x, y])
    return np.array(points)

def test_with_outliers(count: int, minval: float = -100, maxval: float = 100, dimension: int = 2) -> np.ndarray:
    points = np.random.uniform(minval, maxval, (count, dimension))
    for i in range(count // 10):
        x = np.random.uniform(2 * minval, 2 * maxval)
        y = np.random.uniform(2* minval, 2 * maxval)
        points = np.append(points, [[x, y]], axis=0)
    return points

def visualize_test(points: np.ndarray, title: str = "Test", labels: np.ndarray = None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=50)
    plt.title(title)
    plt.show()

def main():
    points = random_uniform_test(100)
    points = rectangle_with_diagonal_test(100)
    visualize_test(points, "Test")

main()