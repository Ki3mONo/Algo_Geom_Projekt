from testmanager import *
from code.kdtree import *
from code.Quadtree import *
from code.rectangle import *
from typing import List, Tuple, Optional
import inspect
import sys

all_tests = []
module = sys.modules[__name__]
for name, obj in inspect.getmembers(module):
    if inspect.isfunction(obj) and name.endswith("_test"):
        all_tests.append((obj, {}))

def brut(points: np.ndarray, rectangle: Rectangle) -> List[Tuple[float, float]]:
    res = filter(lambda x: rectangle.contains(x), points)
    return list(res)

def check_if_equal(l1: List[Node], l2: List[Tuple[Tuple[float, float], int]], l3: List[Tuple[float, float]]) -> bool:
    s1 = sum([x.count for x in l1])
    s2 = sum([x[1] for x in l2])
    if s1 != s2 or s1 != len(l3):
        return False
    
    l3 = [tuple(point) for point in np.unique(l3, axis=0)]
    l1_sorted = sorted(l1, key=lambda x: (x.point[0], x.point[1]))
    l2_sorted = sorted(l2, key=lambda x: (x[0][0], x[0][1]))
    l3_sorted = sorted(l3, key=lambda x: (x[0], x[1]))
    
    if len(l1_sorted) != len(l2_sorted) or len(l1_sorted) != len(l3_sorted):
        return False
    
    for i in range(len(l1_sorted)):
        if (abs(l1_sorted[i].point[0] - l2_sorted[i][0][0]) > 1e-12 or
            abs(l1_sorted[i].point[1] - l2_sorted[i][0][1]) > 1e-12):
            return False
    
    for i in range(len(l1_sorted)):
        if (abs(l1_sorted[i].point[0] - l3_sorted[i][0]) > 1e-12 or
            abs(l1_sorted[i].point[1] - l3_sorted[i][1]) > 1e-12):
            return False
    
    return True

def test_methods():
    for test_func, params in all_tests:
        print(f"Running {test_func.__name__}...")
        points, rect = test_func(1000, **params)

        kd = KdTree(points)
        quad = QuadTree(points)
        result_kd = list(kd.search_rectangle(rect))
        result_quad = list(quad.search_rectangle_with_count(rect))
        result_brut = brut(points, rect)

        if check_if_equal(result_kd, result_quad, result_brut):
            print(f"\033[92m\u2714 {test_func.__name__} passed.\033[0m")
        else:
            print(f"\033[91m\u2718 {test_func.__name__} failed.\033[0m")
            print(len(result_kd), len(result_quad), len(result_brut))

if __name__ == "__main__":
    test_methods()