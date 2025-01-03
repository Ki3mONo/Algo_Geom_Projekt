from testmanager import *
from code.kdtree import *
from code.Quadtree import *
from code.rectangle import *
from typing import List, Tuple, Optional
import inspect
import sys
import timeit

all_tests = []
module = sys.modules[__name__]
for name, obj in inspect.getmembers(module):
    if inspect.isfunction(obj) and name.find("test") != -1:
        all_tests.append((obj, {}))

def brut(points: np.ndarray, rectangle: Rectangle) -> List[Tuple[float, float]]:
    res = filter(lambda x: rectangle.contains(x), points)
    return list(res)

def check_if_equal(l1: List["Node"], l2: List[Tuple[Tuple[float, float], int]], l3: List[Tuple[float, float]]) -> bool:
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
    with open("times.csv", "w") as f:
        f.write("test_func,size,kd_time,quad_time,kd_query_time,quad_query_time\n")
    for test_func, params in all_tests:
        for size in [1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000]:
            print(f"Running {test_func.__name__} with n = {size}...")
            points, rect = test_func(size, **params)
            
            reps = 5
            kd_time = timeit.timeit(lambda: KdTree(points), number=reps)/reps
            kd = KdTree(points)
            quad_time = timeit.timeit(lambda: QuadTree(points), number=reps)/reps
            quad = QuadTree(points)
            kd_query_time = timeit.timeit(lambda: list(kd.search_rectangle(rect)), number=reps)/reps
            result_kd = list(kd.search_rectangle(rect))
            quad_query_time = timeit.timeit(lambda: list(quad.search_rectangle_with_count(rect)), number=reps)/reps
            result_quad = list(quad.search_rectangle_with_count(rect))
            result_brut = brut(points, rect)

            if check_if_equal(result_kd, result_quad, result_brut):
                print(f"\033[92m\u2714 {test_func.__name__} passed.\033[0m")
            else:
                print(f"\033[91m\u2718 {test_func.__name__} failed.\033[0m")
                print(len(result_kd), len(result_quad), len(result_brut))

            print(f"Build Times (averaged over {reps} runs):")
            print(f"  KdTree = {kd_time:.6f}s per run")
            print(f"  QuadTree = {quad_time:.6f}s per run")
            print(f"Query Times (averaged over {reps} runs):")
            print(f"  KdTree = {kd_query_time:.6f}s per run")
            print(f"  QuadTree = {quad_query_time:.6f}s per run")
            print("-" * 60)

            with open("times.csv", "a") as f:
                f.write(f"{test_func.__name__},{size},{kd_time},{quad_time},{kd_query_time},{quad_query_time}\n")

if __name__ == "__main__":
    test_methods()