import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points in the unit square"""
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Returns a Graph where the randomly-generated points are seperated by x coordinate (lt or ge to .5)

    Args:
    ----
        N (int): Number of points in the graph

    Returns:
    -------
        Graph: Graph with N points seperated by x coordinate (lt or ge to .5)

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Returns a Graph where the randomly-generated points are seperated by a diagonal line

    Args:
    ----
        N (int): Number of points in the graph

    Returns:
    -------
        Graph: Graph with N points seperated by a diagonal line

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Returns a Graph where the randomly-generated points are seperated by x coordiate (either in [.2, .8] or not)


    Args:
    ----
        N (int): Number of points in the graph

    Returns:
    -------
        Graph: Graph with N points seperated by x coordinate (either in [.2, .8] or not)

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Returns a Graph where the randomly-generated points are seperated by quadrants (1 and 3 versus 2 and 4)

    Args:
    ----
        N (int): Number of points in the graph

    Returns:
    -------
        Graph: Graph with N points seperated by quadrants (1 and 3 versus 2 and 4)

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Returns a Graph where the randomly-generated points are seperated by a circle

    Args:
    ----
         N (int): Number of points in the graph

    Returns:
    -------
         Graph: Graph with N points seperated by a circle

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Returns a Graph where the points are generated upon two spirals

    Args:
    ----
         N (int): Number of points in the graph

    Returns:
    -------
         Graph: Graph with N points seperated by the spiral upon which they were generated

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
