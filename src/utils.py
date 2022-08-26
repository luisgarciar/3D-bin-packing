"""
Utilities for the Bin Packing Problem
"""
from typing import List
from nptyping import NDArray, Int, Shape
import random as rd
import numpy as np
from copy import deepcopy


def boxes_generator(len_bin_edges: List[int], num_items: int = 64, seed: int = 42) -> List[List[int]]:
    """Generates instances of the 2D and 3D bin packing problems

    Parameters
    ----------
    num_items: int, optional
        Number of boxes to be generated (default = 64)
    len_bin_edges: List[int], optional (default=[10,10,10])
        List of length 2 or 3 with the dimensions of the container (default = (10,10,10))
    seed: int, optional
        seed for the random number generator (default = 42)

    Returns
    -------
    List[List[int]]
    A list of length num_items with the dimensions of the randomly generated boxes.
    """
    rd.seed(seed)
    dim = len(len_bin_edges)
    # initialize the items list
    items = [len_bin_edges]

    while len(items) < num_items:
        # choose an item randomly by its size
        box_edges = [np.prod(box) for box in items]
        index = rd.choices(list(range(len(items))), weights=box_edges)[0]
        box0 = items.pop(index)

        # choose an axis (x or y for 2D or x,y,z for 3D) randomly by item edge length
        axis = rd.choices(list(range(dim)), weights=box0)[0]
        len_edge = box0[axis]
        while len_edge == 1:
            axis = rd.choices(list(range(dim)), weights=box0)[0]
            len_edge = box0[axis]

        # choose a splitting point along this axis
        if len_edge == 2:
            split_point = 1
        else:
            dist_edge_center = [abs(x - len_edge / 2) for x in range(1, len_edge)]
            split_point = rd.choices(list(range(1, len_edge)), weights=dist_edge_center)[0]

        # split box0 into box1 and box2 on the split_point on the chosen axis
        box1 = deepcopy(box0)
        box2 = deepcopy(box0)
        box1[axis] = split_point
        box2[axis] = len_edge - split_point
        assert (np.prod(box1) + np.prod(box2)) == np.prod(box0)
        items.extend([box1, box2])

    return items


def generate_vertices(cuboid_len_edges, cuboid_position) -> NDArray[Shape["3, 8"], Int]:
    """Generates the vertices of a box or container in the correct format to be plotted

      Parameters
      ----------
      cuboid_position: List[int]
            List of length 3 with the coordinates of the back-bottom-left vertex of the box or container
      cuboid_len_edges: List[int]
          List of length 3 with the dimensions of the box or container

      Returns
      -------
      np.nd.array(np.int32)
      An array of shape (3,8) with the coordinates of the vertices of the box or container
      """
    # Generate the list of vertices by adding the lengths of the edges to the coordinates
    v0 = cuboid_position
    v0 = np.asarray(v0, dtype=np.int32)
    v1 = v0 + np.asarray([cuboid_len_edges[0], 0, 0], dtype=np.int32)
    v2 = v0 + np.asarray([0, cuboid_len_edges[1], 0], dtype=np.int32)
    v3 = v0 + np.asarray([cuboid_len_edges[0], cuboid_len_edges[1], 0], dtype=np.int32)
    v4 = v0 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v5 = v1 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v6 = v2 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    v7 = v3 + np.asarray([0, 0, cuboid_len_edges[2]], dtype=np.int32)
    vertices = np.vstack((v0, v1, v2, v3, v4, v5, v6, v7))
    return vertices


def interval_intersection(a: List[int], b: List[int]) -> bool:
    """Checks if two open intervals with integer endpoints have a nonempty intersection.

    Parameters
    ----------
    a: List[int]
        List of length 2 with the start and end of the first interval
    b: List[int]
        List of length 2 with the start and end of the second interval

    Returns
    -------
    bool
    True if the intervals intersect, False otherwise
    """
    assert a[1] > a[0], "a[1] must be greater than a[0]"
    assert b[1] > b[0], "b[1] must be greater than b[0]"
    return np.all(np.array([a[0] < b[1], b[0] < a[1]]))


def cuboids_intersection(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Checks if two cuboids have an intersection.

    Parameters
    ----------
    cuboid_a: List[int]
        List of length 6 [xmin_a, y_mina, zmin_a, xmax_a, ymax_a, zmax_a]
        with the start and end coordinates of the first cuboid in each axis

    cuboid_b: List[int]
        List of length 6 [xmin_b, y_minb, zmin_b, xmax_b, ymax_b, zmax_b]
        with the start and end coordinates of the second cuboid in each axis

    Returns
    -------
    bool
    True if the cuboids intersect, False otherwise
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 3"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 3"

    # Check the coordinates of the back-bottom-left vertex of the first cuboid
    assert cuboid_a[0] >= 0, "cuboid_a[0] must be greater than or equal to 0"
    assert cuboid_a[1] >= 0, "cuboid_a[1] must be greater than or equal to 0"
    assert cuboid_a[2] >= 0, "cuboid_a[2] must be greater than or equal to 0"

    # Check the maximum coordinates of the first cuboid
    assert cuboid_a[0] < cuboid_a[3], "cuboid_a[1] must be greater than cuboid_a[0]"
    assert cuboid_a[1] < cuboid_a[4], "cuboid_a[4] must be greater than cuboid_a[1]"
    assert cuboid_a[2] < cuboid_a[5], "cuboid_a[5] must be greater than cuboid_a[2]"

    # Check the coordinates of the back-bottom-left vertex of the second cuboid
    assert cuboid_b[0] >= 0, "cuboid_b[0] must be greater than or equal to 0"
    assert cuboid_b[1] >= 0, "cuboid_b[1] must be greater than or equal to 0"
    assert cuboid_b[2] >= 0, "cuboid_b[2] must be greater than or equal to 0"

    # Check the dimensions of the second cuboid
    assert cuboid_b[0] < cuboid_b[3], "cuboid_b[3] must be greater than cuboid_b[0]"
    assert cuboid_b[1] < cuboid_b[4], "cuboid_b[4] must be greater than cuboid_b[1]"
    assert cuboid_b[2] < cuboid_b[5], "cuboid_b[5] must be greater than cuboid_b[2]"

    inter = [interval_intersection([cuboid_a[0], cuboid_a[3]], [cuboid_b[0], cuboid_b[3]]),
             interval_intersection([cuboid_a[1], cuboid_a[4]], [cuboid_b[1], cuboid_b[4]]),
             interval_intersection([cuboid_a[2], cuboid_a[5]], [cuboid_b[2], cuboid_b[5]])]

    return np.all(inter)


if __name__ == "__main__":
    pass
