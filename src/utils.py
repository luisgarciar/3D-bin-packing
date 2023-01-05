"""
Utilities for the Bin Packing Problem
"""
import random as rd
from copy import deepcopy
from typing import List

import numpy as np
from nptyping import NDArray, Int, Shape


def boxes_generator(
    bin_size: List[int], num_items: int = 64, seed: int = 42
) -> List[List[int]]:
    """Generates instances of the 2D and 3D bin packing problems

    Parameters
    ----------
    num_items: int, optional
        Number of boxes to be generated (default = 64)
    bin_size: List[int], optional (default = [10,10,10])
        List of length 2 or 3 with the dimensions of the container (default = (10,10,10))
    seed: int, optional
        seed for the random number generator (default = 42)

    Returns
    -------
    List[List[int]]
    A list of length num_items with the dimensions of the randomly generated boxes.
    """
    rd.seed(seed)

    dim = len(bin_size)
    # initialize the items list
    item_sizes = [bin_size]

    while len(item_sizes) < num_items:
        # choose an item randomly by its volume
        box_vols = [np.prod(np.array(box_size)) for box_size in item_sizes]
        index = rd.choices(list(range(len(item_sizes))), weights=box_vols, k=1)[0]
        box0_size = item_sizes.pop(index)

        # choose an axis (x or y for 2D or x,y,z for 3D) randomly by item edge length
        axis = rd.choices(list(range(dim)), weights=box0_size, k=1)[0]
        len_edge = box0_size[axis]
        while len_edge == 1:
            axis = rd.choices(list(range(dim)), weights=box0_size, k=1)[0]
            len_edge = box0_size[axis]

        # choose a splitting point along this axis
        if len_edge == 2:
            split_point = 1
        else:
            dist_edge_center = [abs(x - len_edge / 2) for x in range(1, len_edge)]
            weights = np.reciprocal(np.asarray(dist_edge_center) + 1)
            split_point = rd.choices(list(range(1, len_edge)), weights=weights, k=1)[0]

        # split box0 into box1 and box2 on the split_point on the chosen axis
        box1 = deepcopy(box0_size)
        box2 = deepcopy(box0_size)
        box1[axis] = split_point
        box2[axis] = len_edge - split_point
        assert (np.prod(box1) + np.prod(box2)) == np.prod(box0_size)

        # rotate boxes on the longest side
        # add boxes to the list of items
        # box1.sort(reverse=True)
        # box2.sort(reverse=True)
        item_sizes.extend([box1, box2])

    return item_sizes


def generate_vertices(
    cuboid_len_edges: NDArray, cuboid_position: NDArray
) -> NDArray[Shape["3, 8"], Int]:
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
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def cuboids_intersection(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Checks if two cuboids have an intersection.

    Parameters
    ----------
    cuboid_a: List[int]
        List of length 6 [x_min_a, y_mina, z_min_a, x_max_a, y_max_a, z_max_a]
        with the start and end coordinates of the first cuboid in each axis

    cuboid_b: List[int]
        List of length 6 [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        with the start and end coordinates of the second cuboid in each axis

    Returns
    -------
    bool
    True if the cuboids intersect, False otherwise
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"

    # Check the coordinates of the back-bottom-left vertex of the first cuboid
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have nonnegative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    inter = [
        interval_intersection([cuboid_a[0], cuboid_a[3]], [cuboid_b[0], cuboid_b[3]]),
        interval_intersection([cuboid_a[1], cuboid_a[4]], [cuboid_b[1], cuboid_b[4]]),
        interval_intersection([cuboid_a[2], cuboid_a[5]], [cuboid_b[2], cuboid_b[5]]),
    ]

    return np.all(inter)


def cuboid_fits(cuboid_a: List[int], cuboid_b: List[int]) -> bool:
    """Checks if cuboid_b fits into cuboid_a.
    Parameters
    ----------
    cuboid_a: List[int]
        List of length 6 [x_min_a, y_mina, z_min_a, x_max_a, y_max_a, z_max_a]
        with the start and end coordinates of the first cuboid in each axis
    cuboid_b: List[int]
        List of length 6 [x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b]
        with the start and end coordinates of the second cuboid in each axis
    Returns
    -------
    bool
    True if the cuboid_b fits into cuboid_a, False otherwise
    """
    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 3"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 3"

    assert len(cuboid_a) == 6, "cuboid_a must be a list of length 6"
    assert len(cuboid_b) == 6, "cuboid_b must be a list of length 6"

    # Check the coordinates of the back-bottom-left vertex of the first cuboid
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_a[:3])
    ), "cuboid_a must have non-negative coordinates"
    assert np.all(
        np.less_equal([0, 0, 0], cuboid_b[:3])
    ), "cuboid_b must have non-negative coordinates"

    assert np.all(
        np.less(cuboid_a[:3], cuboid_a[3:])
    ), "cuboid_a must have non-zero volume"

    assert np.all(
        np.less(cuboid_b[:3], cuboid_b[3:])
    ), "cuboid_b must have non-zero volume"

    # Check if the cuboid b fits into the cuboid a
    return np.all(np.less_equal(cuboid_a[:3], cuboid_b[:3])) and np.all(
        np.less_equal(cuboid_b[3:], cuboid_a[3:])
    )


if __name__ == "__main__":
    pass
