"""
Utilities for the Bin Packing Problem
"""
from typing import List, Type #Tuple, Any, Union
from nptyping import NDArray, Int, Shape
import random as rd
import numpy as np
from copy import deepcopy
import plotly.graph_objects as go
# from src.packing_engine import Box, Container


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

        # choose a position randomly on the axis by the distance to the center of edge
        if len_edge == 1:
            items.append(box0)
            continue
        elif len_edge == 2:
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


if __name__ == "__main__":
    box = Box([1, 2, 3], position=[0, 0, 0], id_=0)
    container = Container([10, 10, 10], [0, 0, 0], id_=0)
    container.boxes.append(box)
    fig = plot_container(container)
    fig.show()
