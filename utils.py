"""
Utilities for the Bin Packing Problem
"""
from typing import List
import random as rd
import numpy as np


def boxes_generator(numitems: int = 64, binsize: List[int] = [10, 10, 10], seed: int = 42) -> List[List[int]]:
    """Generates instances of the 2D and 3D bin packing problems

    Parameters
    ----------
    numitems: int, optional
        Number of boxes to be generated (default = 64)
    binsize: List[int], optional
        List of length 2 or 3 with the dimensions of the container (default = (10,10,10))
    seed: int, optional
        seed for the random number generator (default = 42)

    Returns
    -------
    List[List[int]]
    A list of length numitems with the dimensions of the randomly generated boxes.
    """
    rd.seed(seed)
    dim = len(binsize)
    # initialize the items list
    items = [binsize]

    while len(items) < numitems:
        # choose an item randomly by its size
        box_size = [np.product(box) for box in items]
        index = rd.choices(list(range(len(items))), weights=box_size)[0]
        box0 = items.pop(index)
        # choose an axis (among x or y for 2D,among x,y,z for 3D) randomly by item edge length
        axis = rd.choices(list(range(dim)), weights=box0)[0]
        len_edge = box0[axis]

        # choose a position randomly on the axis by the distance to the center of edge (does this make sense?)
        if len_edge == 1:
            continue
        elif len_edge == 2:
            split_point = 1
        else:
            dist_edge_center = [abs(x - len_edge/2) for x in range(1, len_edge)]
            split_point = rd.choices(list(range(1, len_edge)), weights=dist_edge_center)[0]

        # split box0 into box1 and box2 according to split_point on the chosen axis
        box1 = box0.copy()
        box2 = box0.copy()
        box1[axis] = split_point
        box2[axis] = len_edge - split_point
        items.extend([box1, box2])

    return items


if __name__ == "__main__":
    boxes = boxes_generator()
    print(boxes)


