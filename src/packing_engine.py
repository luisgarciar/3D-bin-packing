"""
Packing Engine: Basic Classes for the Bin Packing Problem
We follow the space representation depicted below, all coordinates and lengths of boxes and containers are integers.

    x: depth
    y: length
    z: height

       Z
       |
       |
       |________Y
      /
     /
    X

    Classes:
        Box
        Container

"""
import numpy as np
from copy import deepcopy
from typing import List, Type, Union


class Box:
    """ A class to represent a 3D box

     Attributes
     ----------
      id_: int
            id of the box
      position: int
            Coordinates of the position of the bottom-leftmost-deepest corner of the box
      len_edges: int
            Lengths of the edges of the box
     """

    def __init__(self, len_edges: List[int], position: List[int], id_: int) -> None:
        """ Initializes a box object

        Parameters
        ----------
        id_: int
            id of the box
        position: List[int]
            Coordinates of the position of the bottom-leftmost-deepest corner of the box
        len_edges: List[int]
            Lengths of the edges of the box

        Returns
        -------
        Box object
        """
        assert len(len_edges) == len(position), "Sizes of len_edges and position do not match"
        assert len(len_edges) == 3, "Size of len_edges is different from 3"

        self.position = position
        self.len_edges = len_edges
        self.id_ = id_

    def rotate(self, rotation: int) -> None:
        """Rotates the box in place

        Parameters
        ----------
        rotation: int
        """
        pass  # to be added later


class Container:
    """ A class to represent a 3D container

    Attributes
    ----------
    id_: int
        id of the container
    len_edges: List[int]
        Lengths of the edges of the container
    position: List[int], optional
        Coordinates of the bottom-leftmost-deepest corner of the container (default = [0,0,0])
    boxes: List[Type[Box]]
        List with the boxes placed inside the container
    height_map: np.array
        An array of size (len_x,len_y) representing the height map (top view) of the container
    """

    def __init__(self, len_edges: List[int], position=None, id_: int = 0) -> None:
        """Initializes a 3D container

        Parameters
        ----------
        id_: int, optional
            id of the container (default = 0)
        positions: int, optional
            Coordinates of the bottom-leftmost-deepest corner of the container (default = 0,0,0)
        len_edges: int
            Lengths of the edges of the container
        """

        if position is None:
            position = [0, 0, 0]

        assert len(len_edges) == len(position), "Sizes of len_edges and position do not match"
        assert len(len_edges) == 3, "Size of len_edges is different from 3"

        self.id_ = id_
        self.position = position
        self.len_edges = len_edges
        self.boxes = []
        self.height_map = np.zeros(shape=(len_edges[0], len_edges[1]))

    def reset(self):
        """Resets the container to an empty state"""
        self.boxes = []
        self.height_map = np.zeros(shape=self.height_map.shape())

    def update_height_map(self, box): #make static method??
        """ Updates the height map after placing a box
         Parameters
        ----------
        box: Box
        Box to be placed inside the container
        """
        pass

    def get_height_map(self):
        """ Returns a copy of the height map of the container"""
        return deepcopy(self.height_map)

    def check_valid_box_placement(self, box: Type[Box], new_position: List[int]): #Add different checkmodes?
        pass

    def all_possible_positions(self, box: Type[Box]):
        pass

    def place_box(self, box: Type[Box], new_position: List[int]):
        pass











