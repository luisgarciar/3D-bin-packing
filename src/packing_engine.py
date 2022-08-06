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
    Xn

    Classes:
        Box
        Container

"""
import numpy as np
from copy import deepcopy
from typing import List, Type, Union
from utils import generate_vertices


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

        assert (len_edges[0] > 0 and len_edges[1] > 0 and len_edges[2] > 0)
        assert (position[0] >= 0 and len_edges[1] >= 0 and len_edges[2] >= 0)

        self.position = np.asarray(position)
        self.len_edges = np.asarray(len_edges)
        self.id_ = id_

    def rotate(self, rotation: int) -> None:
        """Rotates the box in place

        Parameters
        ----------
        rotation: int
        """
        pass  # to be added later

    @property
    def area_bottom(self) -> int:
        """ Area of the bottom face of the box """
        return self.len_edges[0] * self.len_edges[1]

    @property
    def vertices(self):
        """Returns a list with the vertices of the box"""
        vert = generate_vertices(self)
        return list(vert)


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
            position = np.zeros(shape=3, dtype=np.int32)

        assert len(len_edges) == len(position), "Sizes of len_edges and position do not match"
        assert len(len_edges) == 3, "Size of len_edges is different from 3"

        self.id_ = id_
        self.position = np.asarray(position, dtype=np.int32)
        self.len_edges = np.asarray(len_edges, dtype=np.int32)
        self.boxes = []
        self.height_map = np.zeros(shape=(len_edges[0], len_edges[1]), dtype=np.int32)

    @property
    def vertices(self):
        """Returns a list with the vertices of the container"""
        vert = generate_vertices(self)
        return list(vert)

    def reset(self):
        """Resets the container to an empty state"""
        self.boxes = []
        self.height_map = np.zeros_like(self.height_map.shape(), dtype=np.int32)

    def _update_height_map(self, box):  # make static method??
        """Updates the height map after placing a box
         Parameters
        ----------
        box: Box
            Box to be placed inside the container
        """
        # Add the height of the new box in the x-y coordinates occupied by the box
        self.height_map[box.position[0]: box.position[0] + box.len_edges[0],
        box.position[1]: box.position[1] + box.len_edges[1]] += box.len_edges[2]

    def get_height_map(self):
        """ Returns a copy of the height map of the container"""
        return deepcopy(self.height_map)

    def check_valid_box_placement(self, box: Type[Box], new_position: List[int],
                                  check_area: int = 100) -> int:  # Add different checkmodes?
        """
        Parameters
        ----------
        box: Box
            Box to be placed
        new_position: List[int]
            Coordinates of new position
        check_area: int, default = 100
             Percentage of area of the bottom of the box that must be supported in the new position

        Returns
        -------
        int
        """
        # Generate vertices of the bottom face of the box
        assert len(new_position) == 2

        # TO DO: Clean this up -- generate vertices without creating a dummy box
        dummy_box = deepcopy(box)
        dummy_box.position = [new_position, 1] # the last coordinate does not matter

        # Check that all bottom vertices of the box in the new position are at the same level
        [v0, v1, v2, v3] = dummy_box.vertices[0:3]  # list with bottom vertices
        corners_levs = [self.height_map[v0[0], v0[1]], self.height_map[v1[0], v1[1]],
                        self.height_map[v2[0], v2[1]], self.height_map[v3[0], v3[1]]]

        if corners_levs.count(corners_levs[0]) != len(corners_levs):
            return 0

        # lev is the level (height) at which the bottom corners of the box will be located
        lev = corners_levs[0]
        # bottom_face_lev contains the levels of all the points in the bottom face
        bottom_face_lev = self.height_map[v0[0]:v0[0] + box.len_edges[0], v0[1]:v0[1] + box.len_edges[1]]

        # Check that the level of the corners is the maximum of all points in the bottom face
        if lev != np.amax(lev):
            return 0

        # we count how many of these points are located at height equal to lev
        count_level = np.count_nonzero(bottom_face_lev == lev)
        # Check the percentage of box bottom area that is supported (at the height equal to lev)
        support_perc = count_level / dummy_box.area_bottom
        if support_perc < check_area:
            return 0

         # Check that the box fits in the container in the new location
        fit_x_axis = dummy_box.position[0] >= self.position[0] and \
                     dummy_box.position[0] + dummy_box.len_edges[0] <= self.position[0] + self.len_edges[0]

        fit_y_axis = dummy_box.position[1] >= self.position[1] and \
                         dummy_box.position[1] + dummy_box.len_edges[1] <= self.position[1] + self.len_edges[1]

        fit_z_axis = dummy_box.position[2] >= self.position[2] and \
                    dummy_box.position[2] + dummy_box.len_edges[2] <= self.position[2] + self.len_edges[0]

        return 1

    def all_possible_positions(self, box: Type[Box]):
        pass

    def place_box(self, box: Type[Box], new_position: List[int]):
        pass
