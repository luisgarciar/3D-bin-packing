"""
Utilities for the Bin Packing Problem
"""
from typing import List, Type, Tuple, Any
import random as rd
import numpy as np
from copy import deepcopy
from packing_engine import Box, Container
import plotly.colors as colors
import plotly.graph_objects as go


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


def generate_box_vertices(box: Type[Box]) -> Tuple[Any, Any, Any]:
    """Generates the eight vertices of a box in the correct format to be plotted

      Parameters
      ----------
      box: Type[Box]
          A Box object

      Returns
      -------
      List[nd.array]
      A list of length three with the x,y,z coordinates of the box vertices
      """
    # Generate the list of vertices by adding the lengths of the edges in the coordinates
    v0 = box.position
    v0 = np.asarray(v0, dtype=np.int32)
    v1 = v0 + np.asarray([box.len_edges[0], 0, 0], dtype=np.int32)
    v2 = v0 + np.asarray([0, box.len_edges[1], 0], dtype=np.int32)
    v3 = v0 + np.asarray([box.len_edges[0], box.len_edges[1], 0], dtype=np.int32)
    v4 = v0 + np.asarray([0, 0, box.len_edges[2]], dtype=np.int32)
    v5 = v1 + np.asarray([0, 0, box.len_edges[2]], dtype=np.int32)
    v6 = v2 + np.asarray([0, 0, box.len_edges[2]], dtype=np.int32)
    v7 = v3 + np.asarray([0, 0, box.len_edges[2]], dtype=np.int32)

    vertices = np.vstack((v0, v1, v2, v3, v4, v5, v6, v7)).T
    x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
    return x, y, z


def plot_box(box: Type[Box], figure: Type[go.Figure] = None) -> Type[go.Figure]:
    """Adds the plot of a box to a given figure

         Parameters
         ----------
         box: Type[Box]
             A Box object
         figure:
             A plotly figure where the box should be plotted

         Returns
         -------
         Type[go.Figure]
         """
    # Generate the coordinates of the vertices
    x, y, z = generate_box_vertices(box)
    # The arrays i, j, k contain the indices of the triangles to be plotted (two per face of the box)
    # The triangles have vertices (x[i[index]], y[j[index]], z[k[index]])
    i = [1, 2, 5, 6, 1, 4, 3, 6, 1, 7, 0, 6]
    j = [0, 3, 4, 7, 0, 5, 2, 7, 3, 5, 2, 4]
    k = [2, 1, 6, 5, 4, 1, 6, 3, 7, 1, 6, 0]
    if figure is None:
        figure = go.Figure(data=[
            go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                      opacity=0.6, color='#DC143C',
                      flatshading=True
                      )
        ]
        )
        figure.update_layout(
            scene=dict(
                xaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(x) + 1]),
                yaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(y) + 1]),
                zaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(z) + 1]),
                aspectmode='cube'),
            width=800,
            margin=dict(r=20, l=10, b=10, t=10),
        )
    else:
        figure.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                   opacity=0.6, color='#DC143C',
                                   flatshading=True
                                   )
                         )
    return figure

def plot_container(container: Type[Container], figure: Type[go.Figure] = None) -> Type[go.Figure]:
    position = container.position
    len_edges = self.len_edges

    pass

    # figure.add_trace(go.Scatter3d(x =[position_0, ]   ))


    pass


if __name__ == "__main__":
    box = Box(id_= 0, position=[0, 0, 0], len_edges=[1, 2, 3])
    fig = plot_box(box)
    fig.show()


