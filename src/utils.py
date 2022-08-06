"""
Utilities for the Bin Packing Problem
"""
from typing import List, Type, Tuple, Any, Union
import random as rd
import numpy as np
from copy import deepcopy
from src.packing_engine import Box, Container
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


def generate_vertices(cuboid: Union[Type[Box], Type[Container]]) -> Tuple[Any, Any, Any]:
    """Generates the vertices of a box or container in the correct format to be plotted

      Parameters
      ----------
      cuboid: Type[Box] or Type[Container]
          A cuboid (box or container object)

      Returns
      -------
      List[nd.array]
      A list of length three with the x,y,z coordinates of the box vertices
      """
    # Generate the list of vertices by adding the lengths of the edges to the coordinates
    v0 = cuboid.position
    v0 = np.asarray(v0, dtype=np.int32)
    v1 = v0 + np.asarray([cuboid.len_edges[0], 0, 0], dtype=np.int32)
    v2 = v0 + np.asarray([0, cuboid.len_edges[1], 0], dtype=np.int32)
    v3 = v0 + np.asarray([cuboid.len_edges[0], cuboid.len_edges[1], 0], dtype=np.int32)
    v4 = v0 + np.asarray([0, 0, cuboid.len_edges[2]], dtype=np.int32)
    v5 = v1 + np.asarray([0, 0, cuboid.len_edges[2]], dtype=np.int32)
    v6 = v2 + np.asarray([0, 0, cuboid.len_edges[2]], dtype=np.int32)
    v7 = v3 + np.asarray([0, 0, cuboid.len_edges[2]], dtype=np.int32)
    vertices = np.vstack((v0, v1, v2, v3, v4, v5, v6, v7)).T
    return vertices


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
    vertices = generate_vertices(box)
    x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
    # The arrays i, j, k contain the indices of the triangles to be plotted (two per each face of the box)
    # The triangles have vertices (x[i[index]], y[j[index]], z[k[index]]), index = 0,1,..7.
    i = [1, 2, 5, 6, 1, 4, 3, 6, 1, 7, 0, 6]
    j = [0, 3, 4, 7, 0, 5, 2, 7, 3, 5, 2, 4]
    k = [2, 1, 6, 5, 4, 1, 6, 3, 7, 1, 6, 0]

    if figure is None:
        figure = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                           opacity=0.6, color='#DC143C',
                                           flatshading=True)])
        figure.update_layout(scene=dict(xaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(x) + 1]),
                                        yaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(y) + 1]),
                                        zaxis=dict(nticks=int(np.max(x) + 2), range=[0, np.max(z) + 1]),
                                        aspectmode='cube'), width=800, margin=dict(r=20, l=10, b=10, t=10))
    else:
        figure.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.6, color='#DC143C',
                                   flatshading=True))
    return figure


def plot_container(container: Type[Container], figure: Type[go.Figure] = None) -> Type[go.Figure]:
    """Adds the plot of a container to a given figure

            Parameters
            ----------
            container: Type[Container]
                A Container object
            figure:
                A plotly figure where the box should be plotted

            Returns
            -------
            Type[go.Figure]
            """
    if figure is None:
        figure = go.Figure()

    # Generate all vertices and edge pairs, the numbering is explained in the function generate_vertices
    vertices = generate_vertices(container)
    x, y, z = vertices[0, :], vertices[1, :], vertices[2, :]
    edge_pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    # Add a line between each pair of edges to the figure
    for (i, j) in edge_pairs:
        vert_x = np.array([x[i], x[j]])
        vert_y = np.array([y[i], y[j]])
        vert_z = np.array([z[i], z[j]])
        figure.add_trace(go.Scatter3d(x=vert_x, y=vert_y, z=vert_z, mode='lines', line=dict(color='black', width=2)))

    for box in container.boxes:
        figure = plot_box(box, figure)

    # Choose the visualization angle
    camera = dict(eye=dict(x=2, y=2, z=0.1))

    # Update figure properties for improved visualization
    figure.update_layout(showlegend=False, scene_camera=camera)
    return figure


if __name__ == "__main__":
    box = Box([1, 2, 3], position=[0, 0, 0], id_=0)
    container = Container([10, 10, 10], [0, 0, 0], id_=0)
    container.boxes.append(box)
    fig = plot_container(container)
    fig.show()
