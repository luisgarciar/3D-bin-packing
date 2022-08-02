import pytest
import numpy as np
from src.packing_engine import Box, Container
import random as rd

rd.seed(42)
num_rd_tests = 20
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
pos_x = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_y = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_z = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)

length_edges = [[height[j], length[j], width[j]] for j in range(num_rd_tests)]
positions = [[pos_x[j], pos_y[j], pos_z[j]] for j in range(num_rd_tests)]
testdata = [(length_edges[j], positions[j]) for j in range(num_rd_tests)]


# Test of initialization of Box class
@pytest.mark.parametrize("len_edges,position", testdata)
def test_box_initialization_random_data(len_edges, position):
    box = Box(len_edges, position, 0)
    assert (box.len_edges == len_edges and box.position == position and box.id_ == 0)


# Test of initialization of Container class
@pytest.mark.parametrize("len_edges,position", testdata)
def test_container_initialization_random_data(len_edges, position):
    container = Container(len_edges, position, 0)
    assert (container.len_edges == len_edges and container.position == position and container.id_ == 0
            and container.height_map.shape == (len_edges[0], len_edges[1]))

# Test of update_height_map with an empty container and a small box
# (all dims of box smaller than half the dims of container)


# Test of update_height_map with an empty container and a large box
# (one dim of box equals one dim of container)


# Test of update_height_map with a container with one box and a new box stacked on top


# Test of update_height_map with a container with one box and a new box stacked next along the x dimension


