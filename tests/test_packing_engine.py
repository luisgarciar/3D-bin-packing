import pytest
import numpy as np
from src.packing_engine import Box, Container
import random as rd

rd.seed(42)
num_rd_tests = 5
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
pos_x = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_y = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_z = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)

length_edges = zip(height, length, width)
positions = zip(pos_x, pos_y, pos_z)
test_data1 = zip(length_edges, positions)
test_data2 = zip(length_edges, positions)


# # Test of initialization of Container class
@pytest.mark.parametrize("len_edges, position", test_data1)
def test_container_initialization_random_data(len_edges, position):
    container = Container(len_edges, position, 0)
    assert np.array_equal(container.len_edges, len_edges)
    assert np.array_equal(container.position, position)
    assert container.id_ == 0
    assert container.height_map.shape == (len_edges[0], len_edges[1])

num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
pos_x = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_y = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)
pos_z = rd.sample(range(0, num_rd_tests + 1), num_rd_tests)

length_edges = zip(height, length, width)
positions = zip(pos_x, pos_y, pos_z)
test_data2 = zip(length_edges, positions)


# Test of initialization of Box class
@pytest.mark.parametrize("len_edges, position", test_data2)
def test_box_initialization_random_data(len_edges, position):
    box = Box(len_edges, position, 0)
    assert np.array_equal(box.len_edges, len_edges)
    assert np.array_equal(box.position, position)
    assert box.id_ == 0


# Test of update_height_map
container_len_edges = [[6, 6, 10], [6, 8, 10]]
box_len_edges = [[2, 2, 5], [2, 8, 3]]
box_pos = [[3, 3, 0], [2, 0, 0]]
hm1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 0],
                [0, 0, 0, 5, 5, 0], [0, 0, 0, 0, 0, 0]], dtype=np.int32)
hm2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3, 3],
               [3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
height_map = [hm1, hm2]
test_data3 = zip(container_len_edges, box_len_edges, box_pos, height_map)


@pytest.mark.parametrize("container_len_edges, box_len_edges, box_pos, height_map", test_data3)
def test_update_height_map(container_len_edges, box_len_edges, box_pos, height_map):
    container = Container(container_len_edges)
    box = Box(box_len_edges, box_pos, 0)
    container._update_height_map(box)
    new_height_map = container.height_map
    assert np.array_equal(container.height_map, height_map)



