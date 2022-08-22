import pytest
import numpy as np
from src.packing_engine import Box, Container
from src.utils import boxes_generator
from src.packing_env import PackingEnv0
import random as rd
from numpy.testing import assert_array_equal

num_rd_tests = 5
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
box_sizes = [[6, 6, 6], [6, 6, 6], [6, 6, 6], [6, 6, 6], [6, 6, 6]]
box_sizes_list = [box_sizes for i in range(num_rd_tests)]
container_size = zip(height, length, width)
num_incoming_boxes = rd.sample(range(1, 1 + num_rd_tests), num_rd_tests)
test_data1 = zip(container_size, box_sizes_list, num_incoming_boxes)

@pytest.mark.parametrize("container_size, box_sizes, num_incoming_boxes", test_data1)
def test_env_initialization_random_data(container_size, box_sizes, num_incoming_boxes):
    env = PackingEnv0(container_size, box_sizes, num_incoming_boxes, seed=42, gen_action_mask=True)
    assert_array_equal(env.observation_space['height_map'].nvec, np.array([container_size[0],  container_size[1]]))
    assert env.observation_space.spaces['height_map'].shape == (container_size[0], container_size[1])
    assert env.observation_space.spaces['incoming_box_sizes'].shape == (num_incoming_boxes, 3)
    assert_array_equal(env.action_space['position'].nvec, np.array([container_size[0],  container_size[1]]))
    assert env.action_space.spaces['position'].shape == (2,)
    #assert env.action_space.spaces['action_mask'].shape == (2, 1)


