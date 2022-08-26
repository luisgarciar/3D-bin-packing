import pytest
import numpy as np
from src.packing_env import PackingEnv0
from src.packing_engine import Box, Container
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
    assert len(env.unpacked_boxes) == 5
    assert env.observation_space.spaces['height_map'].shape == (container_size[0], container_size[1])
    assert env.observation_space.spaces['incoming_box_sizes'].shape == (num_incoming_boxes, 3)
    assert_array_equal(env.action_space['position'].nvec, np.array([container_size[0],  container_size[1]]))
    assert env.action_space.spaces['position'].shape == (2,)


@pytest.fixture
def basic_environment():
    container_size = [10, 10, 10]
    box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    num_incoming_boxes = 1
    env = PackingEnv0(container_size, box_sizes, num_incoming_boxes, seed=42, gen_action_mask=True)
    return env


def test_env_reset(basic_environment):
    env = basic_environment
    obs = env.reset(seed=42)
    assert len(env.unpacked_boxes) == 4
    assert env.observation_space['height_map'].shape == (10, 10)
    assert env.observation_space['incoming_box_sizes'].shape == (1, 3)
    assert env.action_space['position'].shape == (2,)
    assert_array_equal(obs['height_map'], np.zeros((10, 10)))


def test_step(basic_environment):
    env = basic_environment
    obs = env.reset(seed=42)
    assert_array_equal(obs['incoming_box_sizes'], np.array([[3, 3, 3]]))
    action = {'position': [0, 0], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    assert obs['height_map'][0, 0] == 3
    assert len(env.unpacked_boxes) == 3
    assert len(env.container.boxes) == 1
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box0 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[0].position, np.asarray([0, 0, 0]))
    # Check that the height map is correct
    hm = np.zeros((10, 10), dtype=np.int32)
    hm[0:3, 0:3] = 3
    np.testing.assert_array_equal(env.container.get_height_map(), hm)

    # Check the size of the next incoming box (box1)
    assert_array_equal(obs['incoming_box_sizes'], np.array([[3, 2, 3]]))
    # Set an action that is allowed
    action = {'position': [0, 0], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    assert obs['height_map'][0, 0] == 6
    assert len(env.unpacked_boxes) == 2
    assert len(env.container.boxes) == 2
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box1 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[1].position, np.asarray([0, 0, 3]))

    # Check the size of the next incoming box (box2)
    assert_array_equal(obs['incoming_box_sizes'], np.array([[3, 4, 2]]))
    # Set an action that is allowed
    action = {'position': [3, 0], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    # Check the height map after the action
    hm2 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm2[0:3, 0:2] = 6
    hm2[0:3, 2] = 3
    hm2[3:6, 0:4] = 2
    np.testing.assert_array_equal(obs['height_map'], hm2)
    assert len(env.unpacked_boxes) == 1
    assert len(env.container.boxes) == 3
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box2 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[2].position, np.asarray([3, 0, 0]))

    # Check the size of the next incoming box (box3)
    assert_array_equal(obs['incoming_box_sizes'], np.array([[3, 2, 4]]))
    # Set an action that is allowed
    action = {'position': [0, 3], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    # Check the height map after the action
    hm3 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm3[0:3, 0:2] = 6
    hm3[0:3, 2] = 3
    hm3[3:6, 0:4] = 2
    hm3[0:3, 3:5] = 4
    np.testing.assert_array_equal(obs['height_map'], hm3)
    assert len(env.unpacked_boxes) == 0
    assert len(env.container.boxes) == 4
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box3 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[3].position, np.asarray([0, 3, 0]))

    # Check the size of the next incoming box (box4)
    assert_array_equal(obs['incoming_box_sizes'], np.array([[3, 2, 3]]))
    # Set an action that is allowed
    action = {'position': [0, 5], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    # Check the height map after the action
    hm4 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm4[0:3, 0:2] = 6
    hm4[0:3, 2] = 3
    hm4[3:6, 0:4] = 2
    hm4[0:3, 3:5] = 4
    hm4[0:3, 5:7] = 3
    np.testing.assert_array_equal(obs['height_map'], hm4)
    assert len(env.unpacked_boxes) == 0
    assert len(env.container.boxes) == 5
    assert reward == 1
    assert truncated is False
    assert terminated is True













