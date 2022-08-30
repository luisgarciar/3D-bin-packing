import pytest
import numpy as np
from src.packing_env import PackingEnv0
from src.packing_engine import Box, Container
from src.utils import boxes_generator
import random as rd
from numpy.testing import assert_array_equal
from gym import make
from gym.utils.env_checker import check_env


num_rd_tests = 5
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(1, 1 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(2, 2 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(3, 3 + num_rd_tests + 1), num_rd_tests)
box_sizes = [[6, 6, 6], [6, 6, 6], [6, 6, 6], [6, 6, 6], [6, 6, 6]]
box_sizes_list = [box_sizes for i in range(num_rd_tests)]
container_size = zip(height, length, width)
num_visible_boxes = rd.sample(range(1, 1 + num_rd_tests), num_rd_tests)
test_data1 = zip(container_size, box_sizes_list, num_visible_boxes)


@pytest.mark.parametrize("container_size, box_sizes, num_visible_boxes", test_data1)
def test_env_initialization_random_data(container_size, box_sizes, num_visible_boxes):
    env = PackingEnv0(container_size, box_sizes, num_visible_boxes)
    assert len(env.unpacked_hidden_boxes) == 5
    assert env.observation_space.spaces['height_map'].shape == (container_size[0], container_size[1])
    assert env.observation_space.spaces['visible_box_sizes'].shape == (num_visible_boxes, 3)
    assert_array_equal(env.action_space['position'].nvec, np.array([container_size[0],  container_size[1]]))
    assert env.action_space.spaces['position'].shape == (2,)


@pytest.fixture
def basic_environment():
    container_size = [10, 10, 10]
    box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    num_visible_boxes = 1
    env = PackingEnv0(container_size, box_sizes, num_visible_boxes)
    return env


def test_env_reset(basic_environment):
    env = basic_environment
    obs = env.reset(seed=42)
    assert len(env.unpacked_hidden_boxes) == 4
    assert env.observation_space['height_map'].shape == (10, 10)
    assert env.observation_space['visible_box_sizes'].shape == (1, 3)
    assert env.action_space['position'].shape == (2,)
    assert_array_equal(obs['height_map'], np.zeros((10, 10)))


def test_sequence(basic_environment):
    env = basic_environment
    obs = env.reset(seed=42)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 3, 3]]))
    # action for num_visible_boxes = 1
    action = {'position': [0, 0]}
    obs, reward, truncated, terminated, info = env.step(action)
    assert obs['height_map'][0, 0] == 3
    assert len(env.unpacked_hidden_boxes) == 3
    assert len(env.packed_boxes) == 1
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
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 2, 3]]))
    # Set an action that is allowed
    action = {'position': [0, 0]}
    obs, reward, truncated, terminated, info = env.step(action)
    assert obs['height_map'][0, 0] == 6
    assert len(env.unpacked_hidden_boxes) == 2
    assert len(env.container.boxes) == 2
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box1 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[1].position, np.asarray([0, 0, 3]))

    # Check the size of the next incoming box (box2)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 4, 2]]))
    # Set an action that is allowed
    action = {'position': [3, 0], 'box_index': 0}
    obs, reward, truncated, terminated, info = env.step(action)
    # Check the height map after the action
    hm2 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm2[0:3, 0:2] = 6
    hm2[0:3, 2] = 3
    hm2[3:6, 0:4] = 2
    np.testing.assert_array_equal(obs['height_map'], hm2)
    assert len(env.unpacked_hidden_boxes) == 1
    assert len(env.container.boxes) == 3
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box2 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[2].position, np.asarray([3, 0, 0]))

    # Check the size of the next incoming box (box3)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 2, 4]]))
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
    assert len(env.unpacked_hidden_boxes) == 0
    assert len(env.container.boxes) == 4
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box3 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[3].position, np.asarray([0, 3, 0]))

    # Check the size of the next incoming box (box4)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 2, 3]]))
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
    assert len(env.unpacked_hidden_boxes) == 0
    assert len(env.container.boxes) == 5
    assert reward == 1
    assert truncated is False
    assert terminated is True


def test_reset():
    env = make('PackingEnv0', container_size=[10, 10, 10],
               box_sizes=boxes_generator([10, 10, 10], 64, 42),
               num_visible_boxes=1)
    obs1 = env.reset(seed=123)
    obs2 = env.reset(seed=123)
    check_env(env)

    assert obs1 in env.observation_space
    assert env.observation_space.contains(obs2)
    assert obs1['height_map'].shape == (10, 10)
    assert obs1['visible_box_sizes'].shape == (1, 3)


def test_action_mask_sampling():
    env = make('PackingEnv0', new_step_api=False, container_size=[10, 10, 10],
               box_sizes=boxes_generator([10, 10, 10], 64, 42),
               num_visible_boxes=1)
    obs = env.reset()

    for step_num in range(10):
        action_mask = obs['action_mask']
        action_mask_dict = {'position': action_mask}
        action = env.action_space.sample(action_mask_dict)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break




