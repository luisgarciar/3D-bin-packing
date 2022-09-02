import pytest
import numpy as np
from src.packing_env import PackingEnv0
from src.packing_engine import Box, Container
from src.utils import boxes_generator
import random as rd
from numpy.testing import assert_array_equal
from gym import make
from gym.utils.env_checker import check_env
import time

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
    assert_array_equal(obs['height_map'], np.zeros((10, 10)))


@pytest.mark.integtest
def test_sequence(basic_environment):
    env = basic_environment
    obs = env.reset(seed=42)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 3, 3]]))
    # action for num_visible_boxes = 1
    action = 0
    obs, reward, truncated, terminated, info = env.step(action)
    # Check that the height map is correct
    hm0 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm0[0:3, 0:3] = 3
    np.testing.assert_array_equal(obs['height_map'], hm0)
    assert len(env.unpacked_hidden_boxes) == 3
    assert len(env.packed_boxes) == 1
    assert reward == 1
    assert truncated is False
    assert terminated is False
    # Check that box0 is in the container in the right place
    np.testing.assert_array_equal(env.container.boxes[0].position, np.asarray([0, 0, 0]))
    hm = np.zeros((10, 10), dtype=np.int32)

    # Check the size of the next incoming box (box1)
    assert_array_equal(obs['visible_box_sizes'], np.array([[3, 2, 3]]))

    # Set an action that is allowed for box 1
    action = 0
    obs, reward, truncated, terminated, info = env.step(action)

    # Check that the height map after placing box1 is correct
    hm1 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm1[0:3, 0:2] = 6
    hm1[0:3, 2] = 3
    np.testing.assert_array_equal(obs['height_map'], hm1)
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
    action = 30
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
    action = 3
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
    action = 5
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


def test_two_box_action_mask():
    container_size1 = [3, 3, 3]
    box_sizes1 = [[3, 3, 2], [2, 2, 2]]
    env = make('PackingEnv0', new_step_api=False, container_size=container_size1,
               box_sizes=box_sizes1, num_visible_boxes=1, render_mode='human')
    obs = env.reset(seed=5)
    action = env.position_to_action([0, 0])
    obs, reward, done, info = env.step(action)
    np.testing.assert_array_equal(obs['action_mask'], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    fig = env.container.plot()
    fig.show()


def test_invalid_action():
    container_size1 = [3, 3, 3]
    box_sizes1 = [[3, 3, 2], [2, 2, 2]]
    env = make('PackingEnv0', new_step_api=False, container_size=container_size1,
               box_sizes=box_sizes1, num_visible_boxes=1, render_mode='human')
    obs = env.reset(seed=5)
    action_mask = obs['action_mask']
    np.testing.assert_array_equal(action_mask, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    action = env.position_to_action([0, 0])
    obs, reward, done, info = env.step(action)
    np.testing.assert_array_equal(obs['action_mask'], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    action = env.position_to_action([0, 0])
    # action should be invalid and the box should be skipped
    obs, reward, done, info = env.step(action)
    assert len(env.skipped_boxes) == 1
    fig = env.container.plot()
    fig.show()


# def test_randomized_agent():
#     box_sizes = [[2, 4, 4], [1, 5, 3], [1, 5, 2], [1, 5, 1], [1, 5, 3], [9, 2, 1], [1, 2, 1], [1, 4, 5], [1, 4, 5],
#                  [8, 2, 1], [2, 2, 1], [1, 8, 1], [1, 8, 1], [5, 7, 1], [1, 3, 2], [1, 4, 4], [1, 1, 5], [1, 2, 5],
#                  [1, 4, 3], [3, 1, 3], [2, 4, 2], [4, 2, 1], [4, 1, 2], [6, 1, 1], [4, 2, 1], [6, 2, 1], [2, 1, 1],
#                  [3, 1, 1], [4, 2, 1], [4, 2, 3], [6, 2, 3], [4, 4, 1], [4, 1, 1], [5, 2, 1], [3, 5, 2], [3, 1, 2],
#                  [3, 4, 2], [3, 3, 1], [3, 3, 1], [1, 4, 2], [1, 4, 2], [3, 1, 1], [1, 1, 1], [5, 1, 3], [5, 1, 3],
#                  [3, 1, 2], [1, 1, 1], [5, 1, 1], [3, 3, 3], [5, 2, 1], [5, 2, 1], [1, 8, 2], [1, 8, 2], [1, 1, 2],
#                  [1, 1, 3], [3, 1, 2], [3, 3, 2], [2, 3, 1], [3, 3, 1], [2, 5, 1], [3, 1, 3], [3, 2, 3], [2, 2, 2],
#                  [2, 2, 2], [3, 1, 1], [3, 1, 1], [1, 5, 1], [3, 1, 1], [3, 4, 1], [3, 2, 1], [1, 2, 1], [4, 2, 1],
#                  [4, 3, 1], [1, 5, 1], [4, 5, 1], [5, 1, 2], [5, 1, 2], [4, 1, 1], [4, 1, 1], [3, 2, 2], [3, 3, 2],
#                  [6, 2, 1], [2, 1, 1], [2, 1, 1], [3, 2, 1], [3, 2, 1], [3, 1, 2], [1, 2, 1], [1, 2, 4], [3, 1, 2],
#                  [3, 1, 2], [1, 1, 2], [1, 1, 1], [4, 1, 3], [4, 2, 3], [3, 1, 1], [3, 1, 2], [1, 4, 2], [1, 2, 2],
#                  [1, 2, 2]]
#
#     box_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 32,
#               35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 53, 54, 55, 57, 60, 61, 62, 63, 64, 65, 66, 67, 70, 73, 77, 78,
#               82, 83, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 99]
#
#     boxes_to_pack = [box_sizes[id] for id in box_ids]
#
#     position_boxes = [[5, 7, 0], [10, 3, 0], [1, 1, 0], [9, 6, 0], [2, 2, 0], [2, 0, 0], [0, 8, 0], [0, 4, 0],
#                       [6, 3, 0], [2, 0, 1], [7, 8, 0], [3, 2, 0], [4, 2, 0], [4, 6, 1], [5, 2, 0], [1, 9, 0],
#                       [8, 8, 1], [9, 6, 1], [2, 10, 0], [7, 3, 0], [6, 0, 2], [6, 1, 3], [2, 0, 2], [7, 4, 2],
#                       [6, 1, 5], [2, 0, 3], [2, 0, 4], [7, 2, 0], [6, 1, 6], [3, 5, 1], [1, 1, 2], [3, 1, 7],
#                       [6, 7, 4], [3, 1, 8],  [3, 1, 11], [2, 10, 3], [6, 9, 4], [3, 1, 14], [5, 3, 4], [7, 9, 1],
#                       [3, 0, 7], [3, 6, 3],  [8, 2, 1], [3, 7, 4], [3, 7, 7], [3, 7, 9], [3, 1, 15], [3, 1, 16],
#                       [6, 3, 5], [3, 1, 17], [2, 4, 3], [6, 3, 6], [6, 0, 3], [6, 0, 4], [4, 1, 18], [7, 6, 2],
#                       [2, 10, 5], [3, 2, 1], [3, 7, 11], [8, 2, 4], [7, 0, 5], [8, 7, 0], [8, 1, 8], [8, 2, 6],
#                       [3, 0, 9], [1, 1, 4], [0, 6, 5], [10, 8, 0]]
#
#     env = make('PackingEnv0', new_step_api=False, container_size=[11, 11, 11],
#                box_sizes=boxes_to_pack, num_visible_boxes=1, render_mode='human')
#     obs = env.reset(seed=5)
#
#     for step_num in range(len(box_ids)):
#         action_mask = obs['action_mask']
#         action = env.position_to_action(position_boxes[step_num][0:2])
#         if step_num == 34:
#             pass
#         obs, reward, done, info = env.step(action)
#         fig = env.container.plot()
#         fig.show()
#         time.sleep(0.2)
#
#     height_boxes = [box.position[2] + box.size[2] for box in env.packed_boxes]
#     assert np.amax(height_boxes) <= 11
