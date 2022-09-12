import random as rd
import numpy as np
import pytest
from gym import make
from gym.utils.env_checker import check_env
from numpy.testing import assert_array_equal
from src.packing_env import PackingEnv
from src.utils import boxes_generator

num_rd_tests = 3
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
    env = PackingEnv(container_size, box_sizes, num_visible_boxes)
    assert len(env.unpacked_hidden_boxes) == 5
    assert env.observation_space.spaces["height_map"].shape == (
        container_size[0],
        container_size[1],
    )
    assert env.observation_space.spaces["visible_box_sizes"].shape == (
        num_visible_boxes,
        3,
    )


@pytest.fixture
def basic_environment():
    container_size = [10, 10, 10]
    box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    num_visible_boxes = 1
    env = PackingEnv(container_size, box_sizes, num_visible_boxes)
    return env


def test_env_reset(basic_environment):
    env = basic_environment
    obs = env.reset()
    assert len(env.unpacked_hidden_boxes) == 4
    assert env.observation_space["height_map"].shape == (10, 10)
    assert env.observation_space["visible_box_sizes"].shape == (1, 3)
    assert_array_equal(obs["height_map"], np.zeros((10, 10)))


@pytest.mark.integtest
def test_sequence(basic_environment):
    env = basic_environment
    obs = env.reset()
    assert_array_equal(obs["visible_box_sizes"], np.array([[3, 3, 3]]))
    # action for num_visible_boxes = 1
    action = 0
    obs, reward, terminated, info = env.step(action)
    # Check that the height map is correct
    hm0 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm0[0:3, 0:3] = 3
    np.testing.assert_array_equal(obs["height_map"], hm0)
    assert len(env.unpacked_hidden_boxes) == 3
    assert len(env.packed_boxes) == 1
    assert reward == 1
    assert terminated is False
    # Check that box0 is in the container in the right place
    np.testing.assert_array_equal(
        env.container.boxes[0].position, np.asarray([0, 0, 0])
    )
    hm = np.zeros((10, 10), dtype=np.int32)

    # Check the size of the next incoming box (box1)
    assert_array_equal(obs["visible_box_sizes"], np.array([[3, 2, 3]]))

    # Set an action that is allowed for box 1
    action = 0
    obs, reward, terminated, info = env.step(action)

    # Check that the height map after placing box1 is correct
    hm1 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm1[0:3, 0:2] = 6
    hm1[0:3, 2] = 3
    np.testing.assert_array_equal(obs["height_map"], hm1)
    assert len(env.unpacked_hidden_boxes) == 2
    assert len(env.container.boxes) == 2
    assert reward == 1
    assert terminated is False
    # Check that box1 is in the container in the right place
    np.testing.assert_array_equal(
        env.container.boxes[1].position, np.asarray([0, 0, 3])
    )

    # Check the size of the next incoming box (box2)
    assert_array_equal(obs["visible_box_sizes"], np.array([[3, 4, 2]]))
    # Set an action that is allowed
    action = 30
    obs, reward, terminated, info = env.step(action)
    # Check the height map after the action
    hm2 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm2[0:3, 0:2] = 6
    hm2[0:3, 2] = 3
    hm2[3:6, 0:4] = 2

    np.testing.assert_array_equal(obs["height_map"], hm2)
    assert len(env.unpacked_hidden_boxes) == 1
    assert len(env.container.boxes) == 3
    assert reward == 1
    assert terminated is False
    # Check that box2 is in the container in the right place
    np.testing.assert_array_equal(
        env.container.boxes[2].position, np.asarray([3, 0, 0])
    )

    # Check the size of the next incoming box (box3)
    assert_array_equal(obs["visible_box_sizes"], np.array([[3, 2, 4]]))
    # Set an action that is allowed
    action = 3
    obs, reward, terminated, info = env.step(action)
    # Check the height map after the action
    hm3 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm3[0:3, 0:2] = 6
    hm3[0:3, 2] = 3
    hm3[3:6, 0:4] = 2
    hm3[0:3, 3:5] = 4
    np.testing.assert_array_equal(obs["height_map"], hm3)
    assert len(env.unpacked_hidden_boxes) == 0
    assert len(env.container.boxes) == 4
    assert reward == 1
    assert terminated is False
    # Check that box3 is in the container in the right place
    np.testing.assert_array_equal(
        env.container.boxes[3].position, np.asarray([0, 3, 0])
    )

    # Check the size of the next incoming box (box4)
    assert_array_equal(obs["visible_box_sizes"], np.array([[3, 2, 3]]))
    # Set an action that is allowed
    action = 5
    obs, reward, terminated, info = env.step(action)
    # Check the height map after the action
    hm4 = np.zeros(shape=[10, 10], dtype=np.int32)
    hm4[0:3, 0:2] = 6
    hm4[0:3, 2] = 3
    hm4[3:6, 0:4] = 2
    hm4[0:3, 3:5] = 4
    hm4[0:3, 5:7] = 3
    np.testing.assert_array_equal(obs["height_map"], hm4)
    assert len(env.unpacked_hidden_boxes) == 0
    assert len(env.container.boxes) == 5
    assert reward == 1
    assert terminated is True


def test_reset():
    env = make(
        "PackingEnv-v0",
        container_size=[10, 10, 10],
        box_sizes=boxes_generator([10, 10, 10], 64, 42),
        num_visible_boxes=1,
    )
    obs1 = env.reset()
    obs2 = env.reset()
    check_env(env)

    assert obs1 in env.observation_space

    assert env.observation_space.contains(obs2)
    assert obs1["height_map"].shape == (10, 10)
    assert obs1["visible_box_sizes"].shape == (1, 3)


def test_two_box_action_mask():
    container_size1 = [3, 3, 3]
    box_sizes1 = [[3, 3, 2], [2, 2, 2]]
    env = make(
        "PackingEnv-v0",
        container_size=container_size1,
        box_sizes=box_sizes1,
        num_visible_boxes=1,
        render_mode="human",
    )
    env.reset(seed=5)
    action = env.position_to_action([0, 0])
    obs, reward, done, info = env.step(action)
    np.testing.assert_array_equal(
        obs["action_mask"], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_invalid_action():
    container_size1 = [3, 3, 3]
    box_sizes1 = [[3, 3, 2], [2, 2, 2]]
    env = make(
        "PackingEnv-v0",
        container_size=container_size1,
        box_sizes=box_sizes1,
        num_visible_boxes=1,
        render_mode="human",
    )
    obs = env.reset(seed=5)
    action_mask = obs["action_mask"]
    # Only obe position is valid to place the first box
    np.testing.assert_array_equal(action_mask, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    action = env.position_to_action([0, 0])
    obs, reward, done, info = env.step(action)
    np.testing.assert_array_equal(
        obs["action_mask"], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    action = env.position_to_action([0, 0])
    # action should be invalid and the box should be skipped
    obs, reward, done, info = env.step(action)
    assert len(env.skipped_boxes) == 1
