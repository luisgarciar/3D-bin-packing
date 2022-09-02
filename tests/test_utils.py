import pytest
import numpy as np
from src import utils
import random as rd

# Data for random tests
rd.seed(42)
num_rd_tests = 5
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(10, 10 + num_rd_tests + 1), num_rd_tests)
length = rd.sample(range(20, 20 + num_rd_tests + 1), num_rd_tests)
width = rd.sample(range(30, 30 + num_rd_tests + 1), num_rd_tests)

edge_lengths = [[height[j], length[j], width[j]] for j in range(num_rd_tests)]
testdata0 = [(num_items[j], edge_lengths[j], int(np.prod(edge_lengths[j]))) for j in range(num_rd_tests)]


def test_box_generator_default():
    items = utils.boxes_generator([10, 10, 10])
    box_sizes = [np.prod(box) for box in items]
    assert int(np.sum(box_sizes)) == 1000


@pytest.mark.parametrize("num_items,bin_size,expected", testdata0)
def test_box_generator_random_data(num_items, bin_size, expected):
    items = utils.boxes_generator(bin_size, num_items)
    box_sizes = [np.prod(box) for box in items]
    assert int(np.sum(box_sizes)) == expected


interval_a = [[0, 2], [3, 4], [5, 6], [1, 5]]
interval_b = [[1, 2], [2, 3], [5, 6], [0, 1]]
intersect = [True, False, True, False]
testdata1 = zip(interval_a, interval_b, intersect)


@pytest.mark.parametrize("a,b,expected", testdata1)
def test_interval_intersection(a, b, expected):
    assert utils.interval_intersection(a, b) \
           == utils.interval_intersection(b, a) \
           == expected


cuboid_a = [[0, 1, 3, 4, 5, 6], [0, 2, 1, 3, 5, 4], [0, 0, 0, 1, 1, 1]]
cuboid_b = [[1, 2, 3, 4, 5, 6], [0, 1, 2, 5, 3, 4], [2, 2, 2, 3, 3, 3]]
inter = [True, True, False]
testdata2 = zip(cuboid_a, cuboid_b, inter)


@pytest.mark.parametrize("cuboid_a,cuboid_b,expected", testdata2)
def test_cuboid_intersection(cuboid_a, cuboid_b, expected):
    assert utils.cuboids_intersection(cuboid_a, cuboid_b) == \
           utils.cuboids_intersection(cuboid_b, cuboid_a) ==\
           expected


def test_box_generator_custom():
    items = utils.boxes_generator([10, 10, 10], num_items=10, seed=5)
    box_sizes = [np.prod(box) for box in items]
    assert int(np.sum(box_sizes)) == 1000


def test_cuboid_fits_in_bin():
    cuboid = [0, 0, 0, 1, 1, 1]
    bin = [0, 0, 0, 10, 10, 10]
    assert utils.cuboid_fits_in_bin(cuboid, bin)


def test_cuboid_fits_in_bin_false():
    cuboid = [0, 0, 0, 2, 2, 2]
    bin = [0, 0, 0, 1, 1, 1]
    assert not utils.cuboid_fits_in_bin(cuboid, bin)


def test_cuboid_fits_in_bin_false2():
    cuboid = [0, 0, 0, 1, 1, 1]
    bin = [0, 0, 0, 2, 2, 2]
    assert not utils.cuboid_fits_in_bin(cuboid, bin)