import pytest
import numpy as np
from src import utils
import random as rd

# Data for random tests
rd.seed(42)
num_rd_tests = 20
num_items = rd.sample(range(50, 81), num_rd_tests)
height = rd.sample(range(10, 10+num_rd_tests+1), num_rd_tests)
length = rd.sample(range(20, 20+num_rd_tests+1), num_rd_tests)
width = rd.sample(range(30, 30+num_rd_tests+1), num_rd_tests)

edge_lengths = [[height[j], length[j], width[j]] for j in range(num_rd_tests)]
testdata = [(num_items[j], edge_lengths[j], int(np.prod(edge_lengths[j]))) for j in range(num_rd_tests)]


def test_box_generator_default():
    items = utils.boxes_generator([10, 10, 10])
    box_sizes = [np.prod(box) for box in items]
    assert int(np.sum(box_sizes)) == 1000


@pytest.mark.parametrize("num_items,bin_size,expected", testdata)
def test_box_generator_random_data(num_items, bin_size, expected):
    items = utils.boxes_generator(bin_size, num_items)
    box_sizes = [np.prod(box) for box in items]
    assert int(np.sum(box_sizes)) == expected



