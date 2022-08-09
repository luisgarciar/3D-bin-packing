# 3D Bin Packing Optimization

Repository for the Capstone Project 3D Packing Optimization of the [Fourthbrain Machine Learning Engineer program](https://www.fourthbrain.ai/machine-learning-engineer).

Group Members: Luis García Ramos, Przemyslaw Sekula

This repository contains an environment compatible with [OpenAI Gym's API](https://github.com/openai/gym) to solve the 3D bin packing problem with reinforcement learning (RL).

## Problem definition and assumptions:
The environment consists of a list of 3D boxes of varying sizes and a single container of fixed size. The goal is to pack as many boxes as possible in the container minimizing the empty volume. We assume that rotation of the boxes is not possible. 

##  Problem instances: 
The function `boxes_generator` in the file `utils.py` generates instances of the 3D Bin Packing problem using the algorithm described in [Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization](https://arxiv.org/pdf/1807.01672.pdf) (Algorithm 2, Appendix).

## Documentation
The documentation for this project is located in the `doc` folder, with a complete description of the state and action space as well as the rewards to be used for RL.

## Packing engine
The module `packing_engine` (located in `src/packing_engine.py`) implements the `Container` and `Box` objects that are used in the Gym environment. To add custom features (for example, to allow rotations), see the documentation of this module.

## Demo notebooks
A demo notebook implementing the heuristic-based method 'First Fit Decreasing' is available in the `nb` folder.

## Unit tests
The folder `tests` contains unit tests to be run with pytest.







