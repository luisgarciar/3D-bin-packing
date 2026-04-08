# 3D Bin Packing Optimization

Repository for the Capstone Project 3D Packing Optimization of the [Fourthbrain Machine Learning Engineer program](https://www.fourthbrain.ai/machine-learning-engineer).

This repository contains an environment compatible with [Gymnasium's API](https://gymnasium.farama.org/) to solve the 
3D bin packing problem with reinforcement learning (RL).


![Alt text](gifs/random_rollout2.gif?raw=true "A random packing agent in the environment")

## Problem definition and assumptions:
The environment consists of a list of 3D boxes of varying sizes and a single container of fixed size. The goal is to pack
as many boxes as possible in the container minimizing the empty volume. We assume that rotation of the boxes is 
not possible.

##  Problem instances: 
The function `boxes_generator` in the file `utils.py` generates instances of the 3D Bin Packing problem using the 
algorithm described in [Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization](https://arxiv.org/pdf/1807.01672.pdf)
(Algorithm 2, Appendix).

## Documentation
The documentation for this project is located in the `doc` folder, with a complete description of the state and 
action space as well as the rewards to be used for RL training.

## Installation instructions
We recommend Python 3.12. Install [uv](https://docs.astral.sh/uv/) and then run:

```bash
git clone https://github.com/luisgarciar/3D-bin-packing.git
cd 3D-bin-packing
uv sync --extra dev
```

This creates a local `.venv` and installs all runtime and development dependencies from `pyproject.toml`.

## Running code and tests
Run Python commands through `uv` so they execute in the managed environment:

```bash
uv run pytest
uv run python src/train.py
```

## Packing engine
The module `packing_kernel` (located in `src/packing_kernel.py`) implements the `Container` and `Box` objects that are 
used in the Gymnasium environment. To add custom features (for example, to allow rotations), see the documentation of this module.

## Environment
The Gymnasium environment is implemented in the module `src/packing_env.py`.

## Demo notebooks
A demo notebook `demo_ffd` implementing the heuristic-based method 'First Fit Decreasing' is available in the `nb` 
folder.

## Unit tests
The folder `tests` contains unit tests to be run with pytest (`uv run pytest`).

## CI/CD
GitHub Actions pipelines are available in `.github/workflows`:
- `ci.yml`: runs tests on pushes to `main` and pull requests (Python 3.12).
- `release.yml`: builds distribution artifacts on version tags (`v*`) and publishes a GitHub release with the built files.

## Update: 22/08/2022
The following updates have been made to the repository:
- Added the `packing_env.py` file with the Gymnasium environment.
- Added unit tests for the Gymnasium environment.
- Updated the documentation with the full description of the state and action space.
- Updated the demo notebooks.

## Update: 13/09/2022
The following updates have been made to the repository:
- Added functionality for saving rollouts of a policy in a .gif file and
- Added a demo notebook for the random policy.
- Updated the requirements.txt file with the required packages.
- Added a demo script for training agents with Maskable PPO. 

## Update: 7/1/2023 
The following updates have been made to the repository:
- Updated the demo notebook for training agents with Maskable PPO in Google colab.
- Fixed issues with the tests.
