"""
Packing Gym: An OpenAI Gym environment for 3D packing problems.
We follow the space representation depicted below, all coordinates and lengths of boxes and containers are integers.

    x: depth
    y: length
    z: height

       Z
       |
       |
       |________Y
      /
     /
    X

    Classes:
        Box
        Container

"""
import gym
from gym.spaces import Discrete, MultiDiscrete, MultiBinary
import numpy as np
from typing import List, Type, Tuple
from nptyping import NDArray, Int, Shape
from src.packing_engine import Box, Container
from gym.utils import seeding


class PackingEnv0(gym.Env):
    """ A class to represent the packing environment.

    Description:
        The environment consists of a 3D container and an initial list of 3D boxes, the goal
        is to pack the boxes into the container minimizing the empty space. For simplicity
        we assume that the container is loaded from the top.

        The state of the container is represented by a 2D array storing the height map (top view)
        of the container (see the documentation of packing_engine.Container.height_map
        for a detailed explanation) and a list of sizes of the upcoming boxes.

        Observation:
        Type:  Dict(2)

        Key             Description                       Shape - Type:int                       (Min,Max) - Type:int
        height_map      Top view of the container         (container.size[0],container.size[1])  (0,container.size[2])
                        with heights of boxes already
                        placed

        box_sizes       Array with sizes of the upcoming   (num_upcoming_boxes, 3)               (1, container.size[2])
                        boxes

        Action:
        Type:           Dict(2)
        Key             Description                         Shape - Type:int          (Min,Max) - Type:int
        box_index       index for the box to be placed      (1,)                      (0, num_upcoming_boxes)
                        in the container

        position         (x,y) coordinates to place         (2,)                      (0, container.size[0]),
                        the box                                                       (0, container.size[1])

        Reward:
        To be defined

        Starting State:
        height_map is initialized as a zero array and the list of upcoming boxes is initialized as a random list of
        length num_upcoming_boxes from the complete list of boxes.

        Episode Termination:
        The episode is terminated when all the boxes are placed in the container or when the container is full.

        Episode Reward:
        To be defined
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, container_size: List[int], box_sizes: List[List[int]], num_visible_boxes: int = 1) -> None:
        """ Initialize the environment.

         Parameters
        ----------:
            container_size: container_size
            box_size: sizes of boxes to be placed in the container
            num_visible_boxes: number of boxes visible to the agent
        """
        # TO DO: Add parameter check box area
        assert num_visible_boxes <= len(box_sizes)
        self.container = Container(container_size)
        # The initial list of all boxes that should be placed in the container.
        self.initial_boxes = [Box(box_size, position=[-1, -1, -1], id_=index) for index, box_size in enumerate(box_sizes)]

        # The list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = self.initial_boxes.copy()
        # The list of boxes that are not yet packed and are visible to the agent
        self.unpacked_visible_boxes = []
        # The list of boxes that are already packed
        self.packed_boxes = []
        # The list of boxes that could not be packed (didn't fit in the container)
        self.skipped_boxes = []

        # The number and list of boxes that are visible to the agent.
        self.num_visible_boxes = num_visible_boxes
        self.unpacked_visible_boxes = []
        self.state = {}
        self.done = False
        # self.reward = 0 -- not needed

        # Array to define the MultiDiscrete space with the list of sizes of the visible boxes
        # Note: The upper bound for the entries in MultiDiscrete space is not inclusive -- so we add 1
        box_repr = np.zeros(shape=(num_visible_boxes, 3), dtype=np.int32)
        box_repr[:] = self.container.size + [1, 1, 1]
        # Array to define the MultiDiscrete space with the height map of the container
        height_map_repr = np.ones(shape=(container_size[0], container_size[1]), dtype=np.int32)*(container_size[2] + 1)

        # Array to define the MultiDiscrete space with the action mask
        action_mask_repr = np.ones(shape=(container_size[0], container_size[1]), dtype=np.int8)*2

        # Dict to define the observation space
        observation_dict = {'height_map': MultiDiscrete(height_map_repr),
                            'visible_box_sizes': MultiDiscrete(box_repr),
                            'action_mask': MultiDiscrete(action_mask_repr)}

        # Observation space
        self.observation_space = gym.spaces.Dict(observation_dict)

        # Dict to define the action space for num_visible_boxes > 1
        if num_visible_boxes > 1:
            action_dict = {'box_index': Discrete(num_visible_boxes),
                           'position': MultiDiscrete([container_size[0], container_size[1]])}
        else:
            action_dict = {'position': MultiDiscrete([container_size[0], container_size[1]])}

        # Action space
        self.action_space = gym.spaces.Dict(action_dict)

    def seed(self, seed: int = 42):
        """Seed the random number generator for the environment.
         Parameters
         -----------
             seed: int
                 Seed for the environment.
             """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options={}, return_info=False) -> dict[str, object]:
        """ Reset the environment.
        Returns:
        ----------
            observation: Dictionary with the observation of the environment.
        """
        # Check: add info, return info
        self.container.reset()
        # Reset the list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = self.initial_boxes.copy()

        # Reset the list of boxes visible to the agent and deletes them from the list of
        # hidden unpacked boxes to be packed
        if self.num_visible_boxes > 1:
            self.unpacked_visible_boxes = self.unpacked_hidden_boxes[0:self.num_visible_boxes]
            del self.unpacked_hidden_boxes[0:self.num_visible_boxes]
        else:
            self.unpacked_visible_boxes = [self.unpacked_hidden_boxes.pop(0)]

        # Reset the list of boxes that are already packed
        self.packed_boxes = self.container.boxes

        # Set the list of visible box sizes in the observation space
        visible_box_sizes = np.asarray([box.size for box in self.unpacked_visible_boxes])
        visible_box_sizes = np.reshape(visible_box_sizes, (self.num_visible_boxes, 3))
        # Reset the state of the environment
        hm = np.asarray(self.container.height_map, dtype=np.int32)

        self.state = {'height_map': hm, 'visible_box_sizes': visible_box_sizes,
                      'action_mask': self.container.action_mask(box=self.unpacked_visible_boxes[0])}
        self.done = False
        self.seed(seed)

        if return_info is False:
            return self.state
        else:
            return self.state, {}

    def step(self, action: dict) -> Tuple[NDArray, float, bool, bool, dict]:
        """ Step the environment.
        Parameters:
        -----------
            action: Dictionary with the action to be taken.
        Returns:
        ----------
            observation: Dictionary with the observation of the environment.
            reward: Reward for the action.
            truncated: Whether the episode is truncated.
            terminated: Whether the episode is terminated.
            info: Dictionary with additional information.
        """
        # Get the index of the box to be placed in the container
        if self.num_visible_boxes > 1:
            box_index = action['box_index']
        else:
            box_index = 0
        # Get the position of the box to be placed in the container
        position = action['position']
        # Check if the action is valid
        # TO DO: add parameter check area, add info, return info
        if self.container.check_valid_box_placement(self.unpacked_visible_boxes[box_index], position, check_area=100) == 1:
            # Place the box in the container and delete it from the list of unpacked boxes that are visible to the agent
            if self.num_visible_boxes > 1:
                self.container.place_box(self.unpacked_visible_boxes.pop(box_index), position)
            else:
                self.container.place_box(self.unpacked_visible_boxes[0], position)
                self.unpacked_visible_boxes = []
            # Update the list of packed boxes
            self.state['height_map'] = self.container.height_map
            # Update the list of packed boxes
            self.packed_boxes = self.container.boxes
            # set reward
            reward = 1
            # self.reward = self.container.compute_reward()

        # If the action is not valid, remove the box and add it to skipped boxes
        else:
            self.skipped_boxes.append(self.unpacked_visible_boxes.pop(box_index))
            reward = 0

        # Update the list of visible boxes if possible
        if len(self.unpacked_hidden_boxes) > 0:
            self.unpacked_visible_boxes.append(self.unpacked_hidden_boxes.pop(0))

        # If there are no more boxes to be packed, finish the episode
        if len(self.unpacked_visible_boxes) == 0:
            self.done = True
            terminated = self.done
            truncated = False
            self.state['visible_box_sizes'] = []
            # TO DO: add info, return info
        else:
            visible_box_sizes = np.asarray([box.size for box in self.unpacked_visible_boxes])
            visible_box_sizes = np.reshape(visible_box_sizes, (self.num_visible_boxes, 3))
            # Update the state of the environment
            self.state['visible_box_sizes'] = visible_box_sizes
            # Update the action mask
            self.state['action_mask'] = self.container.action_mask(box=self.unpacked_visible_boxes[0])
            terminated = False
            truncated = False

        return self.state, reward, truncated, terminated, {}

    def compute_reward(self) -> float:
        """ Compute the reward for the action.
        Returns:
        ----------
            reward: Reward for the action.
        """
        pass

    def render(self, mode='human') -> None:
        """ Render the environment.
        Args:
            mode: Mode to render the environment.
        """
        if mode == 'human':
            self.container.plot()

    def close(self) -> None:
        """ Close the environment.
        """
        pass
