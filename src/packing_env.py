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

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, container_size: List[int], box_sizes: List[List[int]], num_incoming_boxes: int = 1,
                 seed: int = 42, gen_action_mask: bool = True) -> None:
        """ Initialize the environment.

         Parameters
        ----------:
            container_size: container_size
            box_size: sizes of boxes to be placed in the container
        """
        # TO DO: Add parameter check box area
        assert num_incoming_boxes <= len(box_sizes)
        self.container = Container(container_size)
        # The list of all boxes that should be placed in the container.
        self.unpacked_boxes = [Box(box_size, position=[-1, -1, -1], id_=index) for index, box_size in enumerate(box_sizes)]
        # The number and list of boxes that are visible to the agent.
        self.num_incoming_boxes = num_incoming_boxes
        self.incoming_boxes = []
        self.state = {}
        self.done = False
        self._seed(seed)
        #self.reward = 0 -- not needed

        # Array to define the MultiDiscrete space with the list of sizes of the incoming boxes
        box_repr = np.zeros(shape=(num_incoming_boxes, 3), dtype=np.int32)
        box_repr[:] = self.container.size
        # Array to define the MultiDiscrete space with the height map of the container
        height_map_repr = np.ones(shape=(container_size[0], container_size[1]), dtype=np.int32)*container_size[2]

        # Dict to define the observation space
        observation_dict = {'height_map': MultiDiscrete(height_map_repr),
                            'incoming_box_sizes': MultiDiscrete(box_repr)}

        # if gen_action_mask is True:
        # observation_dict['action_mask'] = MultiBinary([self.container.size[0], self.container.size[1]])
        self.observation_space = gym.spaces.Dict(observation_dict)

        # Dict to define the action space
        action_dict = {'box_index': Discrete(num_incoming_boxes),
                       'position': MultiDiscrete([self.container.size[0], self.container.size[1]])}
        self.action_space = gym.spaces.Dict(action_dict)

    def _seed(self, seed: int = 42) -> None:
        """Seed the environment.
        Parameters
        -----------
            seed: int
                Seed for the environment.
            """
        self._np_random = gym.utils.seeding.np_random(seed)

    def reset(self, seed) -> dict[str, object]:
        """ Reset the environment.
        Returns:
        ----------
            observation: Dictionary with the observation of the environment.
        """
        # Check: add info, return info
        self._seed(seed)
        self.container.reset()
        # Reset the list of incoming boxes visible to the agent and deletes them from the list of boxes to be packed
        self.incoming_boxes = self.unpacked_boxes[0:self.num_incoming_boxes]
        self.unpacked_boxes[:self.num_incoming_boxes] = []
        # Set the list of incoming box sizes in the observation space
        incoming_box_sizes = np.asarray([box.size for box in self.incoming_boxes])
        # Reset the state of the environment
        self.state = {'height_map': self.container.height_map, 'incoming_box_sizes': incoming_box_sizes}
        self.done = False
        return self.state

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
        box_index = action['box_index']
        # Get the position of the box to be placed in the container
        position = action['position']
        # Check if the action is valid
        # TO DO: add parameter check area, add info, return info
        if self.container.check_valid_box_placement(self.incoming_boxes[box_index], position, check_area=100) == 1:
            # Place the box in the container
            self.container.place_box(self.incoming_boxes[box_index], position)
            self.state['height_map'] = self.container.height_map
            # Remove the box from the list of incoming boxes
            del self.incoming_boxes[box_index]
            # Add a new box to the list of incoming boxes if possible
            if len(self.unpacked_boxes) > 0:
                self.incoming_boxes.append(self.unpacked_boxes.pop())
                incoming_box_sizes = np.asarray([box.size for box in self.incoming_boxes])
            # Update the state of the environment
            self.state['incoming_box_sizes'] = incoming_box_sizes
            # Check if the episode is terminated
            self.done = (len(self.incoming_boxes) == 0)
            terminated = self.done
            truncated = False
            # Give a reward for the action
            # self.reward = self.container.compute_reward()
            reward = 1
            # Return the observation, reward, done and info
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

    

