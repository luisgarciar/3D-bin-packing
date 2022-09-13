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
import copy
import warnings
from typing import List, Tuple, Union

import gym
import numpy as np
import plotly.graph_objects as go
from gym.spaces import Discrete, MultiDiscrete, MultiBinary
from gym.utils import seeding
from nptyping import NDArray

from src.packing_kernel import Box, Container
from src.utils import boxes_generator


class PackingEnv(gym.Env):
    """A class to represent the packing environment.

    Description:
        The environment consists of a 3D container and an initial list of 3D boxes, the goal
        is to pack the boxes into the container minimizing the empty space. We assume
        that the container is loaded from the top.

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
        Type:  Discrete(container.size[0]*container.size[1]*num_visible_boxes)
        The agent chooses an integer j in the range [0, container.size[0]*container.size[1]*num_visible_boxes)),
        and the action is interpreted as follows: the box with index  j // (container.size[0]*container.size[1])
        is placed in the position (x,y) = (j//container.size[1], j%container.size[1]) in the container.

        Reward:
        To be defined

        Starting State:
        height_map is initialized as a zero array and the list of upcoming boxes is initialized as a random list of
        length num_visible_boxes from the complete list of boxes.

        Episode Termination:
        The episode is terminated when all the boxes are placed in the container or when the container is full.

        Episode Reward:
        To be defined
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        container_size: List[int],
        box_sizes: List[List[int]],
        num_visible_boxes: int = 1,
        render_mode: str = "human",
        options: dict = None,
    ) -> None:
        """Initialize the environment.

         Parameters
        ----------:
            container_size: size of the container in the form [lx,ly,lz]
            box_sizes: sizes of boxes to be placed in the container in the form [[lx,ly,lz],...]
            num_visible_boxes: number of boxes visible to the agent
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # TO DO: Add parameter check box area
        assert num_visible_boxes <= len(box_sizes)
        self.container = Container(container_size)
        # The initial list of all boxes that should be placed in the container.
        self.initial_boxes = [
            Box(box_size, position=[-1, -1, -1], id_=index)
            for index, box_size in enumerate(box_sizes)
        ]

        # The list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = self.initial_boxes.copy()
        # The list of boxes that are not yet packed and are visible to the agent
        self.unpacked_visible_boxes = []
        # The list of boxes that are already packed
        self.packed_boxes = []
        # The list of boxes that could not be packed (did not fit in the container)
        self.skipped_boxes = []

        # The number and list of boxes that are visible to the agent.
        self.num_visible_boxes = num_visible_boxes
        self.unpacked_visible_boxes = []
        self.state = {}
        self.done = False

        # Array to define the MultiDiscrete space with the list of sizes of the visible boxes
        # The upper bound for the entries in MultiDiscrete space is not inclusive -- we add 1 to each coordinate
        box_repr = np.zeros(shape=(num_visible_boxes, 3), dtype=np.int32)
        box_repr[:] = self.container.size + [1, 1, 1]
        # Array to define the MultiDiscrete space with the height map of the container
        height_map_repr = np.ones(
            shape=(container_size[0], container_size[1]), dtype=np.int32
        ) * (container_size[2] + 1)

        # The action mask is a Multibinary array with the same length as the number of positions
        # in the container times the number of visible boxes
        # Dict to define the observation space
        observation_dict = {
            "height_map": MultiDiscrete(height_map_repr),
            "visible_box_sizes": MultiDiscrete(box_repr),
            "action_mask": MultiBinary(
                container_size[0] * container_size[1] * num_visible_boxes
            ),
        }

        # Observation space
        self.observation_space = gym.spaces.Dict(observation_dict)
        # Action space
        self.action_space = Discrete(
            container_size[0] * container_size[1] * num_visible_boxes
        )

        # if num_visible_boxes > 1:
        #     # Dict to define the action space for num_visible_boxes > 1
        #     action_dict = {
        #         "box_index": Discrete(num_visible_boxes),
        #         "position": MultiDiscrete([container_size[0], container_size[1]]),
        #     }
        #     self.action_space = gym.spaces.Dict(action_dict)
        #
        # else:

    def seed(self, seed: int = 42):
        """Seed the random number generator for the environment.
        Parameters
        -----------
            seed: int
                Seed for the environment.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def action_to_position(self, action: int) -> Tuple[int, NDArray]:
        """Converts an index to a tuple with a box index
        and a position in the container.
        Parameters
        ----------
            action: int
                Index to be converted.
        Returns
        -------
            box_index: int
                Index of the box to be placed.
            position: ndarray
                Position in the container.
        """
        box_index = action // (self.container.size[0] * self.container.size[1])
        res = action % (self.container.size[0] * self.container.size[1])

        position = np.array(
            [res // self.container.size[0], res % self.container.size[0]]
        )

        return box_index, position.astype(np.int32)

    def position_to_action(self, position, box_index=0):
        """Converts a position in the container to an action index
        Returns
        -------
            action: int
                Index in the container.
        """
        action = (
            box_index * self.container.size[0] * self.container.size[1]
            + position[0] * self.container.size[0]
            + position[1]
        )
        return action

    def reset(self, seed=None, options=None, return_info=False) -> dict[str, object]:
        """Reset the environment.
        Parameters
        ----------
            seed: int
                Seed for the environment.
            options: dict
                Options for the environment.
        Returns
        ----------
            dict: Dictionary with the observation of the environment.
        """
        if return_info:
            info = {}

        self.container.reset()
        # Reset the list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = copy.deepcopy(self.initial_boxes)

        # Reset the list of boxes visible to the agent and deletes them from the list of
        # hidden unpacked boxes to be packed
        if self.num_visible_boxes > 1:
            self.unpacked_visible_boxes = self.unpacked_hidden_boxes[
                0 : self.num_visible_boxes
            ]
            del self.unpacked_hidden_boxes[0 : self.num_visible_boxes]
        else:
            self.unpacked_visible_boxes = [self.unpacked_hidden_boxes.pop(0)]

        # Reset the list of boxes that are already packed
        self.packed_boxes = self.container.boxes

        # Set the list of visible box sizes in the observation space
        visible_box_sizes = np.asarray(
            [box.size for box in self.unpacked_visible_boxes]
        )
        visible_box_sizes = np.reshape(visible_box_sizes, (self.num_visible_boxes, 3))
        # Reset the state of the environment
        hm = np.asarray(self.container.height_map, dtype=np.int32)
        action_mask = np.asarray(
            self.container.action_mask(box=self.unpacked_visible_boxes[0]),
            dtype=np.int8,
        )

        self.state = {
            "height_map": hm,
            "visible_box_sizes": visible_box_sizes,
            "action_mask": np.reshape(
                action_mask, (self.container.size[0] * self.container.size[1],)
            ),
        }
        self.done = False
        self.seed(seed)

        if return_info:
            return self.state, info
        else:
            return self.state

    def step(self, action: int) -> Tuple[NDArray, float, bool, dict]:
        """Step the environment.
        Parameters:
        -----------
            action: integer with the action to be taken.
        Returns:
        ----------
            observation: Dictionary with the observation of the environment.
            reward: Reward for the action.
            terminated: Whether the episode is terminated.
            info: Dictionary with additional information.
        """

        # Get the index and position of the box to be placed in the container
        box_index, position = self.action_to_position(action)
        # Check if the action is valid
        # TO DO: add parameter check area, add info, return info
        if (
            self.container.check_valid_box_placement(
                self.unpacked_visible_boxes[box_index], position, check_area=100
            )
            == 1
        ):
            # Place the box in the container and delete it from the list of unpacked boxes that are visible to the agent
            if self.num_visible_boxes > 1:
                self.container.place_box(
                    self.unpacked_visible_boxes.pop(box_index), position
                )
            else:
                self.container.place_box(self.unpacked_visible_boxes[0], position)
                self.unpacked_visible_boxes = []
            # Update the list of packed boxes
            self.state["height_map"] = self.container.height_map
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
            self.state["visible_box_sizes"] = []
            # TO DO: add info, return info
        else:
            visible_box_sizes = np.asarray(
                [box.size for box in self.unpacked_visible_boxes]
            )
            visible_box_sizes = np.reshape(
                visible_box_sizes, (self.num_visible_boxes, 3)
            )
            # Update the state of the environment
            self.state["visible_box_sizes"] = visible_box_sizes
            # Update the action mask
            self.state["action_mask"] = np.reshape(
                self.container.action_mask(box=self.unpacked_visible_boxes[0]),
                (self.container.size[0] * self.container.size[1],),
            )

            terminated = False

        return self.state, reward, terminated, {}

    def compute_reward(self) -> float:
        """Compute the reward for the action.
        Returns:
        ----------
            reward: Reward for the action.
        """
        pass

    def render(self, mode="human") -> Union[go.Figure, NDArray]:
        """Render the environment.
        Args:
            mode: Mode to render the environment.
        """
        if mode == "human":
            return self.container.plot()

        elif mode == "rgb_array":
            import io
            from PIL import Image

            fig_png = self.container.plot().to_image(format="png")
            buf = io.BytesIO(fig_png)
            img = Image.open(buf)
            return np.asarray(img, dtype=np.int8)

    def close(self) -> None:
        """Close the environment."""
        pass


if __name__ == "__main__":
    from src.utils import boxes_generator
    from gym import make
    import warnings
    from plotly_gif import GIF

    # Ignore plotly and gym deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Environment initialization
    env = make(
        "PackingEnv-v0",
        container_size=[10, 10, 10],
        box_sizes=boxes_generator([10, 10, 10], 64, 42),
        num_visible_boxes=1,
    )
    obs = env.reset()

    gif = GIF(gif_name="random_rollout.gif", gif_path="../gifs")
    for step_num in range(80):
        fig = env.render()
        gif.create_image(fig)
        action_mask = obs["action_mask"]
        action = env.action_space.sample(mask=action_mask)
        obs, reward, done, info = env.step(action)
        if done:
            break

    gif.create_gif()
    gif.save_gif("random_rollout.gif")
