"""Gymnasium environment for 3D bin packing."""
import copy
from typing import List, Tuple, Union

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
from gymnasium.spaces import Discrete, MultiDiscrete
from numpy.typing import NDArray

from src.packing_kernel import Box, Container
from src.utils import boxes_generator


class PackingEnv(gym.Env):
    """3D bin-packing environment with masked discrete actions.

    Notes
    -----
    The observation is a dictionary with:

    - ``height_map``: flattened 2D container height map.
    - ``visible_box_sizes``: flattened sizes of currently visible boxes.

    Actions are integer indices over ``(box_index, x, y)`` combinations for the
    visible boxes and container grid locations.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        container_size: List[int],
        box_sizes: List[List[int]],
        num_visible_boxes: int = 1,
        render_mode: str = None,
        options: dict = None,
        random_boxes: bool = False,
        only_terminal_reward: bool = True,
    ) -> None:
        """Initialize the packing environment.

        Parameters
        ----------
        container_size : List[int]
            Container size in the form ``[x, y, z]``.
        box_sizes : List[List[int]]
            Sizes of all boxes to place.
        num_visible_boxes : int, default=1
            Number of boxes visible to the agent at each step.
        render_mode : str | None, default=None
            Gymnasium render mode.
        options : dict | None, default=None
            Reserved for Gymnasium compatibility.
        random_boxes : bool, default=False
            If ``True``, regenerate boxes each time ``reset`` is called.
        only_terminal_reward : bool, default=True
            If ``True``, return reward only at episode termination.
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # This flag determines if the boxes are randomly generated everytime the environment is reset
        self.random_boxes = random_boxes
        # This flag determines if the reward is only given at the end of the episode
        self.only_terminal_reward = only_terminal_reward

        # TO DO: Add parameter check box area
        assert num_visible_boxes <= len(box_sizes)
        self.container = Container(container_size)
        # The initial list of all boxes that should be placed in the container.

        self.initial_boxes = [
            Box(box_size, position=[-1, -1, -1], id_=index)
            for index, box_size in enumerate(box_sizes)
        ]

        self.num_initial_boxes = len(self.initial_boxes)

        # The list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = self.initial_boxes.copy()
        # The list of boxes that are already packed
        self.packed_boxes = []
        # The list of boxes that could not be packed (did not fit in the container)
        self.skipped_boxes = []

        # The number and list of boxes that are not yet packed and are visible to the agent
        self.num_visible_boxes = num_visible_boxes
        self.unpacked_visible_boxes = []
        self.state = {}
        self.done = False

        # Array to define the MultiDiscrete space with the list of sizes of the visible boxes
        # The upper bound for the entries in MultiDiscrete space is not inclusive -- we add 1 to each coordinate
        box_repr = np.zeros(shape=(num_visible_boxes, 3), dtype=np.int32)
        box_repr[:] = self.container.size + [1, 1, 1]
        # Reshape the list of sizes of the visible boxes to a 1D array
        box_repr = np.reshape(box_repr, newshape=(num_visible_boxes * 3,))

        # Array to define the MultiDiscrete space with the height map of the container
        height_map_repr = np.ones(
            shape=(container_size[0], container_size[1]), dtype=np.int32
        ) * (container_size[2] + 1)
        # Reshape the height map to a 1D array
        height_map_repr = np.reshape(
            height_map_repr, newshape=(container_size[0] * container_size[1],)
        )

        # Dict to define the observation space
        observation_dict = {
            "height_map": MultiDiscrete(height_map_repr),
            "visible_box_sizes": MultiDiscrete(box_repr),
        }

        # Observation space
        self.observation_space = gym.spaces.Dict(observation_dict)
        # Action space
        self.action_space = Discrete(
            container_size[0] * container_size[1] * num_visible_boxes
        )

        # Set the initial action_mask to a zero array
        self.action_mask = np.zeros(
            shape=(
                self.container.size[0]
                * self.container.size[1]
                * self.num_visible_boxes,
            ),
            dtype=np.int32,
        )

    def action_to_position(self, action: int) -> Tuple[int, NDArray]:
        """Convert a flat action index to box index and grid position.

        Parameters
        ----------
        action : int
            Encoded action.

        Returns
        -------
        Tuple[int, NDArray]
            Pair ``(box_index, position)`` for the selected action.
        """
        box_index = action // (self.container.size[0] * self.container.size[1])
        res = action % (self.container.size[0] * self.container.size[1])

        position = np.array(
            [res // self.container.size[0], res % self.container.size[0]]
        )

        return box_index, position.astype(np.int32)

    def position_to_action(self, position, box_index=0):
        """Convert a box index and position to a flat action index.

        Parameters
        ----------
        position : Sequence[int]
            Target position in the container grid.
        box_index : int, default=0
            Index of the selected visible box.

        Returns
        -------
        int
            Encoded action index.
        """
        action = (
            box_index * self.container.size[0] * self.container.size[1]
            + position[0] * self.container.size[0]
            + position[1]
        )
        return action

    def reset(self, seed=None, options=None) -> Tuple:
        """Reset the environment state.

        Parameters
        ----------
        seed : int | None
            Random seed used by Gymnasium.
        options : dict | None
            Additional reset options (unused).

        Returns
        -------
        Tuple[dict, dict]
            Initial observation and info dictionary.
        """

        super().reset(seed=seed)
        self.container.reset()

        if self.random_boxes:
            box_sizes = boxes_generator(
                self.container.size, num_items=self.num_initial_boxes
            )
            self.initial_boxes = [
                Box(box_size, position=[-1, -1, -1], id_=index)
                for index, box_size in enumerate(box_sizes)
            ]

        # Reset the list of boxes that are not yet packed and not visible to the agent
        self.unpacked_hidden_boxes = copy.deepcopy(self.initial_boxes)

        # Reset the list of boxes visible to the agent and deletes them from the list of
        # hidden unpacked boxes to be packed
        self.unpacked_visible_boxes = copy.deepcopy(
            self.unpacked_hidden_boxes[0 : self.num_visible_boxes]
        )
        del self.unpacked_hidden_boxes[0 : self.num_visible_boxes]

        # Reset the list of boxes that are already packed
        self.packed_boxes = self.container.boxes

        # Set the list of visible box sizes in the observation space
        visible_box_sizes = np.asarray(
            [box.size for box in self.unpacked_visible_boxes]
        )

        # Reset the state of the environment
        hm = np.asarray(self.container.height_map, dtype=np.int32)
        hm = np.reshape(hm, (self.container.size[0] * self.container.size[1],))

        # Set the initial blank action_mask
        self.action_mask = self.action_masks()

        vbs = np.reshape(visible_box_sizes, (self.num_visible_boxes * 3,))
        self.state = {"height_map": hm, "visible_box_sizes": vbs}

        self.done = False

        return self.state, {}

    def calculate_reward(self, reward_type: str = "terminal_step") -> float:
        """Calculate reward according to the configured reward mode.

        Parameters
        ----------
        reward_type : str, default="terminal_step"
            Reward strategy, either ``terminal_step`` or ``interm_step``.

        Returns
        -------
        float
            Reward value.
        """
        # Volume of packed boxes
        packed_volume = np.sum([box.volume for box in self.packed_boxes])

        if reward_type == "terminal_step":
            # Reward for the terminal step
            container_volume = self.container.volume
            reward = packed_volume / container_volume
        elif reward_type == "interm_step":
            min_x = min([box.position[0] for box in self.packed_boxes])
            min_y = min([box.position[1] for box in self.packed_boxes])
            min_z = min([box.position[2] for box in self.packed_boxes])
            max_x = max([box.position[0] + box.size[0] for box in self.packed_boxes])
            max_y = max([box.position[1] + box.size[1] for box in self.packed_boxes])
            max_z = max([box.position[2] + box.size[2] for box in self.packed_boxes])

            # Reward for the intermediate step
            reward = packed_volume / (
                (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
            )
        else:
            raise ValueError("Invalid reward type")

        return reward

    def step(self, action: int) -> Tuple[NDArray, float, bool, bool, dict]:
        """Apply one environment step.

        Parameters
        ----------
        action : int
            Encoded action selected by the policy.

        Returns:
        -------
        Tuple[dict, float, bool, bool, dict]
            Observation, reward, terminated flag, truncated flag, and info.
        """
        truncated = False

        # Get the index and position of the box to be placed in the container
        box_index, position = self.action_to_position(action)
        # if the box is a dummy box, skip the step
        if box_index >= len(self.unpacked_visible_boxes):
            return self.state, 0, self.done, truncated, {}

        # If it is not a dummy box, check if the action is valid
        # TO DO: add parameter check area, add info, return info
        if (
            self.container.check_valid_box_placement(
                self.unpacked_visible_boxes[box_index], position, check_area=100
            )
            == 1
        ):
            # Place the box in the container and delete it from the list of unpacked visible boxes
            if self.num_visible_boxes > 1:
                self.container.place_box(
                    self.unpacked_visible_boxes.pop(box_index), position
                )
            else:
                self.container.place_box(self.unpacked_visible_boxes[0], position)
                self.unpacked_visible_boxes = []
            # Update the height map, reshapes it and adds it to the observation space
            self.state["height_map"] = np.reshape(
                self.container.height_map,
                (self.container.size[0] * self.container.size[1],),
            )
            # Update the list of packed boxes
            self.packed_boxes = self.container.boxes
            # set reward
            if self.only_terminal_reward:
                reward = 0
            else:
                reward = self.calculate_reward(reward_type="interm_step")

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
            reward = self.calculate_reward(reward_type="terminal_step")
            self.state["visible_box_sizes"] = [[0, 0, 0]] * self.num_visible_boxes
            return self.state, reward, terminated, truncated, {}

        if len(self.unpacked_visible_boxes) == self.num_visible_boxes:
            # Update the list of visible box sizes in the observation space
            visible_box_sizes = np.asarray(
                [box.size for box in self.unpacked_visible_boxes]
            )
            self.state["visible_box_sizes"] = np.reshape(
                visible_box_sizes, (self.num_visible_boxes * 3,)
            )
            terminated = False
            return self.state, reward, terminated, truncated, {}

        if len(self.unpacked_visible_boxes) < self.num_visible_boxes:
            # If there are fewer boxes than the maximum number of visible boxes, add dummy boxes
            dummy_box_size = self.container.size
            num_dummy_boxes = self.num_visible_boxes - len(self.unpacked_visible_boxes)
            box_size_list = [box.size for box in self.unpacked_visible_boxes] + [
                dummy_box_size
            ] * num_dummy_boxes
            visible_box_sizes = np.asarray(box_size_list)
            self.state["visible_box_sizes"] = np.reshape(
                visible_box_sizes, (self.num_visible_boxes * 3,)
            )
            terminated = False
            return self.state, reward, terminated, truncated, {}

    def action_masks(self) -> NDArray:
        """Return a boolean mask of valid actions.

        Returns
        ----------
        NDArray
            Flattened boolean action mask.
        """
        act_mask = np.zeros(
            shape=(
                self.num_visible_boxes,
                self.container.size[0] * self.container.size[1],
            ),
            dtype=np.int8,
        )

        for index in range(len(self.unpacked_visible_boxes)):
            acm = self.container.action_mask(
                box=self.unpacked_visible_boxes[index], check_area=100
            )
            act_mask[index] = np.reshape(
                acm, (self.container.size[0] * self.container.size[1],)
            )
        return np.asarray(act_mask.flatten() == 1, dtype=bool)

    @property
    def get_action_mask(self) -> NDArray:
        """Compatibility accessor used by legacy tests and wrappers."""
        return self.action_masks().astype(np.int8)

    def render(self, mode=None) -> Union[go.Figure, NDArray]:
        """Render the environment.

        Parameters
        ----------
        mode : str | None
            Render mode. If ``None``, uses ``self.render_mode``.

        Returns
        -------
        go.Figure | NDArray | None
            Plotly figure for ``human``, image array for ``rgb_array``, or ``None``.
        """

        if mode is None:
            mode = self.render_mode

        if mode is None:
            return None

        elif mode == "human":
            fig = self.container.plot()
            # fig.show()
            return fig
        #
        elif mode == "rgb_array":
            import io
            from PIL import Image

            fig_png = self.container.plot().to_image(format="png")
            buf = io.BytesIO(fig_png)
            img = Image.open(buf)
            return np.asarray(img, dtype=np.int8)
        else:
            raise NotImplementedError

    def close(self) -> None:
        """Close the environment."""
        pass


if __name__ == "__main__":
    from src.utils import boxes_generator
    from gymnasium import make
    import warnings
    from plotly_gif import GIF

    # Ignore plotly and gymnasium deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Environment initialization
    env = make(
        "PackingEnv-v0",
        container_size=[10, 10, 10],
        box_sizes=boxes_generator([10, 10, 10], 64, 42),
        num_visible_boxes=1,
    )
    obs, _ = env.reset()

    gif = GIF(gif_name="random_rollout.gif", gif_path="../gifs")
    for step_num in range(80):
        fig = env.render()
        gif.create_image(fig)
        action_mask = env.get_action_mask
        action = env.action_space.sample(mask=action_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    gif.create_gif()
    gif.save_gif("random_rollout.gif")
