# Definition of the State and Action Spaces and the Reward Function

We assume that our environment has a container of size `len_edges = (lx,ly,lz)` where each entry is the 
lengths of the container in the corresponding coordinate axis. For simplicity, we only consider the case of 
a container that is loaded from the top. 

In addition, we assume that only a number `num_incoming_boxes` of boxes is visible to the agent, 
with their corresponding sizes stored in an array `box_sizes`. The agent should choose a box and its position
to be packed.

## State Space
We use a dictionary state `observation_space` with keys `height_map` and `incoming_box_sizes` to store the state of 
the environment. 

The two-dimensional array contained in the key `height_map` has shape `(lx,ly)` and represents the top view of the 
container, where `height_map[i,j]` is the maximum height of a box occupying any point of the form $(i,j,k)$ in the container,
or $0$ if no box occupies this point. The height map is an object of type `spaces.MultiDiscrete` with shape `(2,1)` and
containing integers in the range `[0,lx), [0,ly]`.

The array contained in the `incoming_box_sizes` key is an object of type `spaces.MultiDiscrete` with shape 
`(num_incoming_boxes,3)` and contains the sizes of the boxes to be packed.

## Action Space
For the action space, we will use a dictionary `action_space` with keys `box_index` and `position` to store the action.

The `box_index` key is an object of type `spaces.Discrete` containing integer values in `[0,num_incoming_boxes)`. 
The two-dimensional array stored in the key `position` represents the position of the box to be packed. Note that only 
two coordinates are specified, because when the agent places a box in the position `(i,j)` the bottom-leftmost-back 
corner of the box will be at position `(i,j,k)`, where the value `k` of the third coordinate is chosen automatically, 
that is, the box is either placed on an already existing box at this position or on the bottom of the container. 

We assume for the moment that the agent can only place a box  at position `(i,j)` if the bottom face of the box is
completely supported (no empty space below the box) and that no box can be placed below an existing box. 
We automatically check all the feasible locations satisfying these conditions for a given box.

## Reward Function
The reward function given at the agent at the end of each step will be determined by how tight are the boxes packed in
the container. We are still investigating how to measure this.
