# Definition of the State and Action Spaces and the Reward Function

We assume that our environment has a container of size `len_edges = (lx,ly,lz)` where $lx,ly,lz$ are the 
integer lengths of the container in the three coordinate axes. In addition, we only consider the case of 
a container that is loaded from the top. 

## State Space
For the state space, we will use a two-dimensional array `height_map` of shape `(lx,ly)` representing the top view of the 
container, where `height_map[i,j]` is the maximum height of a box occupying any position $(i,j,k)$ in the container.

## Action Space
For the action space, we will use a two-dimensional array `action_space` of shape `(lx,ly)`. If the agent places a box in
the position `(i,j)` the bottom-leftmost-back corner of the box will be at position `(i,j,k)`, where the value `k` of
the third coordinate is chosen automatically, that is, the box is either placed on an already existing
box at this position or on the bottom of the container.  We assume for the moment that the agent can only place a box 
at position `(i,j)` if the bottom face of the box is completely supported (no empty space below the box) and that no box
can be placed below an existing box. We automatically check all the feasible locations satisfying these conditions for
a given box with an array `action_mask` of shape `(lx,ly)` where`action_mask[i,j]` is 1 if the agent can place
a box in the position `(i,j)` and 0 otherwise. 


## Reward Function
The reward function given at the agent at the end of each step will be determined by how tight are the boxes packed in
the container. We are still investigating how to measure this.



