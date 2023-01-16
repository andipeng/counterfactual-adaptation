from minigrid.minigrid import Door, Goal, Grid, Key, Lava, Ball, MiniGridEnv, MissionSpace
import numpy as np

class DoorKeyEnv(MiniGridEnv):

    """
    ### Description

    This environment has a key that the agent must pick up in order to unlock a
    goal and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ### Mission Space

    "use the key to open the door and then get to the goal"

    ### Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-DoorKey-5x5-v0`
    - `MiniGrid-DoorKey-6x6-v0`
    - `MiniGrid-DoorKey-8x8-v0`
    - `MiniGrid-DoorKey-16x16-v0`

    """

    def __init__(self, size=8, **kwargs):
        if "max_steps" not in kwargs:
            kwargs["max_steps"] = 10 * size * size
        mission_space = MissionSpace(
            mission_func=lambda: "use the key to open the door and then get to the goal"
        )
        super().__init__(mission_space=mission_space, grid_size=size, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeySpecial(MiniGridEnv):
    def __init__(self, size=5, agent_start_pos=(1, 1), agent_start_dir=0, tile_size=64, **kwargs):
        if "max_steps" not in kwargs:
            kwargs["max_steps"] = 10 * size * size

        mission_space = MissionSpace(
            mission_func=lambda: "use the key to open the door and then get to the goal"
        )

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(mission_space=mission_space, grid_size=size, **kwargs)

    def get_color(self, color):
        if color == 1:
            return "red"
        elif color == 2:
            return "green"
        elif color == 3:
            return "blue"
        elif color == 4:
            return "yellow"

    def _gen_grid(self, width=8, height=8, keydist=True, doordist=True, goaldist=True,
                    key_locations=[2,7], key_colors=[1,5], keydist_locations=[2,7],
                    door_locations=[2,7], door_colors=[1,5], doordist_locations=[2,7],
                    goal_locations=[2,7], goal_colors=[1,5], goaldist_locations=[2,7]):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width - 2)
        splitIdx = 5
        self.grid.vert_wall(splitIdx, 0)

        #self.place_agent(size=(splitIdx, height))
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.key_location = np.array([3, self._rand_int(key_locations[0], key_locations[1])])
        self.key_color = np.array([self._rand_int(key_colors[0], key_colors[1])])
        self.put_obj(Key(self.get_color(self.key_color[0])), self.key_location[0], self.key_location[1])

        if keydist:
            self.keydist_location = np.array([3, self._rand_int(keydist_locations[0], keydist_locations[1])])
            self.put_obj(Lava(), self.keydist_location[0], self.keydist_location[1])

        self.door_location = np.array([splitIdx, self._rand_int(door_locations[0], door_locations[1])])
        self.door_color = np.array([self._rand_int(door_colors[0], door_colors[1])])
        self.put_obj(Door(self.get_color(self.door_color[0]), is_locked=True), self.door_location[0], self.door_location[1])

        if doordist:
            self.doordist_location = np.array([splitIdx, self._rand_int(doordist_locations[0], doordist_locations[1])])
            self.put_obj(Lava(), self.doordist_location[0], self.doordist_location[1])

        self.goal_location = np.array([10, self._rand_int(goal_locations[0], goal_locations[1])])
        self.goal_color = np.array([self._rand_int(goal_colors[0], goal_colors[1])])
        self.put_obj(Goal(self.get_color(self.goal_color[0])), self.goal_location[0], self.goal_location[1])

        if goaldist:
            self.goaldist_location = np.array([10, self._rand_int(goaldist_locations[0], goaldist_locations[1])])
            self.put_obj(Lava(), 10, self.goaldist_location[1])

        self.goal_lang = np.array(['go to the ' + self.get_color(self.key_color[0]) + ' key ' 
                                    + self.get_color(self.door_color[0]) + ' door '
                                    + self.get_color(self.goal_color[0]) + ' goal'])
        self.goal = np.concatenate([self.key_location, self.key_color, 
                                    self.door_location, self.door_color, 
                                    self.goal_location, self.goal_color])

        self.mission = "use the key to open the door and then get to the goal"

    def reset(self, *, seed=None, key=True, keydist=True, doordist=True, goaldist=True,
                    key_locations=[1,11], key_colors=[1,5], keydist_locations=[1,11],
                    door_locations=[1,11], door_colors=[1,5], doordist_locations=[1,11],
                    goal_locations=[1,11], goal_colors=[1,5], goaldist_locations=[1,11],
                    options=None):

        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height, keydist, doordist, goaldist,
                        key_locations, key_colors, keydist_locations,
                        door_locations, door_colors, doordist_locations,
                        goal_locations, goal_colors, goaldist_locations)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}