# Maze Escape with OMY

Pure Python MuJoCo project for the term-project maze task.

## Run

```bash
cd /home/seohy/colcon_ws/src/maze2/maze_project
python maze_bfs_main.py
```

## Test

```bash
cd /home/seohy/colcon_ws/src/maze2/maze_project
pytest -q
```

## Layout

- `maze/`: CSV parsing, geometry, BFS/LOS planning, waypoint generation
- `robot/`: OMY kinematics and MuJoCo wrapper
- `control/`: CLIK, SE(3) interpolation, joint-space trajectory
- `reference/robotis_omy/`: MuJoCo scene and mesh assets
- `maze_occupancy_grid.csv`: replaceable maze input

Default maze targets keep the tip 1 mm above the maze surface, and
`maze_bfs_main.py` scales the joint trajectory to 4.0 seconds.
The planned cell path follows the 4-connected BFS corridor route; diagonal
line-of-sight shortcuts are not used. The corridor route is rounded at corners
and resampled to a fixed waypoint count before IK. The tool orientation normal
is smoothed around the fold boundary to avoid joint-space spline lift-off or
surface penetration.
Float waypoints are validated against the occupancy grid before IK, and
`maze_bfs_main.py` runs a headless MuJoCo check before opening the viewer.
`maze_bfs_main.py` also generates `reference/robotis_omy/scene_maze_planned.xml`
with a blue planned-path overlay, then boosts the MuJoCo position actuators for
tighter tracking during the 4 second run.
