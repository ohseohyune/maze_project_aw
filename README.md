# Maze Escape with ROBOTIS OMY

MuJoCo-based maze escape project for a 6-DOF ROBOTIS OMY manipulator. The
system reads a 2D occupancy-grid maze, plans a collision-free cell path, maps
that path onto the folded 3D maze surface, solves inverse kinematics for the
end-effector trajectory, and executes the resulting joint trajectory in MuJoCo.

The repository includes both BFS and A* planners, a custom maze-to-world
geometry model, closed-loop inverse kinematics, joint-space spline trajectory
generation, and automated path/collision validation.

## Features

- Occupancy-grid maze parsing from CSV.
- BFS and A* grid planners with 4-connected motion.
- Collinear path simplification and radius-based corner rounding.
- Fixed-count waypoint resampling for stable IK and visualization.
- 2D path validation against the occupancy grid before 3D conversion.
- Folded maze surface model with per-cell 3D position and surface normal.
- SE(3) target generation for the OMY end-effector.
- Closed-loop inverse kinematics with joint-limit clipping.
- Joint-space cubic spline trajectory execution.
- Headless MuJoCo verification before launching the viewer.
- Planned-path XML overlay generation for visual debugging.

## Requirements

Tested with Python 3. The main external dependencies are:

```bash
pip install numpy mujoco pytest
```

The project uses the MuJoCo Python viewer, so a local graphical environment is
required when running the interactive simulation.

## Quick Start

Run the BFS-based planner and simulation:

```bash
cd maze_project
python maze_bfs_main.py
```

Run the A*-based planner and simulation:

```bash
cd maze_project
python maze_Astar_main.py
```

When the MuJoCo viewer opens, press `SPACE` to start trajectory execution.

## Testing

Run the unit tests from the project root:

```bash
cd maze_project
pytest -q
```

The tests cover grid parsing, path planning utilities, maze geometry, robot
model construction, and inverse kinematics behavior.

## Project Structure

```text
maze_project/
|-- maze_bfs_main.py              # BFS planning, IK, validation, MuJoCo run
|-- maze_Astar_main.py            # A* planning variant
|-- maze_occupancy_grid.csv       # Default maze input
|-- maze/
|   |-- parser.py                 # CSV loading and start/goal validation
|   |-- planner_BFS.py            # BFS, path smoothing, resampling, validation
|   |-- planner_Astar.py          # A* grid planner
|   |-- geometry.py               # 2D cell to 3D folded-maze mapping
|   `-- waypoint_gen.py           # SE(3) target generation
|-- control/
|   |-- clik.py                   # Closed-loop inverse kinematics
|   |-- jacobian.py               # Body Jacobian utilities
|   |-- trajectory.py             # SE(3) trajectory helpers
|   `-- maze_trajectory.py        # Joint-space spline timing
|-- robot/
|   |-- kinematics.py             # Product-of-exponentials kinematics
|   |-- model.py                  # OMY screw-axis and home-pose model
|   `-- omy.py                    # MuJoCo robot wrapper
|-- reference/robotis_omy/        # MuJoCo XML scene and mesh assets
|-- generated_mazes/              # Additional random maze examples
|-- tools/                        # Scenario and maze-generation utilities
`-- tests/                        # Unit tests
```

## Planning and Control Pipeline

The main simulation pipeline is:

1. Load the occupancy grid from CSV.
2. Validate the fixed start and goal cells.
3. Plan a 4-connected grid path using BFS or A*.
4. Remove redundant collinear waypoints while preserving corridor turns.
5. Round corners with a circular arc of approximately `0.35` cell radius.
6. Resample the path to `TARGET_WAYPOINT_COUNT` waypoints.
7. Validate the final 2D float path against occupied cells.
8. Convert each 2D cell waypoint to a 3D pose on the folded maze surface.
9. Solve IK sequentially using the previous solution as the next seed.
10. Allocate joint-space spline timing and scale the run to `TARGET_TOTAL_TIME`.
11. Run a headless MuJoCo check, then launch the interactive viewer.

The planner deliberately avoids diagonal line-of-sight shortcuts. This keeps
the path inside the actual 4-connected maze corridor and reduces the risk of
cutting through walls after smoothing.

## Geometry Model

The maze is represented as a 2D grid for planning, but execution happens on a
3D folded surface. `maze/geometry.py` defines the mapping from `(row, col)` to
world-frame position and surface normal:

- Rows before the fold are mapped to the flat maze plane.
- Rows after the fold are rotated by the configured fold angle.
- The generated end-effector target is offset along the local surface normal.

This separation keeps path planning simple and deterministic while still
allowing the robot to follow the physical 3D maze surface.

## Generated Outputs

Running the main scripts updates MuJoCo scene files with the current maze walls
and planned-path overlays:

- `reference/robotis_omy/scene_maze_planned.xml`
- `reference/robotis_omy/scene_maze_astar_planned.xml`

These files are useful for inspecting the planned route directly in the MuJoCo
viewer.

## Notes

- `maze_bfs_main.py` uses `maze_occupancy_grid.csv` by default.
- `maze_Astar_main.py` currently uses one of the generated maze examples.
- The final path is validated in 2D grid coordinates before IK.
- Collision and surface-offset statistics are printed during the headless check.
