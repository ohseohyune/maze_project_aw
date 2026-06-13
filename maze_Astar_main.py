"""
Entry point for running the maze project with A* path planning.
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from control.clik import solve_ik
from control.maze_trajectory import allocate_segment_times, joint_spline_traj
from maze.parser import read_occupancy_grid, validate_start_goal
from maze.planner_BFS import (
    bfs_path,
    corner_round,
    resample_path,
    simplify_collinear,
    validate_free_path,
)
from maze.planner_Astar import astar_path
from maze.waypoint_gen import build_se3_targets
from maze_bfs_main import (
    ACTUATOR_FORCE_LIMIT,
    ACTUATOR_KP_SCALE,
    HERE,
    Q_MAX,
    Q_MIN,
    TARGET_TOTAL_TIME,
    TARGET_WAYPOINT_COUNT,
    TIP_OFFSET,
    TRACKING_LEAD_TIME,
    analyze_collision,
    configure_tracking_actuators,
    simulate_headless,
    write_maze_walls_scene,
    write_planned_path_scene,
    write_simplified_path_scene,
)
from robot.model import define_model
from robot.omy import OMYConfig, OMYRobot


def build_astar_joint_path(csv_path: str, q_init: np.ndarray):
    grid = read_occupancy_grid(csv_path)
    start, goal = validate_start_goal(grid)

    bfs_raw_path = bfs_path(grid, start, goal)
    raw_path = astar_path(grid, start, goal)
    print(
        "Planner comparison: "
        f"BFS cells={len(bfs_raw_path)}, A* cells={len(raw_path)}"
    )

    corridor_path = simplify_collinear(raw_path)
    rounded_path = corner_round(corridor_path, n_subdiv=3)
    cell_path = resample_path(rounded_path, target_count=TARGET_WAYPOINT_COUNT)
    validate_free_path(cell_path, grid, allow_final_exit=False)
    T_targets = build_se3_targets(cell_path)

    _, M, B_list = define_model(tcp_offset=TIP_OFFSET)
    q_path = []
    q = np.asarray(q_init, dtype=float).reshape(6)
    K_p = np.diag([4.0, 4.0, 4.0, 8.0, 8.0, 8.0])

    for idx, T in enumerate(T_targets):
        q, ok = solve_ik(
            T,
            B_list,
            M,
            q,
            K_p,
            max_iter=600,
            tol=2e-4,
            joint_lower_limits=Q_MIN,
            joint_upper_limits=Q_MAX,
            damping=0.06,
            dt=0.04,
        )
        if not ok:
            raise RuntimeError(f"IK failed at waypoint {idx}: p={T[:3, 3]}")
        q_path.append(q.copy())

    return np.vstack(q_path), cell_path, T_targets


def main():
    xml_path = os.path.join(HERE, "reference", "robotis_omy", "scene_maze.xml")
    planned_xml_path = os.path.join(
        HERE, "reference", "robotis_omy", "scene_maze_astar_planned.xml"
    )
    csv_path = os.path.join(HERE, "generated_mazes", "random_maze_03.csv")

    write_maze_walls_scene(csv_path, xml_path)

    q_seed = np.deg2rad(np.array([0, -45, 90, -45, 90, 0], dtype=float))
    q_path, cell_path, T_targets = build_astar_joint_path(csv_path, q_seed)
    write_planned_path_scene(xml_path, planned_xml_path, T_targets)
    grid = read_occupancy_grid(csv_path)
    start, goal = validate_start_goal(grid)
    raw_path = astar_path(grid, start, goal)
    simplified_path = simplify_collinear(raw_path)
    write_simplified_path_scene(planned_xml_path, planned_xml_path, simplified_path)
    robot = OMYRobot(OMYConfig(xml_path=planned_xml_path, tcp_offset_in_ee=TIP_OFFSET))
    configure_tracking_actuators(
        robot,
        kp_scale=ACTUATOR_KP_SCALE,
        force_limit=ACTUATOR_FORCE_LIMIT,
    )

    q_dot_max = np.deg2rad(np.array([180, 180, 180, 180, 180, 180], dtype=float))
    seg_times = allocate_segment_times(q_path, q_dot_max)
    if seg_times[-1] > 0.0:
        seg_times = seg_times * (TARGET_TOTAL_TIME / seg_times[-1])

    print(f"A* waypoints: {len(cell_path)}")
    print(f"Simplified A* path points: {len(simplified_path)}")
    print(f"Estimated total time: {seg_times[-1]:.2f}s")
    print(f"Start q(deg): {np.rad2deg(q_path[0])}")
    print(f"Goal  q(deg): {np.rad2deg(q_path[-1])}")

    robot.reset_home_keyframe(q_home=q_path[0])
    robot.data.ctrl[robot.arm_act_ids] = q_path[0]
    mujoco.mj_forward(robot.model, robot.data)

    verify_log = simulate_headless(robot, q_path, cell_path, seg_times)
    print(
        "Headless check: "
        f"offset=[{verify_log['offset'].min()*1000:.2f}, "
        f"{verify_log['offset'].max()*1000:.2f}] mm, "
        f"contacts={verify_log['contact_count']}, "
        f"floor_collisions={verify_log['floor_collision_count']}, "
        f"wall_collisions={verify_log['wall_collision_count']}, "
        f"max q_err={np.rad2deg(verify_log['q_err'].max()):.2f} deg"
    )
    robot.reset_home_keyframe(q_home=q_path[0])
    robot.data.ctrl[robot.arm_act_ids] = q_path[0]

    collision_count = 0
    floor_collision_count = 0
    wall_collision_count = 0
    goal_wall_elapsed = None
    goal_sim_time = None
    start_state = {"started": False, "t0": None}

    def key_callback(key):
        if key == ord(" ") and not start_state["started"]:
            start_state["started"] = True
            start_state["t0"] = time.perf_counter()
            print("Started trajectory.")

    print("Press SPACE in the MuJoCo viewer to start.")
    with mujoco.viewer.launch_passive(
        robot.model,
        robot.data,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if not start_state["started"]:
                robot.data.ctrl[robot.arm_act_ids] = q_path[0]
                mujoco.mj_forward(robot.model, robot.data)
                viewer.sync()
                time.sleep(0.01)
                continue

            t = time.perf_counter() - start_state["t0"]
            if goal_wall_elapsed is None and t >= seg_times[-1]:
                goal_wall_elapsed = t
                goal_sim_time = robot.data.time
            if t > seg_times[-1] + 0.5:
                break

            q_des, qdot_des, _ = joint_spline_traj(t, q_path, seg_times)
            q_cmd = np.clip(
                q_des + TRACKING_LEAD_TIME * qdot_des,
                robot.arm_qpos_lo,
                robot.arm_qpos_hi,
            )
            robot.data.ctrl[robot.arm_act_ids] = q_cmd
            mujoco.mj_step(robot.model, robot.data)
            collision_count += int(robot.data.ncon)

            floor_col, wall_col = analyze_collision(robot)
            floor_collision_count += floor_col
            wall_collision_count += wall_col

            viewer.sync()

    tip_pos = robot.tcp_position_world()
    if goal_wall_elapsed is None:
        goal_wall_elapsed = (
            time.perf_counter() - start_state["t0"]
            if start_state["started"]
            else 0.0
        )
        goal_sim_time = robot.data.time
    print(f"Actual wall-clock start-to-goal time: {goal_wall_elapsed:.2f}s")
    print(f"MuJoCo sim time at goal command: {goal_sim_time:.2f}s")
    print(f"Final MuJoCo sim time: {robot.data.time:.2f}s")
    print(f"Final tip position: {tip_pos}")
    print(f"Total collision samples: {collision_count}")
    print(f"Floor collisions: {floor_collision_count}")
    print(f"Wall collisions: {wall_collision_count}")


if __name__ == "__main__":
    main()
