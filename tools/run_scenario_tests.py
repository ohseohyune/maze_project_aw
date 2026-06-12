#!/usr/bin/env python3
"""
Run path/IK/headless-MuJoCo checks over many maze CSV files.
"""

import argparse
import glob
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from control.maze_trajectory import allocate_segment_times  # noqa: E402
from maze.parser import read_occupancy_grid, validate_start_goal  # noqa: E402
from maze.planner import bfs_path  # noqa: E402
from maze_main import (  # noqa: E402
    HERE,
    ACTUATOR_FORCE_LIMIT,
    ACTUATOR_KP_SCALE,
    TARGET_TOTAL_TIME,
    TIP_OFFSET,
    build_joint_path,
    configure_tracking_actuators,
    simulate_headless,
    write_planned_path_scene,
)
from robot.omy import OMYConfig, OMYRobot  # noqa: E402


def run_one(csv_path, total_time, kp_scale, force_limit):
    q_seed = np.deg2rad([0, -45, 90, -45, 90, 0])
    grid = read_occupancy_grid(csv_path)
    start, goal = validate_start_goal(grid)
    raw_path = bfs_path(grid, start, goal)

    q_path, cell_path, T_targets = build_joint_path(csv_path, q_seed)
    xml_path = os.path.join(HERE, "reference", "robotis_omy", "scene_maze.xml")
    planned_xml_path = os.path.join(
        HERE, "reference", "robotis_omy", "scene_maze_planned.xml"
    )
    write_planned_path_scene(xml_path, planned_xml_path, T_targets)
    seg_times = allocate_segment_times(q_path, np.deg2rad([180] * 6))
    seg_times *= total_time / seg_times[-1]

    robot = OMYRobot(OMYConfig(xml_path=planned_xml_path, tcp_offset_in_ee=TIP_OFFSET))
    configure_tracking_actuators(robot, kp_scale=kp_scale, force_limit=force_limit)
    robot.reset_home_keyframe(q_home=q_path[0])
    robot.data.ctrl[robot.arm_act_ids] = q_path[0]
    mujoco.mj_forward(robot.model, robot.data)
    log = simulate_headless(robot, q_path, cell_path, seg_times)

    return {
        "csv": csv_path,
        "start": start,
        "goal": goal,
        "raw_len": len(raw_path),
        "waypoints": len(cell_path),
        "time": float(seg_times[-1]),
        "offset_min_mm": float(log["offset"].min() * 1000.0),
        "offset_max_mm": float(log["offset"].max() * 1000.0),
        "abs_gt_5mm": int(np.sum(np.abs(log["offset"] * 1000.0) > 5.0)),
        "contacts": int(log["contact_count"]),
        "qerr_max_deg": float(np.rad2deg(log["q_err"].max())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--time", type=float, default=TARGET_TOTAL_TIME)
    parser.add_argument("--kp-scale", type=float, default=ACTUATOR_KP_SCALE)
    parser.add_argument("--force-limit", type=float, default=ACTUATOR_FORCE_LIMIT)
    args = parser.parse_args()

    files = []
    for pattern in args.patterns:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise SystemExit("No CSV files matched")

    failures = 0
    for path in files:
        try:
            result = run_one(path, args.time, args.kp_scale, args.force_limit)
            ok = (
                result["time"] <= 5.0
                and result["abs_gt_5mm"] == 0
                and result["contacts"] == 0
            )
            status = "PASS" if ok else "RISK"
            if not ok:
                failures += 1
            print(
                f"{status} {os.path.basename(path)} "
                f"raw={result['raw_len']} wp={result['waypoints']} "
                f"time={result['time']:.2f}s "
                f"offset=[{result['offset_min_mm']:.2f},"
                f"{result['offset_max_mm']:.2f}]mm "
                f"abs_gt_5mm={result['abs_gt_5mm']} "
                f"contacts={result['contacts']} "
                f"qerr={result['qerr_max_deg']:.2f}deg"
            )
        except Exception as exc:
            failures += 1
            print(f"FAIL {os.path.basename(path)} {type(exc).__name__}: {exc}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
