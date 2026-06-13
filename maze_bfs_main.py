"""
Entry point for the Maze Escape with OMY project.
"""

import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from control.clik import solve_ik
from control.maze_trajectory import allocate_segment_times, joint_spline_traj
from maze.geometry import (
    MAZE_POS_WORLD, CELL_SIZE, FLAT_TOP_Z, WALL_HALF_HEIGHT,
    FOLD_ANGLE, FOLD_HINGE_ROW, FOLD_SWITCH_ROW,
    FOLD_R6_Y_LOCAL, FOLD_R6_Z_LOCAL,
    cell_to_world,
)
from maze.parser import read_occupancy_grid, validate_start_goal
from maze.planner_BFS import (
    bfs_path,
    corner_round,
    resample_path,
    simplify_collinear,
    validate_free_path,
)
from maze.waypoint_gen import build_se3_targets
from robot.model import define_model
from robot.omy import OMYConfig, OMYRobot


HERE = os.path.dirname(os.path.abspath(__file__))
TIP_OFFSET = np.array([0.0, -0.315, 0.0], dtype=float)
Q_MIN = np.array([-2 * np.pi, -2 * np.pi, -2.618, -2 * np.pi, -2 * np.pi, -2 * np.pi])
Q_MAX = np.array([2 * np.pi, 2 * np.pi, 2.618, 2 * np.pi, 2 * np.pi, 2 * np.pi])
TARGET_TOTAL_TIME = 4.5
TRACKING_LEAD_TIME = 0.005
ACTUATOR_KP_SCALE = 5.5
# ACTUATOR_FORCE_LIMIT = 350.0
ACTUATOR_FORCE_LIMIT = 350.0
TARGET_WAYPOINT_COUNT = 200


def build_joint_path(csv_path: str, q_init: np.ndarray):
    grid = read_occupancy_grid(csv_path)
    start, goal = validate_start_goal(grid)
    
    # BFS로 기본 경로 찾기
    raw_path = bfs_path(grid, start, goal)
    # 직선 구간의 중간 점 제거
    corridor_path = simplify_collinear(raw_path)
    # 코너 근처에 중간 점들을 추가해서 부드러운 곡선처럼 만들기
    rounded_path = corner_round(corridor_path, n_subdiv=3)
    # 정해진 개수의 waypoint로 다시 샘플링
    cell_path = resample_path(rounded_path, target_count=TARGET_WAYPOINT_COUNT)
    # 최종 경로가 벽을 지나지 않는지 검사
    validate_free_path(cell_path, grid, allow_final_exit=False)
    # 2D cell path를 로봇이 따라갈 3D pose로 변환
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

def write_maze_walls_scene(csv_path: str, xml_path: str) -> None:
    """CSV의 벽(1) 셀에 맞게 scene XML의 maze_cube_* geom을 재생성한다."""
    grid = read_occupancy_grid(csv_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    maze_body = root.find(".//body[@name='maze']")
    if maze_body is None:
        raise RuntimeError("Could not find maze body in scene XML")

    for geom in list(maze_body):
        if geom.attrib.get("name", "").startswith("maze_cube_"):
            maze_body.remove(geom)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 1:
                continue
            x = -0.22 + CELL_SIZE * c
            if float(r) < FOLD_SWITCH_ROW:
                y = -0.22 + CELL_SIZE * r
                z = FLAT_TOP_Z + WALL_HALF_HEIGHT
                attribs = {
                    "name": f"maze_cube_r{r:02d}_c{c:02d}",
                    "type": "box",
                    "pos": f"{x:.4f} {y:.4f} {z:.4f}",
                    "size": "0.0200 0.0200 0.0200",
                    "material": "maze_wall",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "1 0.005 0.0001",
                }
            else:
                row_offset = float(r) - FOLD_HINGE_ROW
                y = FOLD_R6_Y_LOCAL + row_offset * CELL_SIZE * np.cos(FOLD_ANGLE)
                z = FOLD_R6_Z_LOCAL + row_offset * CELL_SIZE * np.sin(FOLD_ANGLE)
                attribs = {
                    "name": f"maze_cube_r{r:02d}_c{c:02d}",
                    "type": "box",
                    "pos": f"{x:.4f} {y:.4f} {z:.4f}",
                    "quat": "0.9659258263 0.2588190451 0 0",
                    "size": "0.0200 0.0200 0.0200",
                    "material": "maze_wall",
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "1 0.005 0.0001",
                }
            ET.SubElement(maze_body, "geom", attribs)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


#  계획된 경로를 MuJoCo XML 파일 안에 시각화용 object로 추가하는 함수
def write_planned_path_scene(base_xml_path: str, output_xml_path: str, T_targets):
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is not None and asset.find("material[@name='planned_path']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "planned_path",
                "rgba": "0.05 0.45 1.0 0.85",
                "emission": "0.15",
            },
        )
    if asset is not None and asset.find("material[@name='planned_waypoint']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "planned_waypoint",
                "rgba": "1.0 0.78 0.05 1.0",
                "emission": "0.25",
            },
        )
    if asset is not None and asset.find("material[@name='planned_start']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "planned_start",
                "rgba": "0.05 1.0 0.2 1.0",
                "emission": "0.25",
            },
        )
    if asset is not None and asset.find("material[@name='planned_goal']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "planned_goal",
                "rgba": "1.0 0.12 0.08 1.0",
                "emission": "0.25",
            },
        )

    maze_body = root.find(".//body[@name='maze']")
    if maze_body is None:
        raise RuntimeError("Could not find maze body in scene XML")

    for geom in list(maze_body):
        name = geom.attrib.get("name", "")
        if name.startswith("planned_path_"):
            maze_body.remove(geom)
        elif name.startswith("planned_waypoint_"):
            maze_body.remove(geom)

    points = [np.asarray(T[:3, 3], dtype=float) - MAZE_POS_WORLD for T in T_targets]
    for i, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        ET.SubElement(
            maze_body,
            "geom",
            {
                "name": f"planned_path_seg_{i:03d}",
                "type": "capsule",
                "fromto": (
                    f"{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f} "
                    f"{p1[0]:.5f} {p1[1]:.5f} {p1[2]:.5f}"
                ),
                "size": "0.0022",
                "material": "planned_path",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    for i, p in enumerate(points):
        if i == 0:
            material = "planned_start"
            size = "0.007"
        elif i == len(points) - 1:
            material = "planned_goal"
            size = "0.007"
        else:
            material = "planned_waypoint"
            size = "0.0032"
        ET.SubElement(
            maze_body,
            "geom",
            {
                "name": f"planned_waypoint_{i:03d}",
                "type": "sphere",
                "pos": f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}",
                "size": size,
                "material": material,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return output_xml_path


def write_simplified_path_scene(
    base_xml_path: str,
    output_xml_path: str,
    simplified_path: list,
) -> str:
    """Add post-simplify_collinear path markers to the MuJoCo maze body."""
    tree = ET.parse(base_xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is not None and asset.find("material[@name='simplified_path']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "simplified_path",
                "rgba": "1.00 0.05 0.00 1.00",
                "emission": "0.45",
            },
        )
    if asset is not None and asset.find("material[@name='simplified_waypoint']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "simplified_waypoint",
                "rgba": "1.00 0.85 0.00 1.00",
                "emission": "0.50",
            },
        )

    maze_body = root.find(".//body[@name='maze']")
    if maze_body is None:
        raise RuntimeError("Could not find maze body in scene XML")

    for geom in list(maze_body):
        name = geom.attrib.get("name", "")
        if (
            name.startswith("raw_grid_path_")
            or name.startswith("raw_grid_waypoint_")
            or name.startswith("simplified_path_")
            or name.startswith("simplified_waypoint_")
        ):
            maze_body.remove(geom)

    points = []
    for r, c in simplified_path:
        p_world, _ = cell_to_world(r, c, hover=0.025)
        points.append(p_world - MAZE_POS_WORLD)

    for i, (p0, p1) in enumerate(zip(points[:-1], points[1:])):
        ET.SubElement(
            maze_body,
            "geom",
            {
                "name": f"simplified_path_seg_{i:03d}",
                "type": "capsule",
                "fromto": (
                    f"{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f} "
                    f"{p1[0]:.5f} {p1[1]:.5f} {p1[2]:.5f}"
                ),
                "size": "0.0045",
                "material": "simplified_path",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    for i, p in enumerate(points):
        ET.SubElement(
            maze_body,
            "geom",
            {
                "name": f"simplified_waypoint_{i:03d}",
                "type": "sphere",
                "pos": f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}",
                "size": "0.0070",
                "material": "simplified_waypoint",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return output_xml_path


def configure_tracking_actuators(
    robot: OMYRobot,
    kp_scale: float = ACTUATOR_KP_SCALE,
    force_limit: float = ACTUATOR_FORCE_LIMIT,
) -> None:
    for act_id in robot.arm_act_ids:
        kp = robot.model.actuator_gainprm[act_id, 0] * kp_scale
        robot.model.actuator_gainprm[act_id, 0] = kp
        robot.model.actuator_biasprm[act_id, 1] = -kp
        robot.model.actuator_forcerange[act_id] = [-force_limit, force_limit]


def analyze_collision(robot: OMYRobot):
    """충돌 정보를 분석하여 바닥/벽 충돌 구분"""
    floor_collision_count = 0
    wall_collision_count = 0
    
    for i in range(robot.data.ncon):
        contact = robot.data.contact[i]
        geom_id_1 = contact.geom1
        geom_id_2 = contact.geom2
        
        geom_name_1 = robot.model.geom(geom_id_1).name
        geom_name_2 = robot.model.geom(geom_id_2).name
        
        # 바닥(maze_floor_*) 충돌 확인
        if geom_name_1.startswith("maze_floor_") or geom_name_2.startswith("maze_floor_"):
            floor_collision_count += 1
        # 벽(maze_cube_*) 충돌 확인
        elif geom_name_1.startswith("maze_cube_") or geom_name_2.startswith("maze_cube_"):
            wall_collision_count += 1
    
    return floor_collision_count, wall_collision_count


def simulate_headless(robot: OMYRobot, q_path: np.ndarray, cell_path: list, seg_times):
    rows = []
    contact_count = 0
    floor_collision_count = 0
    wall_collision_count = 0
    n_steps = int(np.ceil(seg_times[-1] / robot.model.opt.timestep)) + 1

    for step in range(n_steps):
        t = step * robot.model.opt.timestep
        q_des, qdot_des, _ = joint_spline_traj(min(t, seg_times[-1]), q_path, seg_times)
        q_cmd = np.clip(
            q_des + TRACKING_LEAD_TIME * qdot_des,
            robot.arm_qpos_lo,
            robot.arm_qpos_hi,
        )
        robot.data.ctrl[robot.arm_act_ids] = q_cmd
        robot.data.qfrc_applied[:] = 0.0
        robot.data.qfrc_applied[robot.arm_qpos_adrs] = robot.data.qfrc_bias[
            robot.arm_qpos_adrs
        ]
        mujoco.mj_step(robot.model, robot.data)
        contact_count += int(robot.data.ncon)
        
        # 충돌 분석
        floor_col, wall_col = analyze_collision(robot)
        floor_collision_count += floor_col
        wall_collision_count += wall_col

        idx = int(np.searchsorted(seg_times, t, side="right") - 1)
        idx = max(0, min(idx, len(cell_path) - 2))
        t0, t1 = seg_times[idx], seg_times[idx + 1]
        alpha = np.clip((t - t0) / (t1 - t0), 0.0, 1.0) if t1 > t0 else 0.0
        rc = (1.0 - alpha) * np.array(cell_path[idx]) + alpha * np.array(
            cell_path[idx + 1]
        )
        p_surface, n = cell_to_world(rc[0], rc[1], hover=0.0)
        p_tip = robot.tcp_position_world()
        offset = float(np.dot(p_tip - p_surface, n))
        q_err = float(
            np.linalg.norm(robot.data.qpos[robot.arm_qpos_adrs] - q_des, ord=np.inf)
        )
        rows.append((t, offset, q_err))

    arr = np.asarray(rows, dtype=float)
    return {
        "time": arr[:, 0],
        "offset": arr[:, 1],
        "q_err": arr[:, 2],
        "contact_count": contact_count,
        "floor_collision_count": floor_collision_count,
        "wall_collision_count": wall_collision_count,
    }


def main():
    xml_path = os.path.join(HERE, "reference", "robotis_omy", "scene_maze.xml")
    planned_xml_path = os.path.join(
        HERE, "reference", "robotis_omy", "scene_maze_planned.xml"
    )
    csv_path = os.path.join(HERE, "maze_occupancy_grid.csv")
    # csv_path = os.path.join(HERE, "generated_mazes", "random_maze_02.csv")


    write_maze_walls_scene(csv_path, xml_path)

    q_seed = np.deg2rad(np.array([0, -45, 90, -45, 90, 0], dtype=float))
    q_path, cell_path, T_targets = build_joint_path(csv_path, q_seed)
    write_planned_path_scene(xml_path, planned_xml_path, T_targets)
    grid = read_occupancy_grid(csv_path)
    start, goal = validate_start_goal(grid)
    raw_path = bfs_path(grid, start, goal)
    simplified_path = simplify_collinear(raw_path)
    write_simplified_path_scene(planned_xml_path, planned_xml_path, simplified_path)
    robot = OMYRobot(OMYConfig(xml_path=planned_xml_path, tcp_offset_in_ee=TIP_OFFSET))
    configure_tracking_actuators(robot)

    q_dot_max = np.deg2rad(np.array([180, 180, 180, 180, 180, 180], dtype=float))
    seg_times = allocate_segment_times(q_path, q_dot_max)
    if seg_times[-1] > 0.0:
        seg_times = seg_times * (TARGET_TOTAL_TIME / seg_times[-1])

    print(f"Waypoints: {len(cell_path)}")
    print(f"Simplified BFS path points: {len(simplified_path)}")
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
        # Contact visualization: 키보드 'C'와 'D'로 활성화 가능
        # C: Contact points 표시/숨김
        # D: Contact forces 토글

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
            # robot.data.qfrc_applied[robot.arm_qpos_adrs] = robot.data.qfrc_bias[
            #     robot.arm_qpos_adrs
            # ]
            mujoco.mj_step(robot.model, robot.data)
            collision_count += int(robot.data.ncon)
            
            # 충돌 분석
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
