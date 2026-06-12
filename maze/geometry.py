"""
Maze grid to world-frame geometry.
"""

import numpy as np

# 로봇 베이스 기준으로 x축 방향으로 0.34m 앞에 미로가 놓여 있음
MAZE_POS_WORLD = np.array([0.34, 0.0, 0.0], dtype=float) 
# 4cm x 4cm 셀 크기
CELL_SIZE = 0.04 
# 미로 바닥판 두께가 8mm
FLAT_TOP_Z = 0.008
FOLD_ANGLE = np.deg2rad(30.0)
FOLD_HINGE_ROW = 6
# r < 5.5 → rows 0~5 → flat
# r >= 5.5 → rows 6~11 → folded
FOLD_SWITCH_ROW = FOLD_HINGE_ROW - 0.5
# row 6 셀의 로컬 좌표
FOLD_R6_Y_LOCAL = 0.0073
FOLD_R6_Z_LOCAL = 0.0353
WALL_HALF_HEIGHT = 0.020

# 미로 CSV의 셀 좌표(row, col) → 3D 월드 좌표 변환
def maze_to_world(p_local: np.ndarray) -> np.ndarray:
    return np.asarray(p_local, dtype=float).reshape(3) + MAZE_POS_WORLD

# 셀의 x 좌표를 계산하는 함수 / 0번 col의 x좌표가 -0.22
def _cell_x(c: float) -> float:
    return -0.22 + CELL_SIZE * float(c)

# flat 구간에서 셀의 y 좌표를 계산하는 함수 / 0번 row의 y좌표가 -0.22
def _cell_y(r: float) -> float:
    return -0.22 + CELL_SIZE * float(r)

# 평평한 구간(rows 0~5)의 셀 좌표를 계산하는 함수
def _flat_cell(r: float, c: float, hover: float):
    """
    x = _cell_x(c)              → 열 방향 위치
    y = -0.22 + 0.04 × r        → 행 방향 위치 (x랑 같은 방식)
    z = FLAT_TOP_Z = 0.008      → 항상 바닥 표면 높이
    """
    n = np.array([0.0, 0.0, 1.0])
    p_local = np.array(
        [_cell_x(c), _cell_y(r), FLAT_TOP_Z],
        dtype=float,
    )
    # n : z축 방향의 법선 벡터, hover : 벽 위로 살짝 띄우는 높이, p_local + hover * n : 벽 위로 띄운 위치
    return maze_to_world(p_local + hover * n), n

# 접힌 구간(rows 6~11)의 셀 좌표를 계산하는 함수
def _folded_cell(r: float, c: float, hover: float):
    n = np.array([0.0, -np.sin(FOLD_ANGLE), np.cos(FOLD_ANGLE)])
    row_offset = float(r) - FOLD_HINGE_ROW
    cube_center = np.array(
        [
            _cell_x(c),
            FOLD_R6_Y_LOCAL + row_offset * CELL_SIZE * np.cos(FOLD_ANGLE),
            FOLD_R6_Z_LOCAL + row_offset * CELL_SIZE * np.sin(FOLD_ANGLE),
        ],
        dtype=float,
    )
    floor_point = cube_center - WALL_HALF_HEIGHT * n
    return maze_to_world(floor_point + hover * n), n

# 셀 좌표(row, col)와 hover 높이를 받아서 월드 좌표 반환하는 함수
def cell_to_world(r: int, c: int, hover: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    if float(r) < FOLD_SWITCH_ROW:
        return _flat_cell(float(r), float(c), hover)
    return _folded_cell(float(r), float(c), hover)
