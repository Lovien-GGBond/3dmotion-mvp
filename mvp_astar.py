"""
低空气象风险智能避险寻路模型 - MVP A* 寻路模块
支持 8 方向移动，代价函数融合气象风险指数。
"""

import heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from mvp_environment import build_dem, GRID_SIZE
from mvp_risk_model import compute_wind_field, compute_risk

# 起终点
START = (3, 3)       # 左上角空地
END = (90, 90)       # 右下角空地

# 8 方向邻居偏移 (dr, dc) 及对应欧氏距离
DIRECTIONS = [
    (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
    ( 0, -1, 1.0),                      ( 0, 1, 1.0),
    ( 1, -1, np.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, np.sqrt(2)),
]


def heuristic(r1, c1, r2, c2) -> float:
    """欧氏距离启发式。"""
    return np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


def astar(risk_matrix: np.ndarray, start: tuple, end: tuple,
          lam: float = 0.0) -> list:
    """
    A* 寻路算法。
    代价 G = 欧氏移动距离 + lam * risk_matrix[r, c]
    返回路径坐标列表 [(r, c), ...]，无解返回空列表。
    """
    sr, sc = start
    er, ec = end

    # 起点/终点合法性检查
    if np.isinf(risk_matrix[sr, sc]) or np.isinf(risk_matrix[er, ec]):
        print(f'错误：起点 {start} 或终点 {end} 位于建筑内部！')
        return []

    open_set = []
    heapq.heappush(open_set, (0.0, sr, sc))
    came_from = {}
    g_score = np.full((GRID_SIZE, GRID_SIZE), np.inf)
    g_score[sr, sc] = 0.0

    while open_set:
        f_cur, r, c = heapq.heappop(open_set)

        if (r, c) == (er, ec):
            # 回溯路径
            path = []
            while (r, c) in came_from:
                path.append((r, c))
                r, c = came_from[(r, c)]
            path.append((sr, sc))
            path.reverse()
            return path

        # 已找到更优路径则跳过
        if f_cur > g_score[r, c] + heuristic(r, c, er, ec) + 1e-9:
            continue

        for dr, dc, dist in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if np.isinf(risk_matrix[nr, nc]):
                continue  # 建筑不可通行

            move_cost = dist + lam * risk_matrix[nr, nc]
            new_g = g_score[r, c] + move_cost

            if new_g < g_score[nr, nc]:
                g_score[nr, nc] = new_g
                came_from[(nr, nc)] = (r, c)
                f = new_g + heuristic(nr, nc, er, ec)
                heapq.heappush(open_set, (f, nr, nc))

    print(f'警告：从 {start} 到 {end} 无可达路径！')
    return []


def path_risk(path: list, risk_matrix: np.ndarray) -> float:
    """计算路径上各格风险值的累加和（不含起点）。"""
    return sum(risk_matrix[r, c] for r, c in path[1:])


def path_length(path: list) -> float:
    """计算路径总欧氏距离（格为单位）。"""
    length = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        length += np.sqrt(dr ** 2 + dc ** 2)
    return length


def plot_paths(dem: np.ndarray, risk_matrix: np.ndarray,
               path_a: list, path_b: list,
               save_path: str = "astar_comparison.png"):
    """绘制两条路径对比图，底图为风险热力图。"""
    risk_display = np.where(np.isinf(risk_matrix), np.nan, risk_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(risk_display, cmap=cmap, norm=norm,
                   origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('风险指数 I', fontsize=12)

    # 建筑灰色覆盖
    building_mask = dem > 0
    grey = np.zeros((*building_mask.shape, 4))
    grey[building_mask] = [0.3, 0.3, 0.3, 1.0]
    ax.imshow(grey, origin='lower', interpolation='nearest')
    ax.contour(dem, levels=[0.5], colors='black', linewidths=1.0, origin='lower')

    # 路线 A
    if path_a:
        ra, ca = zip(*path_a)
        ax.plot(ca, ra, color='blue', linewidth=2.5, linestyle='--',
                label=f'路线A (lambda=0, {len(path_a)}步)')
    # 路线 B
    if path_b:
        rb, cb = zip(*path_b)
        ax.plot(cb, rb, color='magenta', linewidth=2.5, linestyle='-',
                label=f'路线B (lambda=5, {len(path_b)}步)')

    # 起终点
    ax.plot(START[1], START[0], 'go', markersize=12, label=f'起点 S{START}')
    ax.plot(END[1], END[0], 'r^', markersize=12, label=f'终点 E{END}')
    ax.text(START[1] + 2, START[0] - 2, 'S', color='green', fontsize=10, fontweight='bold')
    ax.text(END[1] + 2, END[0] - 2, 'E', color='red', fontsize=10, fontweight='bold')

    ax.set_xlabel('列 (每格 10m)', fontsize=12)
    ax.set_ylabel('行 (每格 10m)', fontsize=12)
    ax.set_title('A* 气象避险寻路对比', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'对比图已保存至: {save_path}')


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    # 构建环境与风险矩阵
    dem = build_dem()
    wind = compute_wind_field(dem)
    risk = compute_risk(wind, dem)

    # --- 路线 A：传统最短路径（lambda=0） ---
    path_a = astar(risk, START, END, lam=0.0)
    if path_a:
        print(f'[路线 A] lambda=0  步数={len(path_a)}  '
              f'距离={path_length(path_a):.1f}格  '
              f'累计风险={path_risk(path_a, risk):.2f}')

    # --- 路线 B：气象感知避险（lambda=5.0） ---
    path_b = astar(risk, START, END, lam=5.0)
    if path_b:
        print(f'[路线 B] lambda=5  步数={len(path_b)}  '
              f'距离={path_length(path_b):.1f}格  '
              f'累计风险={path_risk(path_b, risk):.2f}')

    # 对比可视化
    plot_paths(dem, risk, path_a, path_b)
