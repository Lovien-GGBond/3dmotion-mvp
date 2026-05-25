"""
低空气象风险智能避险寻路模型 - MVP 结果可视化
生成最终对比图并打印两条路线的完整数据。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from mvp_environment import build_dem, GRID_SIZE, V10, WIND_DIR
from mvp_risk_model import compute_wind_field, compute_risk, V80
from mvp_astar import astar, path_length, path_risk, START, END


def path_max_risk(path: list, risk_matrix: np.ndarray) -> float:
    """路径上单步遭遇的最大风险值（不含起点）。"""
    return max(risk_matrix[r, c] for r, c in path[1:])


def plot_final(dem: np.ndarray, risk_matrix: np.ndarray,
               path_a: list, path_b: list,
               save_path: str = "mvp_result.png"):
    """
    绘制最终对比图：
    - 底图：风险热力图 (YlOrRd)
    - 建筑：深灰色方块 + 黑色轮廓
    - 起终点：星标
    - 路线 A：黑色虚线
    - 路线 B：蓝色粗实线
    """
    risk_display = np.where(np.isinf(risk_matrix), np.nan, risk_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 底图：风险热力图 ---
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(risk_display, cmap=cmap, norm=norm,
                   origin='lower', interpolation='nearest')

    # --- 建筑物：深灰色填充 + 黑色轮廓 ---
    building_mask = dem > 0
    grey = np.zeros((*building_mask.shape, 4))
    grey[building_mask] = [0.2, 0.2, 0.2, 1.0]
    ax.imshow(grey, origin='lower', interpolation='nearest')
    ax.contour(dem, levels=[0.5], colors='black', linewidths=1.2, origin='lower')

    # --- 路线 A：黑色虚线 ---
    if path_a:
        ra, ca = zip(*path_a)
        ax.plot(ca, ra, color='black', linewidth=2.0, linestyle='--',
                label=f'Route A (lambda=0, {len(path_a)} steps)')

    # --- 路线 B：蓝色粗实线 ---
    if path_b:
        rb, cb = zip(*path_b)
        ax.plot(cb, rb, color='#1E90FF', linewidth=3.5, linestyle='-',
                label=f'Route B (lambda=5, {len(path_b)} steps)')

    # --- 起终点：星标 ---
    ax.plot(START[1], START[0], marker='*', color='#00FF00',
            markersize=18, markeredgecolor='black', markeredgewidth=1.0,
            label=f'Start S{START}', zorder=5)
    ax.plot(END[1], END[0], marker='*', color='#FF4500',
            markersize=18, markeredgecolor='black', markeredgewidth=1.0,
            label=f'End E{END}', zorder=5)
    ax.text(START[1] + 2, START[0] - 3, 'S', color='#00FF00',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
    ax.text(END[1] + 2, END[0] - 3, 'E', color='#FF4500',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    # --- 颜色条 ---
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Index I (0=safe, 1=dangerous)', fontsize=11)

    # --- 图例 ---
    legend_elements = [
        Patch(facecolor=[0.2, 0.2, 0.2], edgecolor='black', label='Buildings'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(legend_elements)
    ax.legend(handles=handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='black')

    ax.set_xlabel('Column (10m per cell)', fontsize=12)
    ax.set_ylabel('Row (10m per cell)', fontsize=12)
    ax.set_title('Urban Micro-scale Weather Aware Routing MVP', fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {save_path}')


def print_comparison(path_a: list, path_b: list, risk_matrix: np.ndarray):
    """打印两条路线的完整数据对比。"""
    len_a = path_length(path_a)
    len_b = path_length(path_b)
    cum_a = path_risk(path_a, risk_matrix)
    cum_b = path_risk(path_b, risk_matrix)
    max_a = path_max_risk(path_a, risk_matrix)
    max_b = path_max_risk(path_b, risk_matrix)

    print()
    print('=' * 62)
    print('  Route Comparison: Traditional vs Weather-Aware Routing')
    print('=' * 62)
    print(f'  {"Metric":<30} {"Route A":>12} {"Route B":>12}')
    print(f'  {"":─<30} {"":─>12} {"":─>12}')
    print(f'  {"lambda":<30} {"0":>12} {"5":>12}')
    print(f'  {"Steps":<30} {len(path_a):>12d} {len(path_b):>12d}')
    print(f'  {"Total distance (cells)":<30} {len_a:>12.1f} {len_b:>12.1f}')
    print(f'  {"Total distance (m)":<30} {len_a * 10:>12.0f} {len_b * 10:>12.0f}')
    print(f'  {"Cumulative risk":<30} {cum_a:>12.2f} {cum_b:>12.2f}')
    print(f'  {"Max single-step risk":<30} {max_a:>12.4f} {max_b:>12.4f}')
    print(f'  {"":─<30} {"":─>12} {"":─>12}')
    print(f'  {"Risk reduction":<30} {"":>12} {(1 - cum_b / cum_a) * 100:>11.1f}%')
    print(f'  {"Distance overhead":<30} {"":>12} {(len_b / len_a - 1) * 100:>11.1f}%')
    print('=' * 62)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    dem = build_dem()
    wind = compute_wind_field(dem)
    risk = compute_risk(wind, dem)

    print(f'V10={V10} m/s  V80={V80:.2f} m/s  Wind={np.degrees(WIND_DIR):.0f} deg')

    path_a = astar(risk, START, END, lam=0.0)
    path_b = astar(risk, START, END, lam=5.0)

    if path_a and path_b:
        print_comparison(path_a, path_b, risk)
        plot_final(dem, risk, path_a, path_b)
    else:
        print('Error: one or both routes failed.')
