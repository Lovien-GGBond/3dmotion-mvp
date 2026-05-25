"""
低空气象风险智能避险寻路模型 - MVP 风险评估模块
导入 mvp_environment 的高程矩阵，计算微尺度风场和综合风险指数。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from mvp_environment import (
    GRID_SIZE, CELL_SIZE,
    build_dem, V10, WIND_DIR, CLOUD_BASE, VISIBILITY, PRECIPITATION,
)

# ============================================================
# 任务 1：AHP 权重预设（跳过检验）
# ============================================================
# [风速, 云底高度, 能见度, 降水]
W_AHP = np.array([0.5, 0.1, 0.2, 0.2])

# ============================================================
# 任务 2：微尺度风场降尺度推演
# ============================================================

# 垂直修正：V10 → V80（幂律风廓线，α=0.3）
FLIGHT_ALT = 80.0
V80 = V10 * (FLIGHT_ALT / 10.0) ** 0.3

# 网格坐标风向矢量（行增大=向下，列增大=向右）
# θ=45° 东北风 → 吹向西南 → (+行, -列)
U_ROW = V80 * np.sin(WIND_DIR)   # 行分量 > 0（向下）
V_COL = -V80 * np.cos(WIND_DIR)  # 列分量 < 0（向左）

# 尾流 / 狭管系数
K_w = 0.6   # 尾流衰减
K_c = 1.5   # 狭管加速


def compute_wind_field(dem: np.ndarray) -> np.ndarray:
    """
    计算 100x100 飞行高度风速矩阵。
    规则：
      - 建筑内部 → 0
      - 建筑下风向 1~5 格（尾流区）→ V80 * K_w
      - 两栋建筑间空隙且与风向夹角 <45°（狭管区）→ V80 * K_c
    """
    is_building = dem > 0
    wind = np.where(is_building, 0.0, V80)

    # --- 尾流衰减 ---
    norm = np.sqrt(U_ROW**2 + V_COL**2)
    du, dv = U_ROW / norm, V_COL / norm  # 单位方向向量

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not is_building[r, c]:
                continue
            for step in range(1, 6):  # 下风向 1~5 格
                rr = int(round(r + du * step))
                cc = int(round(c + dv * step))
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    if not is_building[rr, cc]:
                        wind[rr, cc] = V80 * K_w
                else:
                    break

    # --- 狭管加速 ---
    def check_canyon(r0, r1, c0, c1):
        """检查两栋建筑之间的空隙是否因风向产生狭管效应。
        检测风在空隙窄方向上的分量是否足够大（夹角 ≤ 45°）。
        """
        if r0 >= r1 or c0 >= c1:
            return
        gap_row_span = r1 - r0
        gap_col_span = c1 - c0
        wind_vec = np.array([U_ROW, V_COL])
        wind_mag = np.linalg.norm(wind_vec)

        cos_angle = 0.0
        if gap_row_span > gap_col_span:
            # 竖向空隙（窄方向 = 列方向）
            cos_angle = abs(V_COL) / wind_mag
        else:
            # 横向空隙（窄方向 = 行方向）
            cos_angle = abs(U_ROW) / wind_mag

        if cos_angle >= np.cos(np.radians(45)) - 1e-9:  # 夹角 ≤ 45°（含浮点容差）
            for r in range(max(0, r0), min(GRID_SIZE, r1)):
                for c in range(max(0, c0), min(GRID_SIZE, c1)):
                    if not is_building[r, c]:
                        wind[r, c] = V80 * K_c

    buildings = [
        (10, 30, 5, 25), (10, 30, 28, 48),   # A, B
        (5, 20, 60, 80), (5, 20, 83, 95),     # C, D
        (45, 60, 10, 35), (45, 60, 38, 55),   # E, F
        (70, 90, 20, 50), (70, 90, 53, 75),   # G, H
    ]
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            r0a, r1a, c0a, c1a = buildings[i]
            r0b, r1b, c0b, c1b = buildings[j]
            # 行重叠 → 列方向空隙
            if r0a < r1b and r0b < r1a:
                lo = max(r0a, r0b)
                hi = min(r1a, r1b)
                check_canyon(lo, hi, c1a, c0b)
                check_canyon(lo, hi, c1b, c0a)
            # 列重叠 → 行方向空隙
            if c0a < c1b and c0b < c1a:
                lo = max(c0a, c0b)
                hi = min(c1a, c1b)
                check_canyon(r1a, r0b, lo, hi)
                check_canyon(r1b, r0a, lo, hi)

    return wind


# ============================================================
# 任务 3：综合风险指数
# ============================================================

def compute_risk(wind: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """
    综合风险指数 I ∈ [0, 1]，建筑内部为 np.inf。
    I = w1*F_wind + w2*F_cloud + w3*F_vis + w4*F_precip
    """
    # 风速风险：线性归一化 [8, 12] m/s → [0, 1]
    f_wind = np.clip((wind - 8.0) / (12.0 - 8.0), 0, 1)

    # 气象常量风险（标量 → 矩阵广播）
    f_cloud = np.clip((200.0 - CLOUD_BASE) / (200.0 - 1000.0), 0, 1)   # 低云底高风险
    f_vis   = np.clip((1000.0 - VISIBILITY) / (1000.0 - 5000.0), 0, 1)  # 低能见度高风险
    f_precip = np.clip(PRECIPITATION / 10.0, 0, 1)                       # 强降水高风险

    risk = (W_AHP[0] * f_wind +
            W_AHP[1] * f_cloud +
            W_AHP[2] * f_vis +
            W_AHP[3] * f_precip)

    # 建筑内部标为无穷大
    risk[dem > 0] = np.inf

    return risk


# ============================================================
# 可视化
# ============================================================

def plot_wind_field(wind: np.ndarray, dem: np.ndarray, save_path: str = "wind_field.png"):
    """绘制风速热力图，叠加建筑轮廓。"""
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=V80 * K_c + 2)
    im = ax.imshow(wind, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('风速 (m/s)', fontsize=12)

    # 建筑轮廓
    ax.contour(dem, levels=[0.5], colors='black', linewidths=1.5, origin='lower')

    ax.set_xlabel('列 (每格 10m)', fontsize=12)
    ax.set_ylabel('行 (每格 10m)', fontsize=12)
    ax.set_title(f'飞行高度 {FLIGHT_ALT:.0f}m 风速场 (NE {V10}m/s)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'风速热力图已保存至: {save_path}')


def plot_risk(risk: np.ndarray, dem: np.ndarray, save_path: str = "risk_heatmap.png"):
    """绘制风险指数热力图 (0~1)，建筑区域用灰色表示。"""
    risk_display = np.where(np.isinf(risk), np.nan, risk)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.RdYlGn_r  # 红=高风险, 绿=低风险
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(risk_display, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('风险指数 I (0=安全, 1=危险)', fontsize=12)

    # 建筑区域用灰色覆盖
    building_mask = dem > 0
    grey = np.zeros((*building_mask.shape, 4))
    grey[building_mask] = [0.3, 0.3, 0.3, 1.0]
    ax.imshow(grey, origin='lower', interpolation='nearest')

    ax.contour(dem, levels=[0.5], colors='black', linewidths=1.5, origin='lower')

    ax.set_xlabel('列 (每格 10m)', fontsize=12)
    ax.set_ylabel('行 (每格 10m)', fontsize=12)
    ax.set_title('综合气象风险指数 (AHP 权重)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'风险热力图已保存至: {save_path}')


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    dem = build_dem()
    wind = compute_wind_field(dem)
    risk = compute_risk(wind, dem)

    # 诊断信息
    print(f'V10={V10} m/s → V80={V80:.2f} m/s')
    print(f'风向分量: U_row={U_ROW:.2f}, V_col={V_COL:.2f}')
    print(f'尾流系数 K_w={K_w}, 狭管系数 K_c={K_c}')
    print(f'AHP 权重 W={W_AHP}')
    print(f'风速范围 (非建筑): {wind[dem == 0].min():.2f} ~ {wind[dem == 0].max():.2f} m/s')
    print(f'风险范围 (非建筑): {risk[dem == 0].min():.4f} ~ {risk[dem == 0].max():.4f}')
    print(f'建筑格数: {(dem > 0).sum()}, 尾流格数: {(wind[dem == 0] < V80 * 0.9).sum()}')
    print(f'狭管格数: {(wind[dem == 0] > V80 * 1.1).sum()}')

    plot_wind_field(wind, dem)
    plot_risk(risk, dem)
