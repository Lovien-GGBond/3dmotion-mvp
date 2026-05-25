"""
低空气象风险智能避险寻路模型 - MVP 环境模块
所有数据采用 numpy 在本地二维矩阵中虚构，不依赖外部地理数据和真实气象 API。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 测试网格参数
# ============================================================
GRID_SIZE = 100          # 100x100 网格
CELL_SIZE = 10.0         # 每格 10m
AREA_SIZE = 1000.0       # 1km x 1km

# ============================================================
# 2. 虚拟城市高程矩阵 (DEM)
# ============================================================

def build_dem() -> np.ndarray:
    """
    创建 100x100 的高程矩阵（单位：米）。
    硬编码若干矩形高楼区块，并设计狭窄街道以模拟狭管效应。
    """
    dem = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    # --- 高楼区块定义 (row_start, row_end, col_start, col_end, height_m) ---
    buildings = [
        # 左侧高楼群（带狭窄街道）
        (10, 30, 5, 25, 80),      # A: 大楼 A  80m
        (10, 30, 28, 48, 60),     # B: 大楼 B  60m
        # A 与 B 之间 col 25~28 为 3 格宽的狭窄街道（30m）

        # 右上角高楼
        (5, 20, 60, 80, 100),     # C: 超高层 100m
        (5, 20, 83, 95, 45),      # D: 中层楼 45m
        # C 与 D 之间 col 80~83 为 3 格窄街

        # 中部横向楼群
        (45, 60, 10, 35, 70),     # E: 大楼 E  70m
        (45, 60, 38, 55, 50),     # F: 大楼 F  50m
        # E 与 F 之间 col 35~38 为 3 格窄街

        # 底部大块建筑
        (70, 90, 20, 50, 90),     # G: 大楼 G  90m
        (70, 90, 53, 75, 55),     # H: 大楼 H  55m
        # G 与 H 之间 col 50~53 为 3 格窄街

        # 散布的小建筑
        (35, 42, 70, 80, 35),     # I
        (75, 85, 80, 95, 40),     # J
    ]

    for r0, r1, c0, c1, h in buildings:
        dem[r0:r1, c0:c1] = h

    return dem


# ============================================================
# 3. 基础气象常量（标量场）
# ============================================================

# 背景 10m 风速
V10 = 6.0          # m/s
WIND_DIR = np.pi / 4   # 风向角 θ = 45°（东北风，吹向西南）

# 风速分量（网格坐标：行增大=向下，列增大=向右）
# θ = π/4 → 风从东北来，吹向西南 → 行分量为正（向下），列分量为负（向左）
U_WIND = V10 * np.sin(WIND_DIR)    # 行方向分量（向下为正）
V_WIND = -V10 * np.cos(WIND_DIR)   # 列方向分量（向左为负）

# 全局气象标量
CLOUD_BASE = 500.0     # 云底高度 (m)
VISIBILITY = 1000.0    # 能见度 (m)
PRECIPITATION = 2.0    # 降水强度 (mm/h)


# ============================================================
# 4. 可视化：高程热力图
# ============================================================

def plot_dem(dem: np.ndarray, save_path: str = "dem_heatmap.png"):
    """
    绘制带有楼宇的虚拟高度热力图。
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 自定义 colormap：地面为浅色，建筑为深色
    cmap = plt.cm.terrain
    norm = mcolors.Normalize(vmin=0, vmax=120)

    im = ax.imshow(dem, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('高程 (m)', fontsize=12)

    # 标注狭窄街道位置
    streets = [
        (25, 28, 10, 30, '街道①'),
        (80, 83, 5, 20,   '街道②'),
        (35, 38, 45, 60,  '街道③'),
        (50, 53, 70, 90,  '街道④'),
    ]
    for c0, c1, r0, r1, label in streets:
        mid_r = (r0 + r1) / 2
        mid_c = (c0 + c1) / 2
        ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0],
                color='red', linewidth=1.5, linestyle='--')
        ax.text(mid_c, mid_r, label, color='red', fontsize=9,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

    # 标注风向箭头（风从东北吹向西南）
    ax.annotate('', xy=(90, 90), xytext=(90 - 12, 90 + 12),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=3))
    ax.text(80, 95, f'风向 θ={np.degrees(WIND_DIR):.0f}°\nV10={V10} m/s',
            color='cyan', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', fc='black', alpha=0.6))

    ax.set_xlabel('列 (每格 10m)', fontsize=12)
    ax.set_ylabel('行 (每格 10m)', fontsize=12)
    ax.set_title('MVP 虚拟城市高程热力图 (1km × 1km)', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'图片已保存至: {save_path}')
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    dem = build_dem()

    # 打印基本信息
    print(f'网格尺寸: {GRID_SIZE}x{GRID_SIZE}  (每格 {CELL_SIZE}m)')
    print(f'区域范围: {AREA_SIZE}m x {AREA_SIZE}m')
    print(f'最高建筑: {dem.max():.0f}m')
    print(f'风速 V10: {V10} m/s, 风向 θ: {np.degrees(WIND_DIR):.0f}°')
    print(f'风速分量: U={U_WIND:.2f} m/s, V={V_WIND:.2f} m/s')
    print(f'云底高度: {CLOUD_BASE}m, 能见度: {VISIBILITY}m, 降水: {PRECIPITATION}mm/h')

    plot_dem(dem)
