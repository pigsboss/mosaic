import numpy as np
import matplotlib.pyplot as plt

# --- 1. 物理参数与网格设置 ---
nx, ny = 60, 60          # 低分辨率网格
lx, ly = 10.0, 10.0      # 物理尺寸 (10米 x 10米)
dx, dy = lx/nx, ly/ny
nt = 1000                # 时间步数
dt = 0.005               # 时间步长
nu = 0.1                 # 运动粘度 (为保持低网格下的稳定，设得偏大)
alpha = 0.05             # 热扩散率
beta_g = 1.0             # 浮力系数 (g * beta)

# --- 2. 场变量初始化 ---
u = np.zeros((ny, nx))   # 水平速度
v = np.zeros((ny, nx))   # 垂直速度
p = np.zeros((ny, nx))   # 压力场
T = np.ones((ny, nx))    # 温度场 (初始全部为暖水 1.0)

# 边界条件常量
T_cold = 0.0             # 顶部冷皮层温度
U_wind = 1.5             # 顶部风剪切速度

# --- 3. 辅助函数：中心差分 ---
def laplacian(f, dx, dy):
    return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1))/dx**2 + \
           (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0))/dy**2

def advection(f, u, v, dx, dy):
    # 为保证极简，这里用中心差分(低雷诺数下可用，高雷诺数需迎风格式)
    df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)
    df_dy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dy)
    return u * df_dx + v * df_dy

# --- 4. 主求解循环 ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for n in range(nt):
    # A. 强制边界条件
    u[-1, :] = U_wind   # 顶部风吹
    u[0, :] = 0.0       # 底部无滑移
    v[-1, :] = 0.0; v[0, :] = 0.0
    T[-1, :] = T_cold   # 顶部维持冷皮层
    T[0, :] = 1.0       # 底部维持暖水
    
    # B. 预测步：计算中间速度 (包含对流、扩散、浮力)
    u_star = u - dt * advection(u, u, v, dx, dy) + dt * nu * laplacian(u, dx, dy)
    v_star = v - dt * advection(v, u, v, dx, dy) + dt * nu * laplacian(v, dx, dy) + dt * beta_g * T
    
    # 维持预测步边界
    u_star[-1, :] = U_wind; u_star[0, :] = 0.0
    v_star[-1, :] = 0.0; v_star[0, :] = 0.0
    
    # C. 泊松方程求压力 (雅可比迭代，简化处理)
    div_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1))/(2*dx) + \
               (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0))/(2*dy)
    
    for _ in range(30): # 迭代求解 p
        p = 0.25 * (np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1) + \
                    np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0) - \
                    (dx**2) * div_star / dt)
        # 压力边界(诺依曼边界)
        p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
        
    # D. 校正步：更新无散速度场
    u = u_star - dt * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/(2*dx)
    v = v_star - dt * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/(2*dy)
    
    # E. 温度场对流-扩散
    T = T - dt * advection(T, u, v, dx, dy) + dt * alpha * laplacian(T, dx, dy)
    
    # --- 5. 实时可视化 ---
    if n % 50 == 0:
        ax.clear()
        im = ax.imshow(T, origin='lower', cmap='RdYlBu_r', extent=[0, lx, 0, ly])
        ax.quiver(np.linspace(0, lx, nx)[::3], np.linspace(0, ly, ny)[::3], 
                  u[::3, ::3], v[::3, ::3], color='white', scale=10)
        ax.set_title(f"Thermal Skin Layer Disruption (Step: {n})")
        plt.pause(0.01)

plt.ioff()
plt.show()