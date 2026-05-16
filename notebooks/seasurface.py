import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML

# Casual greeting
print("Yo! Let's model some thermal skin layers.")

# --- 1. 物理参数与网格设置 ---
nx, ny = 60, 60
lx, ly = 10.0, 10.0
dx, dy = lx/nx, ly/ny
nt = 1000
dt = 0.005
nu = 0.1
alpha = 0.05
beta_g = 1.0

# --- 2. 场变量初始化 ---
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
T = np.ones((ny, nx))

T_cold = 0.0
U_wind = 1.5

# --- 3. 辅助函数 ---
def laplacian(f, dx, dy):
    return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1))/dx**2 + \
           (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0))/dy**2

def advection(f, u, v, dx, dy):
    df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)
    df_dy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dy)
    return u * df_dx + v * df_dy

# --- 4. 创建图形和初始艺术家对象 ---
fig, ax = plt.subplots(figsize=(8, 6))
X = np.linspace(0, lx, nx)
Y = np.linspace(0, ly, ny)
sub = slice(None, None, 3)   # 每 3 个点取一个箭头

# 初始图像
im = ax.imshow(T, origin='lower', cmap='RdYlBu_r', extent=[0, lx, 0, ly],
               animated=True)
# 初始 quiver
q = ax.quiver(X[::3], Y[::3], u[::3, ::3], v[::3, ::3],
              color='white', scale=10, animated=True)
ax.set_title("Hey! Thermal Skin Layer (Step: 0)")

# --- 5. 动画更新函数 ---
step_per_frame = 50
num_frames = nt // step_per_frame

def update(frame):
    global u, v, p, T
    for _ in range(step_per_frame):
        # 强制边界条件
        u[-1, :] = U_wind
        u[0, :] = 0.0
        v[-1, :] = 0.0; v[0, :] = 0.0
        T[-1, :] = T_cold
        T[0, :] = 1.0

        # 预测步
        u_star = u - dt * advection(u, u, v, dx, dy) + dt * nu * laplacian(u, dx, dy)
        v_star = v - dt * advection(v, u, v, dx, dy) + dt * nu * laplacian(v, dx, dy) + dt * beta_g * T

        u_star[-1, :] = U_wind; u_star[0, :] = 0.0
        v_star[-1, :] = 0.0; v_star[0, :] = 0.0

        # 散度
        div_star = (np.roll(u_star, -1, axis=1) - np.roll(u_star, 1, axis=1))/(2*dx) + \
                   (np.roll(v_star, -1, axis=0) - np.roll(v_star, 1, axis=0))/(2*dy)

        # 压力迭代
        for _ in range(30):
            p = 0.25 * (np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1) +
                        np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0) -
                        (dx**2) * div_star / dt)
            p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]; p[-1, :] = p[-2, :]

        # 校正步
        u = u_star - dt * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/(2*dx)
        v = v_star - dt * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/(2*dy)

        # 温度场
        T = T - dt * advection(T, u, v, dx, dy) + dt * alpha * laplacian(T, dx, dy)

    # 更新图像和 quiver 数据
    im.set_array(T)
    q.set_UVC(u[::3, ::3], v[::3, ::3])
    ax.set_title(f"Hey! Thermal Skin Layer (Step: {(frame+1)*step_per_frame})")
    return im, q

# --- 6. 创建动画并显示 ---
anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
display(HTML(anim.to_jshtml()))
# 若不再需要静态图可在此关闭： plt.close(fig)
