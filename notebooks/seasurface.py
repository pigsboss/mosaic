import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import webbrowser
import os
import tempfile


def simulate_thermal_skin(
    nx=60, ny=60, lx=10.0, ly=10.0,
    nt=1000, dt=0.005,
    nu=0.1, alpha=0.05, beta_g=1.0,
    U_wind=1.5, T_cold=0.0,
    step_per_frame=50, interval=100,
    blit=False,
):
    # --- 1. 物理参数与网格设置 ---
    dx, dy = lx/nx, ly/ny

    # --- 2. 场变量初始化 ---
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    T = np.ones((ny, nx))

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

    # 初始图像
    im = ax.imshow(T, origin='lower', cmap='RdYlBu_r', extent=[0, lx, 0, ly],
                   vmin=0, vmax=1)
    # 初始 quiver
    q = ax.quiver(X[::3], Y[::3], u[::3, ::3], v[::3, ::3],
                  color='white', scale=10, animated=True)
    ax.set_title("Hey! Thermal Skin Layer (Step: 0)")

    # --- 5. 动画更新函数 ---
    num_frames = nt // step_per_frame

    def update(frame):
        nonlocal u, v, p, T
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

    # --- 6. 创建动画 ---
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=blit)
    return anim


if __name__ == "__main__":
    print("Yo! Let's model some thermal skin layers.")

    parser = argparse.ArgumentParser(description="2D thermal skin layer simulation")
    parser.add_argument('--nx', type=int, default=60)
    parser.add_argument('--ny', type=int, default=60)
    parser.add_argument('--lx', type=float, default=10.0)
    parser.add_argument('--ly', type=float, default=10.0)
    parser.add_argument('--nt', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--nu', type=float, default=0.1, help="Kinematic viscosity")
    parser.add_argument('--alpha', type=float, default=0.05, help="Thermal diffusivity")
    parser.add_argument('--beta_g', type=float, default=1.0, help="Buoyancy coefficient")
    parser.add_argument('--U_wind', type=float, default=1.5)
    parser.add_argument('--T_cold', type=float, default=0.0)
    parser.add_argument('--step_per_frame', type=int, default=50, help="Simulation steps per animation frame")
    parser.add_argument('--interval', type=int, default=100, help="Animation frame interval (ms)")
    parser.add_argument('--save', type=str, help="Save animation to HTML file instead of displaying")

    args = parser.parse_args()

    # 运行模拟
    anim = simulate_thermal_skin(
        nx=args.nx, ny=args.ny,
        lx=args.lx, ly=args.ly,
        nt=args.nt, dt=args.dt,
        nu=args.nu, alpha=args.alpha,
        beta_g=args.beta_g, U_wind=args.U_wind,
        T_cold=args.T_cold,
        step_per_frame=args.step_per_frame,
        interval=args.interval,
        blit=False,
    )

    # 显示动画
    if args.save:
        # 保存为 HTML 文件
        html_str = anim.to_jshtml()
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(html_str)
        print(f"Animation saved to {args.save}")
    else:
        try:
            # Jupyter / IPython 环境：直接显示
            from IPython import get_ipython
            if get_ipython():  # 在 IPython 中
                from IPython.display import display, HTML
                display(HTML(anim.to_jshtml()))
            else:
                raise ImportError
        except ImportError:
            # 普通 Python 环境：创建临时 HTML 并在浏览器中打开
            html_str = anim.to_jshtml()
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
                f.write(html_str)
                tmpname = f.name
            print("Opening animation in browser...")
            webbrowser.open('file://' + os.path.realpath(tmpname))
