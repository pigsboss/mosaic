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
    U_wind=1.5,
    step_per_frame=50, interval=100,
    blit=False,
    u_stokes_surf=0.0,
    stokes_decay=1.0,
    mix_enhancement=1.0,
    mix_layer_depth=0.5,
    surface_heat_transfer_coeff=0.5,
    surface_ref_temp=0.0,
    return_history=False,
    show_surface_displacement=False,
    surf_disp_scale=0.5,
):
    # --- 1. 物理参数与网格设置 ---
    dx, dy = lx/nx, ly/ny

    # --- 2. 场变量初始化 ---
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    T = np.ones((ny, nx))

    # --- 波浪物理预处理 ---
    y = np.linspace(0, ly, ny)          # 垂直坐标，y=0底部，y=ly顶部
    d = ly - y                           # 距自由表面距离
    u_stokes = u_stokes_surf * np.exp(-d / stokes_decay)  # Stokes 速度剖面 (ny,)

    # 波浪增强混合剖面（指数衰减）
    mix_factor_profile = 1.0 + (mix_enhancement - 1.0) * np.exp(-d / mix_layer_depth)
    # 转换为完整的二维数组 (ny, nx) 以供 diffusion_c 使用
    nu_eff_2d = nu * mix_factor_profile[:, None] * np.ones((1, nx))
    alpha_eff_2d = alpha * mix_factor_profile[:, None] * np.ones((1, nx))

    # --- 3. 辅助函数 ---
    def laplacian(f, dx, dy):
        return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1))/dx**2 + \
               (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0))/dy**2

    def advection(f, u, v, dx, dy):
        df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)
        df_dy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dy)
        return u * df_dx + v * df_dy

    def diffusion_c(f, kappa, dx, dy):
        """Variable-coefficient diffusion term: div(kappa grad f) using arithmetic means."""
        # 水平方向的面系数
        if kappa.shape[1] > 1:
            kx_e = 0.5 * (kappa[:, :-1] + kappa[:, 1:])
        else:
            kx_e = np.zeros_like(kappa)
        kx_w = kx_e
        # 垂直方向的面系数
        if kappa.shape[0] > 1:
            ky_n = 0.5 * (kappa[1:, :] + kappa[:-1, :])
        else:
            ky_n = np.zeros_like(kappa)
        ky_s = ky_n

        diff_x = np.zeros_like(f)
        diff_y = np.zeros_like(f)

        # 内部节点
        diff_x[:, 1:-1] = ((kx_e[:, 1:] * (f[:, 2:] - f[:, 1:-1]) / dx) -
                           (kx_w[:, :-1] * (f[:, 1:-1] - f[:, :-2]) / dx)) / dx
        diff_y[1:-1, :] = ((ky_n[1:, :] * (f[2:, :] - f[1:-1, :]) / dy) -
                           (ky_s[:-1, :] * (f[1:-1, :] - f[:-2, :]) / dy)) / dy
        # 边界点留空，将被边界条件覆盖
        return diff_x + diff_y

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
    ax.set_title(f"Us={u_stokes_surf:.1f}, mix_enh={mix_enhancement:.1f}, mix_d={mix_layer_depth:.2f} Step 0")

    # 海气界面形变（基于顶部温度）
    if show_surface_displacement:
        x_disp = np.linspace(0, lx, nx)
        eta0 = np.zeros(nx)
        surf_line, = ax.plot(x_disp, ly + eta0, 'c-', linewidth=1.5)
        surf_fill = [ax.fill_between(x_disp, ly + eta0, ly + 2.0,
                                     color='cyan', alpha=0.2)]
    else:
        surf_line = None
        surf_fill = None

    # --- 5. 动画更新函数 ---
    num_frames = nt // step_per_frame

    if return_history:
        history = []
    else:
        history = None

    def update(frame):
        nonlocal u, v, p, T
        for _ in range(step_per_frame):
            # 强制边界条件 (速度)
            u[-1, :] = U_wind
            u[0, :] = 0.0
            v[-1, :] = 0.0; v[0, :] = 0.0
            # 温度边界：底部固定暖水源，顶部使用半隐式牛顿冷却（指数松弛）
            T[0, :] = 1.0
            alpha_c = surface_heat_transfer_coeff
            T[-1, :] = (T[-1, :] + dt * alpha_c * surface_ref_temp) / (1.0 + dt * alpha_c)

            # 预测步（使用变系数扩散 nu_eff_2d）
            u_star = u - dt * advection(u, u, v, dx, dy) + dt * diffusion_c(u, nu_eff_2d, dx, dy)
            v_star = v - dt * advection(v, u, v, dx, dy) + dt * diffusion_c(v, nu_eff_2d, dx, dy) + dt * beta_g * T

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

            # 温度场 (平流项中加入 Stokes 漂移，扩散使用 alpha_eff_2d)
            T_adv = advection(T, u + u_stokes[None, :], v, dx, dy)
            T = T - dt * T_adv + dt * diffusion_c(T, alpha_eff_2d, dx, dy)

        # 更新图像和 quiver 数据
        im.set_array(T)
        if return_history:
            history.append(T.copy())
        q.set_UVC(u[::3, ::3], v[::3, ::3])

        # 更新海气界面形变
        if show_surface_displacement:
            T_top = T[-1, :]                     # 顶部温度 (表层)
            eta = surf_disp_scale * (T_top - T_top.mean())
            surf_line.set_ydata(ly + eta)
            surf_fill[0].remove()
            surf_fill[0] = ax.fill_between(x_disp, ly + eta, ly + 2.0,
                                           color='cyan', alpha=0.2)

        ax.set_title(f"Us={u_stokes_surf:.1f}, mix_enh={mix_enhancement:.1f}, mix_d={mix_layer_depth:.2f} Step{(frame+1)*step_per_frame}")
        return im, q

    # --- 6. 创建动画 ---
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=blit)
    if return_history:
        return anim, history
    else:
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
    parser.add_argument('--step_per_frame', type=int, default=50, help="Simulation steps per animation frame")
    parser.add_argument('--interval', type=int, default=100, help="Animation frame interval (ms)")
    parser.add_argument('--save', type=str, help="Save animation to HTML file instead of displaying")
    # 波浪效应参数
    parser.add_argument('--u_stokes_surf', type=float, default=0.0,
                        help="Surface Stokes drift velocity [m/s]")
    parser.add_argument('--stokes_decay', type=float, default=1.0,
                        help="e-folding depth for Stokes drift [m]")
    parser.add_argument('--mix_enhancement', type=float, default=1.0,
                        help='Wave mixing enhancement factor at surface')
    parser.add_argument('--mix_layer_depth', type=float, default=0.5,
                        help='e-folding depth of wave-enhanced mixing')
    # 海面热交换参数
    parser.add_argument('--surface_heat_transfer_coeff', type=float, default=0.5,
                        help='Surface heat transfer coefficient λ (0 = insulated top)')
    parser.add_argument('--surface_ref_temp', type=float, default=0.0,
                        help='Reference atmospheric temperature for surface cooling')

    parser.add_argument('--show_surface_displacement', action='store_true',
                        help='Overlay surface deformation from temperature fluctuations')
    parser.add_argument('--surf_disp_scale', type=float, default=0.5,
                        help='Scaling factor for thermal expansion of the surface')

    args = parser.parse_args()

    # 运行模拟
    anim = simulate_thermal_skin(
        nx=args.nx, ny=args.ny,
        lx=args.lx, ly=args.ly,
        nt=args.nt, dt=args.dt,
        nu=args.nu, alpha=args.alpha,
        beta_g=args.beta_g, U_wind=args.U_wind,
        step_per_frame=args.step_per_frame,
        interval=args.interval,
        blit=False,
        u_stokes_surf=args.u_stokes_surf,
        stokes_decay=args.stokes_decay,
        mix_enhancement=args.mix_enhancement,
        mix_layer_depth=args.mix_layer_depth,
        surface_heat_transfer_coeff=args.surface_heat_transfer_coeff,
        surface_ref_temp=args.surface_ref_temp,
        show_surface_displacement=args.show_surface_displacement,
        surf_disp_scale=args.surf_disp_scale,
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
