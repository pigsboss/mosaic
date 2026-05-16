import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import webbrowser
import os
import tempfile
from numba import njit, prange

# ----------------------------------------------------------------------
# Numba‑compatible roll functions (2D and 3D versions)
# ----------------------------------------------------------------------

@njit
def roll_2d_0(f, shift):
    """roll along axis=0 for 2D arrays"""
    ny, nx = f.shape
    res = np.empty_like(f)
    if shift == -1:
        res[0, :] = f[-1, :]
        res[1:, :] = f[:-1, :]
    elif shift == 1:
        res[-1, :] = f[0, :]
        res[:-1, :] = f[1:, :]
    else:
        res = f
    return res

@njit
def roll_2d_1(f, shift):
    """roll along axis=1 for 2D arrays"""
    ny, nx = f.shape
    res = np.empty_like(f)
    if shift == -1:
        res[:, 0] = f[:, -1]
        res[:, 1:] = f[:, :-1]
    elif shift == 1:
        res[:, -1] = f[:, 0]
        res[:, :-1] = f[:, 1:]
    else:
        res = f
    return res

@njit
def roll_3d_0(f, shift):
    """roll along axis=0 for 3D arrays"""
    ny, nx, nz = f.shape
    res = np.empty_like(f)
    if shift == -1:
        res[0, :, :] = f[-1, :, :]
        res[1:, :, :] = f[:-1, :, :]
    elif shift == 1:
        res[-1, :, :] = f[0, :, :]
        res[:-1, :, :] = f[1:, :, :]
    else:
        res = f
    return res

@njit
def roll_3d_1(f, shift):
    """roll along axis=1 for 3D arrays"""
    ny, nx, nz = f.shape
    res = np.empty_like(f)
    if shift == -1:
        res[:, 0, :] = f[:, -1, :]
        res[:, 1:, :] = f[:, :-1, :]
    elif shift == 1:
        res[:, -1, :] = f[:, 0, :]
        res[:, :-1, :] = f[:, 1:, :]
    else:
        res = f
    return res

@njit
def roll_3d_2(f, shift):
    """roll along axis=2 for 3D arrays"""
    ny, nx, nz = f.shape
    res = np.empty_like(f)
    if shift == -1:
        res[:, :, 0] = f[:, :, -1]
        res[:, :, 1:] = f[:, :, :-1]
    elif shift == 1:
        res[:, :, -1] = f[:, :, 0]
        res[:, :, :-1] = f[:, :, 1:]
    else:
        res = f
    return res


# ----------------------------------------------------------------------
# Numba‑accelerated helper functions (JIT compiled)
# ----------------------------------------------------------------------

@njit
def advection_2d(f, u, v, dx, dy):
    df_dx = (roll_2d_1(f, -1) - roll_2d_1(f, 1)) / (2.0 * dx)
    df_dy = (roll_2d_0(f, -1) - roll_2d_0(f, 1)) / (2.0 * dy)
    return u * df_dx + v * df_dy

@njit
def diffusion_2d(f, kappa, dx, dy):
    """Variable‑coefficient diffusion (interior points only)."""
    ny, nx = f.shape
    out = np.zeros_like(f)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            kx_e = 0.5 * (kappa[i, j] + kappa[i, j+1])
            kx_w = 0.5 * (kappa[i, j] + kappa[i, j-1])
            ky_n = 0.5 * (kappa[i, j] + kappa[i+1, j])
            ky_s = 0.5 * (kappa[i, j] + kappa[i-1, j])
            fx = (kx_e*(f[i,j+1]-f[i,j]) - kx_w*(f[i,j]-f[i,j-1])) / dx**2
            fy = (ky_n*(f[i+1,j]-f[i,j]) - ky_s*(f[i,j]-f[i-1,j])) / dy**2
            out[i,j] = fx + fy
    return out

@njit
def advection_3d(f, u, v, w, dx, dy, dz):
    df_dx = (roll_3d_1(f, -1) - roll_3d_1(f, 1)) / (2.0*dx)
    df_dy = (roll_3d_0(f, -1) - roll_3d_0(f, 1)) / (2.0*dy)
    df_dz = (roll_3d_2(f, -1) - roll_3d_2(f, 1)) / (2.0*dz)
    return u*df_dx + v*df_dy + w*df_dz

@njit
def diffusion_3d(f, kappa, dx, dy, dz):
    ny, nx, nz = f.shape
    out = np.zeros_like(f)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            for k in range(1, nz-1):
                kx_e = 0.5*(kappa[i,j,k] + kappa[i,j+1,k])
                kx_w = 0.5*(kappa[i,j,k] + kappa[i,j-1,k])
                ky_n = 0.5*(kappa[i,j,k] + kappa[i+1,j,k])
                ky_s = 0.5*(kappa[i,j,k] + kappa[i-1,j,k])
                kz_f = 0.5*(kappa[i,j,k] + kappa[i,j,k+1])
                kz_b = 0.5*(kappa[i,j,k] + kappa[i,j,k-1])
                fx = (kx_e*(f[i,j+1,k]-f[i,j,k]) - kx_w*(f[i,j,k]-f[i,j-1,k])) / dx**2
                fy = (ky_n*(f[i+1,j,k]-f[i,j,k]) - ky_s*(f[i,j,k]-f[i-1,j,k])) / dy**2
                fz = (kz_f*(f[i,j,k+1]-f[i,j,k]) - kz_b*(f[i,j,k]-f[i,j,k-1])) / dz**2
                out[i,j,k] = fx + fy + fz
    return out

@njit(parallel=True)
def step_2d(u, v, p, T, nu_eff, alpha_eff, dx, dy, dt, beta_g, U_wind,
            surface_heat_transfer_coeff, surface_ref_temp, u_stokes_full):
    """Execute one full sub‑step (2D case)."""
    ny, nx = u.shape
    # ---- 边界条件 ----
    u[-1, :] = U_wind
    u[0, :] = 0.0
    v[-1, :] = 0.0; v[0, :] = 0.0
    T[0, :] = 1.0
    alpha_c = surface_heat_transfer_coeff
    T[-1, :] = (T[-1, :] + dt*alpha_c*surface_ref_temp) / (1.0 + dt*alpha_c)

    # ---- 预测步 ----
    u_star = u - dt*advection_2d(u, u, v, dx, dy) + dt*diffusion_2d(u, nu_eff, dx, dy)
    v_star = v - dt*advection_2d(v, u, v, dx, dy) + dt*diffusion_2d(v, nu_eff, dx, dy) + dt*beta_g*T

    u_star[-1, :] = U_wind; u_star[0, :] = 0.0
    v_star[-1, :] = 0.0; v_star[0, :] = 0.0

    # ---- 散度 ----
    div_star = (roll_2d_1(u_star, -1) - roll_2d_1(u_star, 1))/(2*dx) + \
               (roll_2d_0(v_star, -1) - roll_2d_0(v_star, 1))/(2*dy)

    # ---- 压力迭代 (Jacobi) ----
    for _ in range(30):
        p_new = p.copy()
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                p_new[i, j] = 0.25 * (p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] -
                                      (dx**2)*div_star[i,j]/dt)
        p[:, :] = p_new
        p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]

    # ---- 校正步 ----
    u[:, :] = u_star - dt * (roll_2d_1(p, -1) - roll_2d_1(p, 1))/(2*dx)
    v[:, :] = v_star - dt * (roll_2d_0(p, -1) - roll_2d_0(p, 1))/(2*dy)

    # ---- 温度场 (Stokes 漂流 + 扩散) ----
    T_adv = advection_2d(T, u + u_stokes_full, v, dx, dy)
    T[:, :] = T - dt*T_adv + dt*diffusion_2d(T, alpha_eff, dx, dy)

@njit(parallel=True)
def step_3d(u, v, w, p, T, nu_eff, alpha_eff, dx, dy, dz, dt, beta_g, U_wind,
            surface_heat_transfer_coeff, surface_ref_temp, u_stokes_full):
    """Execute one full sub‑step (3D case)."""
    ny, nx, nz = u.shape
    # 边界
    u[-1, :, :] = U_wind; u[0, :, :] = 0.0
    v[-1, :, :] = 0.0; v[0, :, :] = 0.0
    w[-1, :, :] = 0.0; w[0, :, :] = 0.0
    T[0, :, :] = 1.0
    alpha_c = surface_heat_transfer_coeff
    T[-1, :, :] = (T[-1, :, :] + dt*alpha_c*surface_ref_temp) / (1.0 + dt*alpha_c)

    # 预测
    u_star = u - dt*advection_3d(u, u, v, w, dx, dy, dz) + dt*diffusion_3d(u, nu_eff, dx, dy, dz)
    v_star = v - dt*advection_3d(v, u, v, w, dx, dy, dz) + dt*diffusion_3d(v, nu_eff, dx, dy, dz) + dt*beta_g*T
    w_star = w - dt*advection_3d(w, u, v, w, dx, dy, dz) + dt*diffusion_3d(w, nu_eff, dx, dy, dz)

    u_star[-1, :, :] = U_wind; u_star[0, :, :] = 0.0
    v_star[-1, :, :] = 0.0; v_star[0, :, :] = 0.0
    w_star[-1, :, :] = 0.0; w_star[0, :, :] = 0.0

    # 散度
    div_star = (roll_3d_1(u_star, -1) - roll_3d_1(u_star, 1))/(2*dx) + \
               (roll_3d_0(v_star, -1) - roll_3d_0(v_star, 1))/(2*dy) + \
               (roll_3d_2(w_star, -1) - roll_3d_2(w_star, 1))/(2*dz)

    # 压力迭代（各向异性）
    coeff_x = 1.0/dx**2; coeff_y = 1.0/dy**2; coeff_z = 1.0/dz**2
    denom = 2.0*(coeff_x + coeff_y + coeff_z)
    for _ in range(30):
        p_new = p.copy()
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                for k in range(1, nz-1):
                    laplacian = coeff_x*(p[i,j-1,k] + p[i,j+1,k]) + \
                                coeff_y*(p[i-1,j,k] + p[i+1,j,k]) + \
                                coeff_z*(p[i,j,k-1] + p[i,j,k+1])
                    p_new[i,j,k] = (laplacian - div_star[i,j,k]/dt) / denom
        p[:,:,:] = p_new
        # 垂直边界外推
        p[:,0,:] = p[:,1,:]; p[:,-1,:] = p[:,-2,:]
        p[0,:,:] = p[1,:,:]; p[-1,:,:] = p[-2,:,:]

    # 校正
    u[:,:,:] = u_star - dt*(roll_3d_1(p,-1) - roll_3d_1(p,1))/(2*dx)
    v[:,:,:] = v_star - dt*(roll_3d_0(p,-1) - roll_3d_0(p,1))/(2*dy)
    w[:,:,:] = w_star - dt*(roll_3d_2(p,-1) - roll_3d_2(p,1))/(2*dz)

    # 温度
    T_adv = advection_3d(T, u + u_stokes_full, v, w, dx, dy, dz)
    T[:,:,:] = T - dt*T_adv + dt*diffusion_3d(T, alpha_eff, dx, dy, dz)


# ----------------------------------------------------------------------
# Main simulation function
# ----------------------------------------------------------------------

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
    dims=2,
    nz=60, lz=10.0,
    init_temp_noise=0.01,  # 新增：初始温度随机扰动幅度
):
    # --- 1. 物理参数与网格设置 ---
    if dims == 2:
        dx, dy = lx/nx, ly/ny
        u = v = p = T = w = None
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        p = np.zeros((ny, nx))
        T = np.ones((ny, nx))
    else:
        dx, dy, dz = lx/nx, ly/ny, lz/nz
        u = v = p = T = w = None
        u = np.zeros((ny, nx, nz))
        v = np.zeros((ny, nx, nz))
        w = np.zeros((ny, nx, nz))
        p = np.zeros((ny, nx, nz))
        T = np.ones((ny, nx, nz))

    # 添加随机小扰动，打破对称性
    if init_temp_noise > 0:
        T += init_temp_noise * np.random.randn(*T.shape)
        T = np.clip(T, 0.0, 1.0)

    # --- 波浪物理预处理 ---
    y = np.linspace(0, ly, ny)          # 垂直坐标，y=0底部，y=ly顶部
    d = ly - y                           # 距自由表面距离
    u_stokes_1d = u_stokes_surf * np.exp(-d / stokes_decay)  # 1D profile
    mix_factor = 1.0 + (mix_enhancement - 1.0) * np.exp(-d / mix_layer_depth)

    if dims == 2:
        nu_eff = nu * mix_factor[:, None] * np.ones((1, nx))
        alpha_eff = alpha * mix_factor[:, None] * np.ones((1, nx))
        u_stokes_full = u_stokes_1d[:, None] * np.ones((1, nx))
    else:
        nu_eff = nu * mix_factor[:, None, None] * np.ones((1, nx, nz))
        alpha_eff = alpha * mix_factor[:, None, None] * np.ones((1, nx, nz))
        u_stokes_full = u_stokes_1d[:, None, None] * np.ones((1, nx, nz))

    # --- 3. 创建图形和初始艺术家对象 ---
    if dims == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        X = np.linspace(0, lx, nx)
        Y = np.linspace(0, ly, ny)

        # 初始图像
        im = ax.imshow(T, origin='lower', cmap='RdYlBu_r', extent=[0, lx, 0, ly],
                       vmin=0, vmax=1)
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
    else:  # dims == 3
        fig, ax = plt.subplots(figsize=(8, 8))
        X_plot = np.linspace(0, lx, nx)
        Z_plot = np.linspace(0, lz, nz)
        im = ax.imshow(T[-1, :, :], origin='lower', cmap='RdYlBu_r',
                       extent=[0, lz, 0, lx], vmin=0, vmax=1)
        q = ax.quiver(X_plot[::3], Z_plot[::3],
                      u[-1, ::3, ::3], w[-1, ::3, ::3],
                      color='white', scale=10, animated=True)
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_title(f"3D surface | Us={u_stokes_surf:.1f}, mix_enh={mix_enhancement:.1f} Step 0")
        # No surface displacement for 3D
        surf_line = None
        surf_fill = None

    # --- 4. 动画更新函数 ---
    num_frames = nt // step_per_frame

    if return_history:
        history = []
    else:
        history = None

    def update(frame):
        nonlocal u, v, p, T, w
        for _ in range(step_per_frame):
            if dims == 2:
                step_2d(u, v, p, T, nu_eff, alpha_eff, dx, dy, dt, beta_g, U_wind,
                        surface_heat_transfer_coeff, surface_ref_temp, u_stokes_full)
            else:
                step_3d(u, v, w, p, T, nu_eff, alpha_eff, dx, dy, dz, dt, beta_g, U_wind,
                        surface_heat_transfer_coeff, surface_ref_temp, u_stokes_full)

        if dims == 2:
            im.set_array(T)
            q.set_UVC(u[::3, ::3], v[::3, ::3])
        else:
            im.set_array(T[-1, :, :])
            q.set_UVC(u[-1, ::3, ::3], w[-1, ::3, ::3])

        if return_history:
            history.append(T.copy())

        # 海气界面形变（仅2D）
        if dims == 2 and show_surface_displacement:
            T_top = T[-1, :]
            eta = surf_disp_scale * (T_top - T_top.mean())
            surf_line.set_ydata(ly + eta)
            surf_fill[0].remove()
            surf_fill[0] = ax.fill_between(x_disp, ly + eta, ly + 2.0,
                                           color='cyan', alpha=0.2)

        if dims == 2:
            ax.set_title(f"Us={u_stokes_surf:.1f}, mix_enh={mix_enhancement:.1f}, mix_d={mix_layer_depth:.2f} Step{(frame+1)*step_per_frame}")
        else:
            ax.set_title(f"3D surface | Us={u_stokes_surf:.1f}, mix_enh={mix_enhancement:.1f} Step{(frame+1)*step_per_frame}")
        return im, q

    # --- 5. 创建动画 ---
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=blit)
    if return_history:
        return anim, history
    else:
        return anim


if __name__ == "__main__":
    print("Yo! Let's model some thermal skin layers.")

    parser = argparse.ArgumentParser(description="2D/3D thermal skin layer simulation")
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
                        help='Overlay surface deformation from temperature fluctuations (2D only)')
    parser.add_argument('--surf_disp_scale', type=float, default=0.5,
                        help='Scaling factor for thermal expansion of the surface')
    # 维度选项
    parser.add_argument('--dims', type=int, default=2, choices=[2, 3],
                        help='Simulation dimensions (2 or 3)')
    parser.add_argument('--nz', type=int, default=60, help='Grid points in z (3D only)')
    parser.add_argument('--lz', type=float, default=10.0, help='Domain size in z (3D only)')

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
        dims=args.dims,
        nz=args.nz, lz=args.lz,
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
