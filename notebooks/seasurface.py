"""
Simplified 2D thermal skin layer simulation for multi‑scale demonstration.
Generates two static figures:
  - temperature cross‑section
  - surface deformation from thermal expansion
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ----------------------------------------------------------------------
# Numba‑accelerated finite‑difference kernels
# ----------------------------------------------------------------------

@njit
def roll0(f, s):
    """roll along axis=0 (rows)"""
    ny, nx = f.shape
    r = np.empty_like(f)
    if s == -1:
        r[0, :] = f[-1, :]
        r[1:, :] = f[:-1, :]
    elif s == 1:
        r[-1, :] = f[0, :]
        r[:-1, :] = f[1:, :]
    else:
        r = f
    return r

@njit
def roll1(f, s):
    """roll along axis=1 (columns)"""
    ny, nx = f.shape
    r = np.empty_like(f)
    if s == -1:
        r[:, 0] = f[:, -1]
        r[:, 1:] = f[:, :-1]
    elif s == 1:
        r[:, -1] = f[:, 0]
        r[:, :-1] = f[:, 1:]
    else:
        r = f
    return r

@njit
def advection(f, u, v, dx, dy):
    return u * (roll1(f, -1) - roll1(f, 1)) / (2.0*dx) + \
           v * (roll0(f, -1) - roll0(f, 1)) / (2.0*dy)

@njit
def diffusion_var(f, kappa, dx, dy):
    """div(kappa grad f) with harmonic mean for face coefficients."""
    ny, nx = f.shape
    out = np.zeros_like(f)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            kxe = (2 * kappa[i,j] * kappa[i,j+1]) / (kappa[i,j] + kappa[i,j+1] + 1e-12)
            kxw = (2 * kappa[i,j] * kappa[i,j-1]) / (kappa[i,j] + kappa[i,j-1] + 1e-12)
            kyn = (2 * kappa[i,j] * kappa[i+1,j]) / (kappa[i,j] + kappa[i+1,j] + 1e-12)
            kys = (2 * kappa[i,j] * kappa[i-1,j]) / (kappa[i,j] + kappa[i-1,j] + 1e-12)
            fx = (kxe*(f[i,j+1]-f[i,j]) - kxw*(f[i,j]-f[i,j-1])) / dx**2
            fy = (kyn*(f[i+1,j]-f[i,j]) - kys*(f[i,j]-f[i-1,j])) / dy**2
            out[i,j] = fx + fy
    return out

@njit(parallel=True)
def step(u, v, p, T, nu_eff, alpha_eff, dx, dy, dt, beta_g, U_wind,
         heat_coeff, ref_temp):
    """Perform one full sub‑step (boundary, predictor, Poisson, corrector, temperature)."""
    ny, nx = u.shape

    # ---- boundary conditions ----
    u[-1, :] = U_wind
    u[0, :] = 0.0
    v[-1, :] = 0.0
    v[0, :] = 0.0
    T[0, :] = 1.0
    c = heat_coeff
    T[-1, :] = (T[-1, :] + dt * c * ref_temp) / (1.0 + dt * c)

    # ---- predictor ----
    u_star = u - dt*advection(u, u, v, dx, dy) + dt*diffusion_var(u, nu_eff, dx, dy)
    v_star = v - dt*advection(v, u, v, dx, dy) + dt*diffusion_var(v, nu_eff, dx, dy) + dt*beta_g*T

    u_star[-1, :] = U_wind; u_star[0, :] = 0.0
    v_star[-1, :] = 0.0; v_star[0, :] = 0.0

    # ---- divergence ----
    div = (roll1(u_star,-1)-roll1(u_star,1))/(2*dx) + (roll0(v_star,-1)-roll0(v_star,1))/(2*dy)

    # ---- pressure (Jacobi) ----
    for _ in range(30):
        pn = p.copy()
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                pn[i,j] = 0.25*(p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - (dx**2)*div[i,j]/dt)
        p[:,:] = pn
        p[:,0] = p[:,1]; p[:,-1] = p[:,-2]
        p[0,:] = p[1,:]; p[-1,:] = p[-2,:]

    # ---- corrector ----
    u[:,:] = u_star - dt * (roll1(p,-1)-roll1(p,1))/(2*dx)
    v[:,:] = v_star - dt * (roll0(p,-1)-roll0(p,1))/(2*dy)

    # ---- temperature ----
    T_adv = advection(T, u, v, dx, dy)
    T[:,:] = T - dt*T_adv + dt*diffusion_var(T, alpha_eff, dx, dy)

# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------

def run_simulation(
    nx=128, ny=128, lx=10.0, ly=10.0,
    n_steps=2000, dt=0.005,
    nu=0.1, alpha=0.05, beta_g=1.0,
    U_wind=2.0,
    mix_enhancement=2.5,
    mix_layer_depth=0.8,
    heat_coeff=0.8,
    ref_temp=0.0,
    init_noise=0.02,
):
    """
    Run the 2D thermal skin simulation and return the final temperature field T.
    """
    dx, dy = lx/nx, ly/ny

    # initial fields
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    T = np.ones((ny, nx))

    if init_noise > 0:
        T += init_noise * np.random.randn(ny, nx)
        T = np.clip(T, 0.0, 1.0)

    # wave‑enhanced mixing profile (exponential decay from surface)
    y = np.linspace(0, ly, ny)
    d_from_surface = ly - y
    mix_prof = 1.0 + (mix_enhancement - 1.0) * np.exp(-d_from_surface / mix_layer_depth)

    nu_eff = nu * mix_prof[:, None] * np.ones((1, nx))
    alpha_eff = alpha * mix_prof[:, None] * np.ones((1, nx))

    # time loop
    for _ in range(n_steps):
        step(u, v, p, T, nu_eff, alpha_eff, dx, dy, dt, beta_g, U_wind, heat_coeff, ref_temp)

    return T  # shape (ny, nx)


def plot_temperature_field(T, lx=10.0, ly=10.0, savepath="temperature_field.png", show=True):
    """Save and optionally display the temperature cross‑section."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(T, origin='lower', cmap='RdYlBu_r', extent=[0, lx, 0, ly], vmin=0, vmax=1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Depth y (m)')
    ax.set_title('Sea surface thermal skin layer')
    plt.colorbar(im, ax=ax, label='Normalized temperature')
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_surface_deformation(T, lx=10.0, ly=10.0, scale=0.5, savepath="wave_deformation.png", show=True):
    """
    Compute and plot the surface deformation from the top‑layer temperature.
    """
    T_top = T[-1, :]
    eta = scale * (T_top - np.mean(T_top))
    x = np.linspace(0, lx, len(T_top))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, eta, 'c-', linewidth=2)
    ax.fill_between(x, eta, 0, color='cyan', alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Surface elevation (m)')
    ax.set_title('Wave‑like surface deformation from thermal expansion')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Quick self‑test (executed only when the file is run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running 2D thermal skin simulation...")
    T = run_simulation(nx=128, ny=128, n_steps=2000)
    print("Generating figures...")
    plot_temperature_field(T)
    plot_surface_deformation(T)
    print("Done. Files saved: temperature_field.png, wave_deformation.png")
