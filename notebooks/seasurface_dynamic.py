#!/usr/bin/env python3
"""
Continuous dynamic sea‑surface generation with state transitions and
optional SST‑SSH coupling.

Based on Proposal 0001:
  - State parameters interpolated between keyframes (calm → langmuir → turbulent).
  - Ornstein‑Uhlenbeck process in Fourier space with wavenumber‑dependent memory.
  - Coupling via a common stochastic driver (coherence ρ).

Advection feature (semi‑Lagrangian):
  - Velocity field generated from a streamfunction with the same stochastic model.
  - Fields are advected forward each time step.

For Langmuir circulation a mean east‑west shear profile is superimposed
on the stochastic velocity field, producing the characteristic streak
drifting and rolling.

Requirements:
  - numpy, matplotlib (for animation), scipy (for advection), seasurface (for state_params).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from importlib import reload
import sys
import argparse

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    map_coordinates = None

sys.path.insert(0, '.')   # ensure local imports work
import seasurface
reload(seasurface)
from seasurface import state_params

# Increase embed limit to avoid warnings with large animations
plt.rcParams['animation.embed_limit'] = 100  # MB

# ----------------------------------------------------------------------
# Append velocity parameters to state_params for advection
# ----------------------------------------------------------------------
for state in state_params:
    if state == 'calm':
        state_params[state]['velocity'] = {
            'alpha': 4.0,   # was 3.5 → smoother
            's': 0.2, 'theta0': 0.0,
            'isotropic_component': 0.9, 'peaks': None,
            'tau0': 7200.0, 'tau_alpha': 0.5
        }
    elif state == 'langmuir':
        state_params[state]['velocity'] = {
            'alpha': 3.5,   # was 3.0 → smoother
            's': 2.0, 'theta0': 0.0,
            'isotropic_component': 0.2, 'peaks': None,
            'tau0': 3600.0, 'tau_alpha': 0.5
        }
    elif state == 'turbulent':
        state_params[state]['velocity'] = {
            'alpha': 4.0,   # was 3.0 → smoother
            's': 0.3, 'theta0': 0.0,
            'isotropic_component': 0.7, 'peaks': None,
            'tau0': 900.0, 'tau_alpha': 0.8
        }

# ----------------------------------------------------------------------
# Semi‑Lagrangian advection
# ----------------------------------------------------------------------

def advect_semilag(field, u, v, dx, dy, dt):
    """
    Semi‑Lagrangian advection of a 2D periodic field.
    field : 2D ndarray (ny, nx)
    u, v  : velocity components on the same grid (m/s)
    dx, dy: grid spacing in meters
    dt    : time step in seconds
    Returns the advected field.
    """
    if map_coordinates is None:
        raise ImportError("scipy.ndimage is required for advection. Install scipy.")
    ny, nx = field.shape
    y, x = np.mgrid[0:ny, 0:nx]
    # Departure point in index space (periodic)
    x_dep = (x - u * dt / dx) % nx
    y_dep = (y - v * dt / dy) % ny
    coords = np.stack([y_dep.ravel(), x_dep.ravel()])
    interpolated = map_coordinates(field, coords, order=3, mode='wrap')
    return interpolated.reshape(ny, nx)

# ----------------------------------------------------------------------
# Internal helper: build target 2D PSD from a parameter dict
# ----------------------------------------------------------------------
def _target_psd_from_params(params, k_rad, theta):
    """2D power spectrum P(kx, ky) according to the anisotropic model."""
    alpha = params['alpha']
    theta0 = params['theta0']
    s = params['s']
    iso = params['isotropic_component']
    peaks = params.get('peaks')

    # angular distribution
    dtheta = theta - theta0
    D = np.cos(0.5 * dtheta) ** (2 * s)
    D = np.maximum(D, 0.0)
    D = D / D.mean()

    direction = (1 - iso) * D + iso
    P = k_rad ** (-alpha) * direction
    P[0, 0] = 0.0

    if peaks is not None:
        for k0, w, amp in peaks:
            sigma = k0 * w
            gauss = amp * np.exp(-((k_rad - k0) ** 2) / (2 * sigma ** 2))
            P += gauss * direction
    return P

# ----------------------------------------------------------------------
# Parameter interpolation (imported from seasurface for convenience)
# ----------------------------------------------------------------------
interpolate_state_params = seasurface.interpolate_state_params

# ----------------------------------------------------------------------
# Coupled time‑series generator
# ----------------------------------------------------------------------
def generate_coupled_timeseries(duration, dt, nx, ny, lx, ly, script, seed=42,
                                tau0=5.0, tau_alpha=1.0,
                                rho=0.0,
                                save_path=None,
                                advection=False):
    """
    Generate a continuous time series of SST and SSH fields with an
    Ornstein‑Uhlenbeck process in Fourier space.  A common stochastic
    driver can be used to introduce coherence between the two fields.

    Parameters
    ----------
    duration : float
        Total simulation time in seconds.
    dt : float
        Time step in seconds.
    nx, ny : int
        Spatial grid size.
    lx, ly : float
        Domain size in km.
    script : list of (t_start, state_name)
        Timeline, e.g. [(0, 'calm'), (3600, 'langmuir'), (7200, 'turbulent')].
        Times in seconds.  Must start at t=0.
    seed : int
        Random seed.
    tau0 : float
        Decorrelation time (seconds) at the smallest non‑zero wavenumber.
    tau_alpha : float
        Exponent for wavenumber scaling: tau(k) = tau0 * (k_min/k)^tau_alpha.
    rho : float, 0 ≤ ρ ≤ 1
        Coherence between SST and SSH innovations.
        ρ=0 → independent fields; ρ=1 → perfectly synchronised random forcing.
    save_path : str or None
        If provided, the arrays are saved as ``<save_path>.npz``.
    advection : bool
        If True, advect the fields using a stochastically generated velocity field.
        When Langmuir circulation is active, a mean east‑west shear is added.

    Returns
    -------
    t_out : 1D ndarray (n_frames,)
    sst_ts : ndarray (n_frames, ny, nx)
    ssh_ts : ndarray (n_frames, ny, nx)
    """
    rng = np.random.default_rng(seed)

    # sort script, ensure start at 0
    script = sorted(script, key=lambda x: x[0])
    if script[0][0] > 0:
        script.insert(0, (0.0, script[0][1]))

    # ---- spatial frequency grids ----
    kx = np.fft.fftfreq(nx, d=lx / nx)
    ky = np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_rad[0, 0] = 1e-12               # avoid singularity later
    theta = np.arctan2(KY, KX)

    # ---- Hermitian half‑plane mask ----
    half_mask = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            if (kx[j] > 0) or (kx[j] == 0 and ky[i] >= 0):
                half_mask[i, j] = True

    half_indices = np.where(half_mask)
    conj_i = (ny - half_indices[0]) % ny
    conj_j = (nx - half_indices[1]) % nx
    conj_indices = (conj_i, conj_j)

    # ---- decorrelation time ----
    k_min = np.min(k_rad[k_rad > 0])
    tau_k = tau0 * (k_min / (k_rad + 1e-12)) ** tau_alpha
    tau_k[0, 0] = tau0   # DC (irrelevant)

    # ---- AR(2) coefficient arrays ----
    p_exp = np.exp(-dt / tau_k)
    p2 = p_exp ** 2
    sigma_ar2_full = np.sqrt(np.maximum(1e-30, (1 - p2) ** 3 / (1 + p2)))

    def _fill_full(half_data):
        full = np.zeros((ny, nx), dtype=complex)
        full[half_indices] = half_data
        full[conj_indices] = np.conj(half_data)
        return full

    # ── helper: return the blended parameter dict for a given field type ──
    def _blended_params(t, field_type):
        # find the segment containing time t
        n_seg = len(script) - 1
        i = 0
        while i < n_seg and t >= script[i+1][0]:
            i += 1
        t_start, state_a = script[i]
        t_end = script[i+1][0] if i < n_seg else t_start
        state_b = script[i+1][1] if i < n_seg else state_a
        if t_end - t_start > 1e-9:
            frac = (t - t_start) / (t_end - t_start)
        else:
            frac = 0.0
        params_a = state_params[state_a][field_type]
        params_b = state_params[state_b][field_type]
        return interpolate_state_params(params_a, params_b, frac)

    # ── helper: return the state name active at time t (for shear logic) ──
    def _current_state(t):
        n_seg = len(script) - 1
        i = 0
        while i < n_seg and t >= script[i+1][0]:
            i += 1
        return script[i][1]

    # ---- initialise past states from target spectrum at t=0 ----
    params_sst0 = _blended_params(0.0, 'sst')
    params_ssh0 = _blended_params(0.0, 'ssh')
    P_sst0 = _target_psd_from_params(params_sst0, k_rad, theta)
    P_ssh0 = _target_psd_from_params(params_ssh0, k_rad, theta)

    noise1_sst = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
    noise2_sst = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
    a2_sst = np.sqrt(P_sst0[half_indices]) * noise1_sst[half_indices]
    a1_sst = np.sqrt(P_sst0[half_indices]) * noise2_sst[half_indices]

    noise1_ssh = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
    noise2_ssh = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
    a2_ssh = np.sqrt(P_ssh0[half_indices]) * noise1_ssh[half_indices]
    a1_ssh = np.sqrt(P_ssh0[half_indices]) * noise2_ssh[half_indices]

    # ----------  velocity AR(2) state (if advection) ----------
    if advection:
        # blending helper for velocity
        def _blended_vel_params(t):
            n_seg = len(script) - 1
            i = 0
            while i < n_seg and t >= script[i+1][0]:
                i += 1
            t_start, state_a = script[i]
            t_end = script[i+1][0] if i < n_seg else t_start
            state_b = script[i+1][1] if i < n_seg else state_a
            frac = max(0.0, min(1.0, (t - t_start) / (t_end - t_start))) if t_end > t_start else 0.0
            params_a = state_params[state_a]['velocity']
            params_b = state_params[state_b]['velocity']
            return interpolate_state_params(params_a, params_b, frac)

        # initialize velocity AR(2) state at t=0
        vel_params0 = _blended_vel_params(0.0)
        P_vel0 = _target_psd_from_params(vel_params0, k_rad, theta)
        noise1_v = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
        noise2_v = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))) / np.sqrt(2)
        a2_vel = np.sqrt(P_vel0[half_indices]) * noise1_v[half_indices]
        a1_vel = np.sqrt(P_vel0[half_indices]) * noise2_v[half_indices]
    # ----------------------------------------------------------------

    # ---- time loop ----
    n_frames = int(np.ceil(duration / dt)) + 1
    t_out = np.linspace(0, duration, n_frames)
    sst_ts = np.empty((n_frames, ny, nx), dtype=np.float64)
    ssh_ts = np.empty((n_frames, ny, nx), dtype=np.float64)

    for idx in range(n_frames):
        t = t_out[idx]
        p_sst = _blended_params(t, 'sst')
        p_ssh = _blended_params(t, 'ssh')
        P_sst = _target_psd_from_params(p_sst, k_rad, theta)
        P_ssh = _target_psd_from_params(p_ssh, k_rad, theta)

        # innovation factor to reduce random fluctuations when advection is active
        innov_factor = 0.3 if advection else 1.0

        sigma_sst_half = sigma_ar2_full[half_indices] * np.sqrt(np.maximum(P_sst[half_indices], 0.0)) * innov_factor
        sigma_ssh_half = sigma_ar2_full[half_indices] * np.sqrt(np.maximum(P_ssh[half_indices], 0.0)) * innov_factor

        # coupled noise on half‑plane
        w_common_half = (rng.normal(size=(ny, nx))[half_indices] +
                         1j * rng.normal(size=(ny, nx))[half_indices]) / np.sqrt(2)
        w_sst_ind_half = (rng.normal(size=(ny, nx))[half_indices] +
                          1j * rng.normal(size=(ny, nx))[half_indices]) / np.sqrt(2)
        w_ssh_ind_half = (rng.normal(size=(ny, nx))[half_indices] +
                          1j * rng.normal(size=(ny, nx))[half_indices]) / np.sqrt(2)

        w_sst_half = rho * w_common_half + np.sqrt(1 - rho**2) * w_sst_ind_half
        w_ssh_half = rho * w_common_half + np.sqrt(1 - rho**2) * w_ssh_ind_half

        # AR(2) update
        a_new_sst = (2.0 * p_exp[half_indices] * a1_sst
                     - p2[half_indices] * a2_sst
                     + sigma_sst_half * w_sst_half)
        a_new_ssh = (2.0 * p_exp[half_indices] * a1_ssh
                     - p2[half_indices] * a2_ssh
                     + sigma_ssh_half * w_ssh_half)

        a2_sst, a1_sst = a1_sst, a_new_sst
        a2_ssh, a1_ssh = a1_ssh, a_new_ssh

        # ---------- 更新速度场并得到 u, v ----------
        if advection:
            vp = _blended_vel_params(t)
            P_vel = _target_psd_from_params(vp, k_rad, theta)
            # recompute tau and AR(2) coefficients for velocity
            tau_k_vel = vp['tau0'] * (k_min / (k_rad + 1e-12)) ** vp['tau_alpha']
            tau_k_vel[0, 0] = vp['tau0']
            p_vel = np.exp(-dt / tau_k_vel)
            p2_vel = p_vel**2
            sigma_vel_full = np.sqrt(np.maximum(1e-30, (1 - p2_vel)**3 / (1 + p2_vel)))
            sigma_vel_half = sigma_vel_full[half_indices] * np.sqrt(np.maximum(P_vel[half_indices], 0.0))
            w_vel_half = (rng.normal(size=(ny, nx))[half_indices] +
                          1j * rng.normal(size=(ny, nx))[half_indices]) / np.sqrt(2)
            a_new_vel = (2.0 * p_vel[half_indices] * a1_vel -
                         p2_vel[half_indices] * a2_vel +
                         sigma_vel_half * w_vel_half)
            a2_vel, a1_vel = a1_vel, a_new_vel
            # Build full streamfunction
            psi_full = np.zeros((ny, nx), dtype=complex)
            psi_full[half_indices] = a1_vel
            psi_full[conj_indices] = np.conj(a1_vel)
            psi = np.real(np.fft.ifft2(psi_full))
            # velocity components via centered differences (m/s)
            dx_m = lx * 1000.0 / nx   # metres
            dy_m = ly * 1000.0 / ny
            u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dy_m)
            v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx_m)

            # ── Add mean east‑west shear for Langmuir circulation ──
            if _current_state(t) == 'langmuir':
                U0 = 0.1   # m/s, maximum shear velocity
                # y coordinates in metres (domain height ly in km)
                y_m = np.linspace(0, ly * 1000.0, ny)
                shear_profile = U0 * np.sin(np.pi * y_m / (ly * 1000.0))
                u += shear_profile[:, np.newaxis]   # broadcast to (ny, nx)
        # ----------------------------------------------------------------

        # full Hermitian arrays
        a_full_sst = _fill_full(a1_sst)
        a_full_ssh = _fill_full(a1_ssh)

        sst = np.real(np.fft.ifft2(a_full_sst))
        ssh = np.real(np.fft.ifft2(a_full_ssh))

        # ---------- 平流 (advection) with sub‑stepping ----------
        if advection:
            n_sub = 3
            dt_sub = dt / n_sub
            sst_sub = sst
            ssh_sub = ssh
            for _ in range(n_sub):
                sst_sub = advect_semilag(sst_sub, u, v, dx_m, dy_m, dt_sub)
                ssh_sub = advect_semilag(ssh_sub, u, v, dx_m, dy_m, dt_sub)
            sst_ts[idx] = sst_sub
            ssh_ts[idx] = ssh_sub
        else:
            sst_ts[idx] = sst
            ssh_ts[idx] = ssh

    # ── Global normalization (all frames together) ────────────────
    sst_min_all = sst_ts.min()
    sst_max_all = sst_ts.max()
    if sst_max_all > sst_min_all:
        sst_ts = (sst_ts - sst_min_all) / (sst_max_all - sst_min_all)
    else:
        sst_ts[:] = 0.5

    ssh_min_all = ssh_ts.min()
    ssh_max_all = ssh_ts.max()
    if ssh_max_all > ssh_min_all:
        ssh_norm = (ssh_ts - ssh_min_all) / (ssh_max_all - ssh_min_all)
    else:
        ssh_norm = np.full_like(ssh_ts, 0.5)
    ssh_ts = (ssh_norm - 0.5) * 0.1

    if save_path is not None:
        np.savez(save_path, t=t_out, sst=sst_ts, ssh=ssh_ts,
                 tau0=tau0, tau_alpha=tau_alpha, rho=rho, script=script)

    return t_out, sst_ts, ssh_ts


# ----------------------------------------------------------------------
# NEW: generate_state_sequence (single‑state wrapper)
# ----------------------------------------------------------------------
def generate_state_sequence(state, duration, dt,
                            nx=256, ny=256, lx=1.0, ly=1.0,
                            tau0=600.0, tau_alpha=0.8, seed=42, rho=0.0,
                            advection=False):
    """
    Generate SST and SSH time series for a fixed sea state (no transition).
    Thin wrapper around generate_coupled_timeseries with a single‑state script.

    Parameters
    ----------
    state : str
        'calm', 'langmuir' or 'turbulent'.
    duration : float
        Total simulation time in seconds.
    dt : float
        Time step in seconds.
    nx, ny : int
        Grid size (default 256).
    lx, ly : float
        Domain size in km.
    tau0 : float
        Decorrelation time (seconds) at the smallest wavenumber.
    tau_alpha : float
        Exponent for τ(k) = τ₀ · (k_min/k)^α.
    seed : int
        Random seed.
    rho : float
        SST‑SSH coherence (0 = independent).
    advection : bool
        If True, advect the fields.

    Returns
    -------
    t : ndarray (n_frames,)
    sst_ts : ndarray (n_frames, ny, nx)
    ssh_ts : ndarray (n_frames, ny, nx)
    """
    t, sst_ts, ssh_ts = generate_coupled_timeseries(
        duration=duration, dt=dt, nx=nx, ny=ny, lx=lx, ly=ly,
        script=[(0, state)],
        tau0=tau0, tau_alpha=tau_alpha, rho=rho, seed=seed,
        advection=advection
    )
    return t, sst_ts, ssh_ts


# ----------------------------------------------------------------------
# Jupyter animation utilities
# ----------------------------------------------------------------------
def animate_fields(t, sst_ts, ssh_ts, lx=1.0, ly=1.0, interval=100,
                   as_html5=True, show=True, save_html=None, save_gif=None):
    """
    Create a dual‑panel animation of SST (left) and SSH (right) evolving in time.
    Parameters
    ----------
    t : 1D array (n_frames,)          time in seconds
    sst_ts : 3D array (n_frames, ny, nx)
    ssh_ts : 3D array (n_frames, ny, nx)
    lx, ly : float                    domain size in km
    interval : int                    time between frames in ms
    as_html5 : bool                   if True, return HTML5 <video> tag string;
                                      otherwise return the FuncAnimation object.
    show : bool                       if True, display the animation in an interactive window
    save_html : str or None           if given, save the animation as an HTML file
    save_gif : str or None            if given, save the animation as a GIF file (requires pillow)

    Returns
    -------
    HTML string or FuncAnimation
    """
    from matplotlib import animation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    extent = [0, lx, 0, ly]
    im1 = ax1.imshow(sst_ts[0], origin='lower', cmap='RdYlBu_r',
                     extent=extent, vmin=0, vmax=1, animated=True)
    ax1.set_title('SST')
    ax1.set_xlabel('x (km)'); ax1.set_ylabel('y (km)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # SSH color limits from full range
    vmin_ssh, vmax_ssh = ssh_ts.min(), ssh_ts.max()
    im2 = ax2.imshow(ssh_ts[0], origin='lower', cmap='coolwarm',
                     extent=extent, vmin=vmin_ssh, vmax=vmax_ssh, animated=True)
    ax2.set_title('SSH')
    ax2.set_xlabel('x (km)'); ax2.set_ylabel('y (km)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    time_text = fig.suptitle(f't = {t[0]:.1f} s')

    def update(frame):
        im1.set_array(sst_ts[frame])
        im2.set_array(ssh_ts[frame])
        time_text.set_text(f't = {t[frame]:.1f} s')
        return im1, im2, time_text

    ani = FuncAnimation(fig, update, frames=len(t),
                        interval=interval, blit=True, repeat=True)

    # Generate HTML string if needed
    if as_html5 or save_html is not None:
        html_str = ani.to_html5_video()
    else:
        html_str = None

    # Save files before potentially closing figure
    if save_html is not None:
        with open(save_html, 'w') as f:
            f.write(html_str)
    if save_gif is not None:
        ani.save(save_gif, writer='pillow', fps=5)

    # Display or close
    if show:
        plt.show()
    else:
        plt.close(fig)

    if as_html5:
        return html_str
    else:
        return ani


# ----------------------------------------------------------------------
# Spectral evolution snapshot plot
# ----------------------------------------------------------------------
def plot_spectral_evolution(t, sst_ts, ssh_ts, lx=1.0, ly=1.0,
                            times=None, n_snapshots=5):
    """
    Show radial power spectrum at several time snapshots for SST and SSH.
    Parameters
    ----------
    t : 1D array
    sst_ts, ssh_ts : 3D arrays
    lx, ly : float
    times : list of floats or None.  If None, n_snapshots equally spaced times.
    n_snapshots : int

    Returns
    -------
    fig
    """
    ny, nx = sst_ts.shape[1:]
    dx = lx / nx
    dy = ly / ny

    if times is None:
        indices = np.linspace(0, len(t)-1, n_snapshots, dtype=int)
        times = t[indices]
    else:
        indices = [np.argmin(np.abs(t - time)) for time in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(indices)))

    for idx, time, color in zip(indices, times, colors):
        sst_frame = sst_ts[idx]
        ssh_frame = ssh_ts[idx]
        k_sst, psd_sst = seasurface.radial_power_spectrum(sst_frame, dx, dy)
        k_ssh, psd_ssh = seasurface.radial_power_spectrum(ssh_frame, dx, dy)
        ax1.loglog(k_sst, psd_sst/psd_sst[0], color=color, label=f'{time:.0f} s')
        ax2.loglog(k_ssh, psd_ssh/psd_ssh[0], color=color, label=f'{time:.0f} s')

    ax1.set_title('SST Radial Power Spectra')
    ax2.set_title('SSH Radial Power Spectra')
    for ax in (ax1, ax2):
        ax.set_xlabel('Wavenumber (cyc/km)')
        ax.set_ylabel('Normalized Power')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# NEW: time_averaged_radial_psd
# ----------------------------------------------------------------------
def time_averaged_radial_psd(frames, dx, dy):
    """
    Compute time‑averaged radial power spectrum.
    For each frame, compute |FFT|² ; average over time; then radial binning.

    Parameters
    ----------
    frames : ndarray (n_frames, ny, nx)
    dx, dy : float
        Grid spacing (same unit as wavenumber).

    Returns
    -------
    k_center : 1D array
    radial_psd : 1D array
    """
    n_frames, ny, nx = frames.shape
    psd2d_sum = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_frames):
        F = np.fft.fft2(frames[i])
        psd2d_sum += np.abs(F) ** 2
    psd2d_avg = psd2d_sum / n_frames

    # radial binning (as in seasurface.radial_power_spectrum without mask)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    k_max = np.max(k_rad)
    n_bins = 100
    bins = np.linspace(0, k_max, n_bins + 1)
    radial_psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    psd_shifted = np.fft.fftshift(psd2d_avg)
    for i in range(ny):
        for j in range(nx):
            kr = k_rad[i, j]
            idx = np.digitize(kr, bins) - 1
            if 0 <= idx < n_bins:
                radial_psd[idx] += psd_shifted[i, j]
                counts[idx] += 1
    valid = counts > 0
    radial_psd[valid] /= counts[valid]
    k_center = 0.5 * (bins[1:] + bins[:-1])
    return k_center[valid], radial_psd[valid]


# ----------------------------------------------------------------------
# NEW: compute_time_averaged_moment_tensor
# ----------------------------------------------------------------------
def compute_time_averaged_moment_tensor(frames, dx, dy, n_bins=100):
    """
    Time‑averaged 2D power spectrum → spectral moment tensor per radial bin.
    Returns k_centers, m20, m11, m02 arrays.
    """
    n_frames, ny, nx = frames.shape
    psd2d_sum = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_frames):
        F = np.fft.fft2(frames[i])
        psd2d_sum += np.abs(F) ** 2
    psd2d_avg = psd2d_sum / n_frames

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    m20 = np.zeros(n_bins)
    m11 = np.zeros(n_bins)
    m02 = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_rad >= bins[i]) & (k_rad < bins[i+1])
        if not np.any(mask):
            continue
        p_vals = psd2d_avg[mask]
        kx_vals = KX[mask]
        ky_vals = KY[mask]
        total = np.sum(p_vals)
        if total == 0:
            continue
        m20[i] = np.sum(p_vals * kx_vals ** 2) / total
        m11[i] = np.sum(p_vals * kx_vals * ky_vals) / total
        m02[i] = np.sum(p_vals * ky_vals ** 2) / total

    return k_centers, m20, m11, m02


# =============================================================================
# Main – command line interface
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sea surface dynamic time series & spectral analysis"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ----- fields subcommand -----
    fields_parser = subparsers.add_parser('fields', help='Generate time series and save as .npy')
    fields_parser.add_argument('--state', choices=['calm','langmuir','turbulent','all'],
                               default='all', help='Sea state (default: all)')
    fields_parser.add_argument('--duration', type=float, default=300, help='Duration in seconds')
    fields_parser.add_argument('--dt', type=float, default=1, help='Time step in seconds')
    fields_parser.add_argument('--nx', type=int, default=256, help='Grid points in x')
    fields_parser.add_argument('--ny', type=int, default=256, help='Grid points in y')
    fields_parser.add_argument('--lx', type=float, default=1.0, help='Domain size x (km)')
    fields_parser.add_argument('--ly', type=float, default=1.0, help='Domain size y (km)')
    fields_parser.add_argument('--tau0', type=float, default=7200.0, help='Decorrelation time at largest scale')
    fields_parser.add_argument('--tau_alpha', type=float, default=0.3, help='Exponent for tau scaling')
    fields_parser.add_argument('--seed', type=int, default=42)
    fields_parser.add_argument('--advection', action='store_true', default=True)
    fields_parser.add_argument('--no-advection', dest='advection', action='store_false')
    fields_parser.add_argument('--save', default='timeseries', help='Base name prefix for output .npy files')

    # ----- spectra subcommand -----
    spectra_parser = subparsers.add_parser('spectra', help='Load .npy and compute spectra')
    spectra_parser.add_argument('--load', required=True, help='Path to .npy time series (3D array)')
    spectra_parser.add_argument('--lx', type=float, default=1.0, help='Domain size x (km)')
    spectra_parser.add_argument('--ly', type=float, default=1.0, help='Domain size y (km)')
    spectra_parser.add_argument('--save_fig', default=None, help='Save figure to this path (e.g. spectra.png)')
    spectra_parser.add_argument('--save_data', default=None, help='Save computed spectra as .npz with this prefix')

    args = parser.parse_args()

    if args.command == 'fields':
        states_to_run = ['calm', 'langmuir', 'turbulent'] if args.state == 'all' else [args.state]
        titles = {'calm': 'Calm', 'langmuir': 'Langmuir', 'turbulent': 'Turbulent'}

        for state in states_to_run:
            print(f"Generating {state} time series (advection={args.advection})...")
            t, sst_ts, ssh_ts = generate_state_sequence(
                state, args.duration, args.dt,
                nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
                tau0=args.tau0, tau_alpha=args.tau_alpha,
                rho=0.0, seed=args.seed,
                advection=args.advection
            )
            fname_sst = f"{args.save}_{state}_sst.npy"
            fname_ssh = f"{args.save}_{state}_ssh.npy"
            np.save(fname_sst, sst_ts)
            np.save(fname_ssh, ssh_ts)
            print(f"  Saved {fname_sst} and {fname_ssh}")

    elif args.command == 'spectra':
        data = np.load(args.load)
        if data.ndim != 3:
            raise ValueError("Loaded data must be a 3D array (time, y, x)")
        nx, ny = data.shape[2], data.shape[1]
        dx = args.lx / nx
        dy = args.ly / ny

        k_iso, power_iso = time_averaged_radial_psd(data, dx, dy)
        k_mom, m20, m11, m02 = compute_time_averaged_moment_tensor(data, dx, dy)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Isotropic power
        axes[0].loglog(k_iso, power_iso / power_iso[0], 'k-')
        axes[0].set_title('Time‑averaged isotropic radial power')
        axes[0].set_xlabel('Wavenumber (cyc/km)')
        axes[0].set_ylabel('Normalized Power')
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

        # Moment tensor components
        axes[1].semilogx(k_mom, m20, label='m20')
        axes[1].semilogx(k_mom, m11, label='m11')
        axes[1].semilogx(k_mom, m02, label='m02')
        axes[1].set_title('Spectral moment tensor components')
        axes[1].set_xlabel('Wavenumber (cyc/km)')
        axes[1].set_ylabel('Moment')
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

        fig.tight_layout()
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=150)
            print(f"Saved figure: {args.save_fig}")
        plt.show()

        if args.save_data:
            np.savez(f"{args.save_data}.npz",
                     k_iso=k_iso, power_iso=power_iso,
                     k_mom=k_mom, m20=m20, m11=m11, m02=m02)
            print(f"Saved data: {args.save_data}.npz")
