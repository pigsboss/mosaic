#!/usr/bin/env python3
"""
Continuous dynamic sea‑surface generation with state transitions and
optional SST‑SSH coupling.

Based on Proposal 0001:
  - State parameters interpolated between keyframes (calm → langmuir → turbulent).
  - Ornstein‑Uhlenbeck process in Fourier space with wavenumber‑dependent memory.
  - Coupling via a common stochastic driver (coherence ρ).

Requirements:
  - numpy, matplotlib (for animation), seasurface (for state_params).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from importlib import reload
import sys
sys.path.insert(0, '.')   # ensure local imports work
import seasurface
reload(seasurface)
from seasurface import state_params

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
                                save_path=None):
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

        sigma_sst_half = sigma_ar2_full[half_indices] * np.sqrt(np.maximum(P_sst[half_indices], 0.0))
        sigma_ssh_half = sigma_ar2_full[half_indices] * np.sqrt(np.maximum(P_ssh[half_indices], 0.0))

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

        # full Hermitian arrays
        a_full_sst = _fill_full(a1_sst)
        a_full_ssh = _fill_full(a1_ssh)

        sst = np.real(np.fft.ifft2(a_full_sst))
        ssh = np.real(np.fft.ifft2(a_full_ssh))

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
                            tau0=600.0, tau_alpha=0.8, seed=42, rho=0.0):
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

    Returns
    -------
    t : ndarray (n_frames,)
    sst_ts : ndarray (n_frames, ny, nx)
    ssh_ts : ndarray (n_frames, ny, nx)
    """
    t, sst_ts, ssh_ts = generate_coupled_timeseries(
        duration=duration, dt=dt, nx=nx, ny=ny, lx=lx, ly=ly,
        script=[(0, state)],
        tau0=tau0, tau_alpha=tau_alpha, rho=rho, seed=seed
    )
    return t, sst_ts, ssh_ts


# ----------------------------------------------------------------------
# Jupyter animation utilities
# ----------------------------------------------------------------------
def animate_fields(t, sst_ts, ssh_ts, lx=1.0, ly=1.0, interval=100, as_html5=True):
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
    plt.close(fig)
    if as_html5:
        return ani.to_html5_video()
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


# =============================================================================
# Main – demo (updated: three separate state sequences)
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    duration = 7200   # 2 hours
    dt = 60           # timestep (seconds)
    nx, ny = 256, 256
    lx, ly = 1.0, 1.0   # km
    tau0 = 600.0        # decorrelation time at largest scale
    tau_alpha = 0.8

    states = ['calm', 'langmuir', 'turbulent']
    titles = {'calm': 'Calm', 'langmuir': 'Langmuir', 'turbulent': 'Turbulent'}

    for state in states:
        print(f"Generating {state} time series...")
        t, sst_ts, ssh_ts = generate_state_sequence(
            state, duration, dt,
            nx=nx, ny=ny, lx=lx, ly=ly,
            tau0=tau0, tau_alpha=tau_alpha, rho=0.0, seed=42
        )

        # Animation
        try:
            video = animate_fields(t, sst_ts, ssh_ts, lx=lx, ly=ly, interval=80, as_html5=True)
            fname = f"animation_{state}.html"
            with open(fname, 'w') as f:
                f.write(f"<html><body>{video}</body></html>")
            print(f"  Saved {fname}")
        except Exception as e:
            print(f"  Animation skipped: {e}")

        # Time‑averaged vs single‑frame power spectra
        dx = lx / nx
        dy = ly / ny

        k_sst1, psd_sst1 = seasurface.radial_power_spectrum(sst_ts[0], dx, dy)
        k_ssh1, psd_ssh1 = seasurface.radial_power_spectrum(ssh_ts[0], dx, dy)

        k_sst_avg, psd_sst_avg = time_averaged_radial_psd(sst_ts, dx, dy)
        k_ssh_avg, psd_ssh_avg = time_averaged_radial_psd(ssh_ts, dx, dy)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].loglog(k_sst1, psd_sst1/psd_sst1[0], 'k--', alpha=0.5, label='single frame')
        axes[0].loglog(k_sst_avg, psd_sst_avg/psd_sst_avg[0], 'r-', label='time averaged')
        axes[0].set_title(f'SST – {titles[state]}')
        axes[0].set_xlabel('Wavenumber (cyc/km)')
        axes[0].set_ylabel('Normalized Power')
        axes[0].legend(); axes[0].grid(True, which='both', linestyle='--', alpha=0.5)

        axes[1].loglog(k_ssh1, psd_ssh1/psd_ssh1[0], 'k--', alpha=0.5, label='single frame')
        axes[1].loglog(k_ssh_avg, psd_ssh_avg/psd_ssh_avg[0], 'b-', label='time averaged')
        axes[1].set_title(f'SSH – {titles[state]}')
        axes[1].set_xlabel('Wavenumber (cyc/km)')
        axes[1].set_ylabel('Normalized Power')
        axes[1].legend(); axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

        fig.suptitle(f'Time‑averaged vs Instantaneous Spectra ({titles[state]})', fontweight='bold')
        fig.tight_layout()
        fname = f"spectra_avg_{state}.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved {fname}")
        plt.close(fig)

    print("All state sequences processed.")
