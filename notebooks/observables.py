"""
Observation pipeline: Apply synthetic aperture systems to SST and SSH fields
that were pre‑computed and saved as .npy files (by seasurface.py).

Usage:
  python observables.py --state calm [--lx_km 1.0] [--ly_km 1.0]

Generates side‑by‑side comparisons and power‑spectrum comparisons.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from apertures import (full_aperture, golay3, golay9,
                       compute_psf, compute_mtf,
                       D_FULL, D_GOLAY, GEO_HEIGHT)

# ----------------------------------------------------------------------
# Radial power spectrum (copied from seasurface.py to avoid import issues)
# ----------------------------------------------------------------------

def radial_power_spectrum(field, dx, dy):
    """Compute 2D radial average power spectrum of a 2D field."""
    ny, nx = field.shape
    F = np.fft.fft2(field)
    psd2d = np.abs(F) ** 2
    psd2d_shifted = np.fft.fftshift(psd2d)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    k_max = np.max(k_rad)
    n_bins = 100
    bins = np.linspace(0, k_max, n_bins + 1)
    radial_psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(ny):
        for j in range(nx):
            kr = k_rad[i, j]
            bin_idx = np.digitize(kr, bins) - 1
            if 0 <= bin_idx < n_bins:
                radial_psd[bin_idx] += psd2d_shifted[i, j]
                counts[bin_idx] += 1

    valid = counts > 0
    radial_psd[valid] /= counts[valid]
    k_center = 0.5 * (bins[1:] + bins[:-1])
    return k_center[valid], radial_psd[valid]


# ----------------------------------------------------------------------
# Interferometric observation simulation (physical)
# ----------------------------------------------------------------------

def observe_interferometric(scene, pupil, diameter, lx, ly,
                            noise_level=0.01, threshold=0.1, normalize=True):
    """
    Simulate observation through a synthetic aperture using its actual MTF,
    mapped to angular frequencies via GEO height.
    """
    # --- MTF from pupil ---
    psf, theta = compute_psf(pupil, diameter, pad_factor=1)
    dtheta = theta[1] - theta[0]
    mtf, freq_ang = compute_mtf(psf, dtheta)

    # --- Scene spatial frequency grid ---
    ny, nx = scene.shape
    dx = lx / nx
    dy = ly / ny
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    AX = FX * GEO_HEIGHT
    AY = FY * GEO_HEIGHT

    interp = RegularGridInterpolator((freq_ang, freq_ang), mtf,
                                     bounds_error=False, fill_value=0.0)
    sample = interp(np.stack((AY, AX), axis=-1))

    sample[sample < threshold] = 0.0

    F = np.fft.fft2(scene)
    if normalize:
        eps = 1e-12
        F_obs = F * np.where(sample > 0, 1.0 / (sample + eps), 0.0)
    else:
        F_obs = F * sample

    observed = np.real(np.fft.ifft2(F_obs))

    noise = noise_level * np.random.randn(ny, nx)
    observed += noise
    observed = np.clip(observed, scene.min(), scene.max())
    return observed


# ----------------------------------------------------------------------
# Generate observations for Full, Golay‑3, Golay‑9
# ----------------------------------------------------------------------

def generate_observed(sst, ssh, noise_level=0.01, N=256,
                      lx_km=1.0, ly_km=1.0,
                      threshold=0.1, normalize=True):
    """Simulate observation through the three apertures.
    lx_km, ly_km : scene extent in km, converted to metres.
    """
    lx = lx_km * 1e3
    ly = ly_km * 1e3

    apertures = {
        'Full':   (full_aperture(N, D_FULL),   D_FULL),
        'Golay3': (golay3(N, D_GOLAY),         D_GOLAY),
        'Golay9': (golay9(N, D_GOLAY),         D_GOLAY),
    }

    results = {}
    for name, (pupil, diam) in apertures.items():
        sst_obs = observe_interferometric(sst, pupil, diam, lx, ly,
                                         noise_level, threshold, normalize)
        ssh_obs = observe_interferometric(ssh, pupil, diam, lx, ly,
                                         noise_level, threshold, normalize)
        results[name] = {'sst_obs': sst_obs, 'ssh_obs': ssh_obs}
    return results


# ----------------------------------------------------------------------
# Plotting comparisons
# ----------------------------------------------------------------------

def plot_observation_comparison(sst, ssh, obs_results, lx=1.0, ly=1.0,
                                savepath_sst='obs_sst.png', savepath_ssh='obs_ssh.png',
                                show=True):
    """Create two figures: one for SST, one for SSH, each with 4 subplots."""
    extent = [0, lx, 0, ly]
    apertures = ['Full', 'Golay3', 'Golay9']

    # ----- SST -----
    fig1, axes = plt.subplots(1, 4, figsize=(20, 5))
    im = axes[0].imshow(sst, origin='lower', cmap='RdYlBu_r',
                        extent=extent, vmin=0, vmax=1)
    axes[0].set_title('True SST')
    axes[0].set_xlabel('x (km)'); axes[0].set_ylabel('y (km)')
    plt.colorbar(im, ax=axes[0], label='Normalized SST')

    for idx, name in enumerate(apertures):
        sst_obs = obs_results[name]['sst_obs']
        im = axes[idx+1].imshow(sst_obs, origin='lower', cmap='RdYlBu_r',
                                extent=extent, vmin=0, vmax=1)
        axes[idx+1].set_title(f'{name} observed')
        axes[idx+1].set_xlabel('x (km)'); axes[idx+1].set_ylabel('y (km)')
        plt.colorbar(im, ax=axes[idx+1], label='Normalized SST')

    fig1.suptitle('SST Observation through Apertures', fontweight='bold')
    fig1.tight_layout()
    fig1.savefig(savepath_sst, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig1)

    # ----- SSH -----
    fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin_ssh, vmax_ssh = ssh.min(), ssh.max()
    im = axes[0].imshow(ssh, origin='lower', cmap='coolwarm',
                        extent=extent, vmin=vmin_ssh, vmax=vmax_ssh)
    axes[0].set_title('True SSH')
    axes[0].set_xlabel('x (km)'); axes[0].set_ylabel('y (km)')
    plt.colorbar(im, ax=axes[0], label='Height (m)')

    for idx, name in enumerate(apertures):
        ssh_obs = obs_results[name]['ssh_obs']
        im = axes[idx+1].imshow(ssh_obs, origin='lower', cmap='coolwarm',
                                extent=extent, vmin=vmin_ssh, vmax=vmax_ssh)
        axes[idx+1].set_title(f'{name} observed')
        axes[idx+1].set_xlabel('x (km)'); axes[idx+1].set_ylabel('y (km)')
        plt.colorbar(im, ax=axes[idx+1], label='Height (m)')

    fig2.suptitle('SSH Observation through Apertures', fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(savepath_ssh, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig2)


# ----------------------------------------------------------------------
# Power spectrum comparison
# ----------------------------------------------------------------------

def plot_power_spectra_comparison(sst, ssh, obs_results, lx=1.0, ly=1.0,
                                  savepath_sst="spectra_sst_obs.png",
                                  savepath_ssh="spectra_ssh_obs.png",
                                  show=True):
    """Compare radial power spectra of true SST/SSH with observed fields."""
    dx = lx / sst.shape[1]
    dy = ly / sst.shape[0]

    k_sst, psd_sst = radial_power_spectrum(sst, dx, dy)
    k_ssh, psd_ssh = radial_power_spectrum(ssh, dx, dy)

    apertures = ['Full', 'Golay3', 'Golay9']
    colors = ['red', 'green', 'blue']

    # ---------- SST ----------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.loglog(k_sst, psd_sst / psd_sst[0], 'k-', linewidth=2, label='True SST')
    for name, color in zip(apertures, colors):
        sst_obs = obs_results[name]['sst_obs']
        k_obs, psd_obs = radial_power_spectrum(sst_obs, dx, dy)
        ax1.loglog(k_obs, psd_obs / psd_obs[0], color=color, linestyle='--',
                   label=f'{name} observed')
    ax1.set_xlabel('Wavenumber (cycles/km)')
    ax1.set_ylabel('Normalized Power')
    ax1.set_title('SST Power Spectra: True vs Observed')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(savepath_sst, dpi=200)
    if show:
        plt.show()
    plt.close(fig1)

    # ---------- SSH ----------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.loglog(k_ssh, psd_ssh / psd_ssh[0], 'k-', linewidth=2, label='True SSH')
    for name, color in zip(apertures, colors):
        ssh_obs = obs_results[name]['ssh_obs']
        k_obs, psd_obs = radial_power_spectrum(ssh_obs, dx, dy)
        ax2.loglog(k_obs, psd_obs / psd_obs[0], color=color, linestyle='--',
                   label=f'{name} observed')
    ax2.set_xlabel('Wavenumber (cycles/km)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title('SSH Power Spectra: True vs Observed')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(savepath_ssh, dpi=200)
    if show:
        plt.show()
    plt.close(fig2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Observe pre‑computed sea surface fields with synthetic apertures")
    parser.add_argument('--state', type=str, default='calm',
                        choices=['calm', 'langmuir', 'turbulent'],
                        help='Which sea state to load (default: calm)')
    parser.add_argument('--lx_km', type=float, default=1.0,
                        help='Scene width in km (default: 1.0)')
    parser.add_argument('--ly_km', type=float, default=1.0,
                        help='Scene height in km (default: 1.0)')
    parser.add_argument('--noise', type=float, default=0.02,
                        help='Observation noise level (default: 0.02)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='MTF threshold for observation (default: 0.1)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize MTF in observation (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    args = parser.parse_args()

    # Load pre‑computed fields
    try:
        sst = np.load(f"{args.state}_sst.npy")
        ssh = np.load(f"{args.state}_ssh.npy")
        print(f"Loaded {args.state} state fields")
    except FileNotFoundError:
        print(f"Error: Could not find {args.state}_sst.npy or {args.state}_ssh.npy.")
        print("Make sure you have run seasurface.py first to generate these files.")
        sys.exit(1)

    lx_km = args.lx_km
    ly_km = args.ly_km

    print("Running observation simulation...")
    obs = generate_observed(sst, ssh, noise_level=args.noise,
                            lx_km=lx_km, ly_km=ly_km,
                            threshold=args.threshold, normalize=args.normalize)
    print("Plotting comparisons...")
    plot_observation_comparison(sst, ssh, obs, lx=lx_km, ly=ly_km, show=True)
    print("Comparing power spectra...")
    plot_power_spectra_comparison(sst, ssh, obs, lx=lx_km, ly=ly_km, show=True)
    print("All done! Images saved.")
