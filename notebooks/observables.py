"""
Observation pipeline: Apply synthetic aperture systems to SST and SSH fields.
Generates side‑by‑side comparisons (true vs observed) for:
  - Full aperture (4 m)
  - Golay‑3 (10 m virtual)
  - Golay‑9 (10 m virtual)

Also compares the radial power spectra of the true fields and the observed fields
to demonstrate the loss of high‑frequency information.

Outputs: obs_sst.png, obs_ssh.png, spectra_sst_obs.png, spectra_ssh_obs.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from seasurface import generate_multiscale_sst, generate_ssh_from_sst, radial_power_spectrum
from apertures import full_aperture, golay3, golay9, D_FULL, D_GOLAY, WAVELENGTH

# ----------------------------------------------------------------------
# Observation helpers
# ----------------------------------------------------------------------

def observe_frequency(scene, pupil, noise_level=0.01):
    """
    Simulate observation by masking the scene's Fourier transform with the pupil
    (which acts as the optical transfer function support), then add noise.
    """
    F = np.fft.fft2(scene)
    mask = pupil / pupil.max()            # ensure 0/1
    F_obs = F * mask
    observed = np.real(np.fft.ifft2(F_obs))
    noise = noise_level * np.random.randn(*scene.shape)
    observed += noise
    observed = np.clip(observed, scene.min(), scene.max())
    return observed


# ----------------------------------------------------------------------
# Main function to generate observed fields for three apertures
# ----------------------------------------------------------------------

def generate_observed(sst, ssh, noise_level=0.01, N=256):
    """
    Simulate observation through Full, Golay‑3, Golay‑9 apertures
    by frequency‑domain masking (interferometric sampling).
    Returns dict with observed SST and SSH.
    """
    pupils = {
        'Full': full_aperture(N, D_FULL),
        'Golay3': golay3(N, D_GOLAY),
        'Golay9': golay9(N, D_GOLAY),
    }

    results = {}
    for name, pupil in pupils.items():
        sst_obs = observe_frequency(sst, pupil, noise_level)
        ssh_obs = observe_frequency(ssh, pupil, noise_level)
        results[name] = {'sst_obs': sst_obs, 'ssh_obs': ssh_obs}
    return results


# ----------------------------------------------------------------------
# Plotting comparisons
# ----------------------------------------------------------------------

def plot_observation_comparison(sst, ssh, obs_results, lx=10.0, ly=10.0,
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

def plot_power_spectra_comparison(sst, ssh, obs_results, lx=10.0, ly=10.0,
                                  savepath_sst="spectra_sst_obs.png",
                                  savepath_ssh="spectra_ssh_obs.png",
                                  show=True):
    """
    Compare radial power spectra of true SST/SSH with those of the observed fields
    from each aperture.  This reveals how much high‑frequency content is lost due to
    the incomplete frequency coverage of the interferometric systems.
    """
    dx = lx / sst.shape[1]
    dy = ly / sst.shape[0]

    # True spectra
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
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating multiscale SST...")
    sst = generate_multiscale_sst(nx=256, ny=256, spectral_exponent=2.5)
    print("Deriving SSH...")
    ssh = generate_ssh_from_sst(sst, expansion_scale=0.2)
    print("Running observation simulation...")
    obs = generate_observed(sst, ssh, noise_level=0.02)
    print("Plotting comparisons...")
    plot_observation_comparison(sst, ssh, obs, show=True)
    print("Comparing power spectra...")
    plot_power_spectra_comparison(sst, ssh, obs, show=True)
    print("All done! obs_sst.png, obs_ssh.png, spectra_sst_obs.png, spectra_ssh_obs.png saved.")
