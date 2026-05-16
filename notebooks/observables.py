"""
Observation pipeline: Apply synthetic aperture systems to SST and SSH fields.
Generates side‑by‑side comparisons (true vs observed) for:
  - Full aperture (4 m)
  - Golay‑3 (10 m virtual)
  - Golay‑9 (10 m virtual)
Outputs: obs_sst.png, obs_ssh.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from seasurface import generate_multiscale_sst, generate_ssh_from_sst
from apertures import full_aperture, golay3, golay9, compute_psf, D_FULL, D_GOLAY

# ----------------------------------------------------------------------
# Observation helpers
# ----------------------------------------------------------------------

def observe_scene(scene, psf, noise_level=0.01):
    """Convolve scene with PSF and add Gaussian noise."""
    blurred = fftconvolve(scene, psf, mode='same')
    noise = noise_level * np.random.randn(*scene.shape)
    observed = blurred + noise
    observed = np.clip(observed, scene.min(), scene.max())
    return observed

# ----------------------------------------------------------------------
# Main function to generate observed fields for three apertures
# ----------------------------------------------------------------------

def generate_observed(sst, ssh, noise_level=0.01, N=256):
    """
    Simulate observation through Full, Golay‑3, Golay‑9 apertures.
    Returns dict with observed SST and SSH.
    """
    # Generate physical pupils
    pupils = {
        'Full': full_aperture(N, D_FULL),
        'Golay3': golay3(N, D_GOLAY),
        'Golay9': golay9(N, D_GOLAY),
    }

    results = {}
    for name, pupil in pupils.items():
        psf, _ = compute_psf(pupil, pad_factor=1)
        psf /= psf.sum()
        sst_obs = observe_scene(sst, psf, noise_level)
        ssh_obs = observe_scene(ssh, psf, noise_level)
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
    print("All done! obs_sst.png, obs_ssh.png saved.")
