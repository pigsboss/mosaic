"""
Generate two static 2D maps of sea surface:
  - SST (sea surface temperature) – multiscale synthetic field
  - SSH (sea surface height) – derived from thermal expansion + wave roughness
Both are saved as PNG and can be displayed in a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq, fftshift

# ----------------------------------------------------------------------
# Multiscale SST generator (Fourier‑filtered random field)
# ----------------------------------------------------------------------

def generate_multiscale_sst(
    nx=256, ny=256,              # grid size
    lx=10.0, ly=10.0,           # physical domain size (km)
    spectral_exponent=2.5,       # power‑law slope of SST spectrum
    seed=42,
):
    """
    Create a synthetic SST field with a prescribed spatial power spectrum.
    The spectrum is proportional to k^{-spectral_exponent}, giving a realistic
    multi‑scale structure (no single dominant scale).
    """
    rng = np.random.default_rng(seed)

    # Wavenumbers
    kx = fftfreq(nx, d=lx/nx)
    ky = fftfreq(ny, d=ly/ny)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_rad[0, 0] = 1.0               # avoid division by zero

    # Target spectral amplitude
    amplitude = k_rad ** (-spectral_exponent/2)
    amplitude[0, 0] = 0.0

    # Random phases
    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    S = amplitude * noise

    # Inverse FFT to real space
    sst = np.real(ifft2(S))
    # Normalize to [0, 1]
    sst = (sst - sst.min()) / (sst.max() - sst.min())
    return sst


# ----------------------------------------------------------------------
# Surface height derived from SST (thermal expansion) + short waves
# ----------------------------------------------------------------------

def generate_ssh_from_sst(
    sst,
    expansion_scale=0.2,        # m per unit SST anomaly
    wave_amplitude=0.05,         # random short‑wave roughness (m)
    wave_seed=123,
):
    """
    Derive sea surface height anomaly from SST via linear thermal expansion,
    and add small‑scale random wave roughness.
    """
    sst_anomaly = sst - np.mean(sst)
    ssh = expansion_scale * sst_anomaly

    # Add short‑scale wave roughness (small amplitude)
    rng = np.random.default_rng(wave_seed)
    roughness = wave_amplitude * rng.normal(size=sst.shape)
    ssh += roughness
    return ssh


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_fields(
    sst,
    ssh,
    lx=10.0, ly=10.0,
    savepath_sst="sst_field.png",
    savepath_ssh="ssh_field.png",
    show=True,
):
    """
    Generate and optionally show/save two static figures:
      - SST (colormap)
      - SSH (surface height, with topographic colormap)
    Both are 2D maps in (x, y) coordinates.
    """
    extent = [0, lx, 0, ly]

    # --- SST plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(sst, origin='lower', cmap='RdYlBu_r', extent=extent, vmin=0, vmax=1)
    ax1.set_title('Sea Surface Temperature (multiscale)', fontweight='bold')
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    plt.colorbar(im1, ax=ax1, label='Normalized SST')
    fig1.tight_layout()
    fig1.savefig(savepath_sst, dpi=200)
    if show:
        plt.show()
    plt.close(fig1)

    # --- SSH plot ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(ssh, origin='lower', cmap='coolwarm', extent=extent)
    ax2.set_title('Sea Surface Height anomaly', fontweight='bold')
    ax2.set_xlabel('x (km)')
    ax2.set_ylabel('y (km)')
    plt.colorbar(im2, ax=ax2, label='Height (m)')
    fig2.tight_layout()
    fig2.savefig(savepath_ssh, dpi=200)
    if show:
        plt.show()
    plt.close(fig2)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Hey, generating multiscale SST...")
    sst = generate_multiscale_sst(nx=256, ny=256, spectral_exponent=2.5)
    print("Deriving SSH...")
    ssh = generate_ssh_from_sst(sst, expansion_scale=0.2)
    print("Plotting fields...")
    plot_fields(sst, ssh, show=True)
    print("All done! Files saved: sst_field.png, ssh_field.png")
