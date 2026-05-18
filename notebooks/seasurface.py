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
# Power spectrum analysis
# ----------------------------------------------------------------------

def radial_power_spectrum(field, dx, dy):
    """Compute 2D radial average power spectrum of a 2D field."""
    ny, nx = field.shape
    F = np.fft.fft2(field)
    psd2d = np.abs(F)**2
    psd2d_shifted = np.fft.fftshift(psd2d)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)

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


def plot_power_spectrum(sst, ssh, lx=10.0, ly=10.0,
                        savepath="spectra_comparison.png", show=True):
    """
    Plot radial power spectra of SST and SSH on a log‑log scale.
    Overlaid with a reference k^{-2} line for comparison.
    """
    dx = lx / sst.shape[1]
    dy = ly / sst.shape[0]

    k_sst, psd_sst = radial_power_spectrum(sst, dx, dy)
    k_ssh, psd_ssh = radial_power_spectrum(ssh, dx, dy)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_sst, psd_sst / psd_sst[0], 'b-', label='SST')
    ax.loglog(k_ssh, psd_ssh / psd_ssh[0], 'r-', label='SSH')

    # Reference line k^{-2}
    ref_k = k_sst[k_sst > 0]
    ref_line = 1e-3 * ref_k**(-2.0)
    ax.loglog(ref_k, ref_line / ref_line[0], 'k--', label=r'$\propto k^{-2}$')

    ax.set_xlabel('Wavenumber (cycles/km)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Spatial Power Spectra of Sea Surface Variables')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Yo, generating multiscale SST...")
    sst = generate_multiscale_sst(nx=256, ny=256, spectral_exponent=2.5)
    print("Deriving SSH...")
    ssh = generate_ssh_from_sst(sst, expansion_scale=0.2)
    print("Plotting fields...")
    plot_fields(sst, ssh, show=True)
    print("Plotting power spectra...")
    plot_power_spectrum(sst, ssh, show=True)
    print("All done! Files saved: sst_field.png, ssh_field.png, spectra_comparison.png")
