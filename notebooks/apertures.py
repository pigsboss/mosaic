"""
Optical synthetic aperture pupil functions, PSF and MTF.
- Full aperture (monolithic)
- Golay‑3 (3 sub‑apertures)
- Golay‑9 (9 sub‑apertures, non‑redundant)

Engineering constraints:
  - Single aperture diameter = 4 m -> cutoff ~400 kcyc/rad (at 10 µm)
  - Golay‑3 and Golay‑9 share a virtual aperture of 10 m -> cutoff ~1 Mcyc/rad
  - Sub‑aperture size: 0.5 m

PSF is shown in ground‑projected coordinates (metres) assuming
a geostationary orbit height of 35,786 km.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# Engineering constraints
# ----------------------------------------------------------------------
WAVELENGTH = 10.0e-6          # 10 µm (thermal infrared)
D_FULL     = 4.0              # diameter of monolithic telescope (m)
D_GOLAY    = 10.0             # virtual aperture diameter for Golay arrays (m)
SUBSIZE    = 0.5              # sub‑aperture diameter (m)
GEO_HEIGHT = 35_786_000       # geostationary orbit altitude (m)

# Standard Golay‑9 relative positions (unit circle)
_GOLAY9_REL = np.array([
    [ 0.0,         0.0        ],
    [ 1.0,         0.0        ], [-1.0,         0.0        ],
    [ 0.5,  np.sqrt(3)/2], [-0.5,  np.sqrt(3)/2],
    [ 0.5, -np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2],
    [ 0.0,  np.sqrt(3)  ], [ 0.0, -np.sqrt(3)  ]
], dtype=np.float64)

# ======================================================================
# Pupil generators (physical units)
# ======================================================================

def _pupil_mask(N, radius_px):
    """Helper to create a circular pupil in a square grid."""
    x, y = np.mgrid[-N//2:N//2, -N//2:N//2]
    return (x**2 + y**2) <= radius_px**2

def full_aperture(N, diameter):
    """Monolithic circular pupil. `diameter` in metres."""
    pixel_scale = diameter / N
    radius_px = (diameter / 2) / pixel_scale
    pupil = _pupil_mask(N, radius_px)
    return pupil.astype(float)

def golay3(N, diameter):
    """
    Golay‑3: three equal sub‑apertures on an equilateral triangle.
    The triangle circumradius is set to 0.4 times the virtual radius
    so that the whole array fits within the virtual aperture.
    """
    pixel_scale = diameter / N
    r_sub_m = SUBSIZE / 2
    r_sub_px = r_sub_m / pixel_scale

    # circumradius in metres
    triangle_radius_m = 0.4 * (diameter / 2)
    triangle_radius_px = triangle_radius_m / pixel_scale

    pupil = np.zeros((N, N))
    cx, cy = N//2, N//2
    angles = np.deg2rad([0, 120, 240])
    xv = np.arange(N) - cx
    yv = np.arange(N) - cy
    xx, yy = np.meshgrid(xv, yv)
    for ang in angles:
        x0 = triangle_radius_px * np.cos(ang)
        y0 = triangle_radius_px * np.sin(ang)
        mask = (xx - x0)**2 + (yy - y0)**2 <= r_sub_px**2
        pupil[mask] = 1.0
    return pupil

def golay9(N, diameter):
    """
    Golay‑9: 9 sub‑apertures arranged according to the classic
    non‑redundant Golay pattern. The outermost elements are placed
    at 0.45 times the virtual radius so they stay inside.
    """
    pixel_scale = diameter / N
    r_sub_m = SUBSIZE / 2
    r_sub_px = r_sub_m / pixel_scale

    max_rel = np.max(np.sqrt(_GOLAY9_REL[:,0]**2 + _GOLAY9_REL[:,1]**2))
    scale_px = 0.45 * (diameter / 2) / max_rel / pixel_scale

    pupil = np.zeros((N, N))
    cx, cy = N//2, N//2
    xv = np.arange(N) - cx
    yv = np.arange(N) - cy
    xx, yy = np.meshgrid(xv, yv)
    for pos in _GOLAY9_REL:
        x0 = pos[0] * scale_px
        y0 = pos[1] * scale_px
        mask = (xx - x0)**2 + (yy - y0)**2 <= r_sub_px**2
        pupil[mask] = 1.0
    return pupil

# ======================================================================
# PSF and MTF computation (physical units)
# ======================================================================

def compute_psf(pupil, diameter, pad_factor=1):
    """
    PSF (intensity) from pupil, returned on a ground‑projected coordinate grid.
    - pupil   : 2D array (N×N)
    - diameter: physical diameter of the aperture (m)
    - pad_factor: zero‑padding factor (default 1 = no padding)
    Returns:
      psf   : 2D normalised intensity array
      theta : 1D array of angular coordinates (rad) for each pixel
    """
    N = pupil.shape[0]
    dx = diameter / N                   # pupil sampling interval (m)
    N_pad = N * pad_factor
    off = (N_pad - N) // 2
    padded = np.zeros((N_pad, N_pad))
    padded[off:off+N, off:off+N] = pupil

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(field)**2
    psf /= psf.sum()                    # normalise to unit energy

    # angular coordinate (rad) from spatial frequency
    freq_spatial = np.fft.fftshift(np.fft.fftfreq(N_pad, d=dx))   # cycles/m
    theta = freq_spatial * WAVELENGTH                               # rad
    return psf, theta

def compute_mtf(psf, dtheta):
    """
    MTF from a PSF sampled with angular step dtheta (rad).
    Returns mtf and the corresponding frequency axis (cycles/rad).
    """
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf = np.abs(otf)
    mtf /= mtf.max()
    N = len(psf)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dtheta))   # cycles/rad
    return mtf, freq

# ======================================================================
# Main comparison plot
# ======================================================================

def plot_apertures(show=True, savepath="apertures_comparison.png"):
    """3×3 figure: rows = Full, Golay‑3, Golay‑9; cols = pupil, PSF(ground,m), MTF."""
    N = 256

    pupils = {
        "Full":    (full_aperture(N, D_FULL),  D_FULL),
        "Golay‑3": (golay3(N, D_GOLAY),        D_GOLAY),
        "Golay‑9": (golay9(N, D_GOLAY),        D_GOLAY),
    }

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    for row, (name, (pupil, diam)) in enumerate(pupils.items()):
        # ---- Pupil (physical axes in metres) ----
        axp = fig.add_subplot(gs[row, 0])
        extent_pupil = [-diam/2, diam/2, -diam/2, diam/2]
        axp.imshow(pupil, cmap='gray', origin='lower', extent=extent_pupil)
        axp.set_title(f'{name} pupil\n(diameter {diam} m)')
        axp.set_xlabel('x (m)')
        axp.set_ylabel('y (m)')

        # ---- PSF (ground‑projected, metres) ----
        psf, theta = compute_psf(pupil, diam, pad_factor=2)
        # convert angle to ground distance
        ground = theta * GEO_HEIGHT        # metres
        axp = fig.add_subplot(gs[row, 1])
        psf_log = np.log10(np.maximum(psf, 1e-12))
        im = axp.imshow(psf_log, cmap='inferno', origin='lower',
                        extent=[ground[0], ground[-1], ground[0], ground[-1]])
        axp.set_title(f'{name} PSF (log)')
        axp.set_xlabel('Ground x (m)'); axp.set_ylabel('Ground y (m)')
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

        # ---- MTF (cycles/rad) ----
        dtheta = theta[1] - theta[0]
        mtf, freq = compute_mtf(psf, dtheta)
        axp = fig.add_subplot(gs[row, 2])
        axp.imshow(mtf, cmap='gray', origin='lower',
                   extent=[freq[0]/1e6, freq[-1]/1e6, freq[0]/1e6, freq[-1]/1e6],
                   vmin=0, vmax=1)
        axp.set_title(f'{name} MTF')
        axp.set_xlabel('f_x (Mcyc/rad)'); axp.set_ylabel('f_y (Mcyc/rad)')

    fig.suptitle('Optical Synthetic Aperture Comparison (engineering limits)',
                 fontsize=14, fontweight='bold')
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_single_aperture(type, N=256, diameter=None, **kwargs):
    """Quickly show pupil, PSF, MTF for one aperture type."""
    if diameter is None:
        if type == 'full':
            diameter = D_FULL
        else:
            diameter = D_GOLAY

    if type == 'full':
        pupil = full_aperture(N, diameter)
    elif type == 'golay3':
        pupil = golay3(N, diameter)
    elif type == 'golay9':
        pupil = golay9(N, diameter)
    else:
        raise ValueError("Unknown type: 'full', 'golay3', 'golay9'")

    # PSF
    psf, theta = compute_psf(pupil, diameter, pad_factor=2)
    ground = theta * GEO_HEIGHT

    # MTF
    dtheta = theta[1] - theta[0]
    mtf, freq = compute_mtf(psf, dtheta)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # pupil
    diam = diameter
    extent_pupil = [-diam/2, diam/2, -diam/2, diam/2]
    axes[0].imshow(pupil, cmap='gray', origin='lower', extent=extent_pupil)
    axes[0].set_title('Pupil function')
    axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')

    # PSF
    psf_log = np.log10(np.maximum(psf, 1e-12))
    im1 = axes[1].imshow(psf_log, cmap='inferno', origin='lower',
                         extent=[ground[0], ground[-1], ground[0], ground[-1]])
    axes[1].set_title('PSF (log)')
    axes[1].set_xlabel('Ground x (m)'); axes[1].set_ylabel('Ground y (m)')
    plt.colorbar(im1, ax=axes[1])

    # MTF
    axes[2].imshow(mtf, cmap='gray', origin='lower',
                   extent=[freq[0]/1e6, freq[-1]/1e6, freq[0]/1e6, freq[-1]/1e6],
                   vmin=0, vmax=1)
    axes[2].set_title('MTF')
    axes[2].set_xlabel('fx (Mcyc/rad)'); axes[2].set_ylabel('fy (Mcyc/rad)')

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Drawing aperture comparison (engineering limits)...")
    plot_apertures(show=True)
    print("Saved -> apertures_comparison.png")
