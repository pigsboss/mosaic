"""
Optical synthetic aperture pupil functions, PSF and MTF.
- Full aperture (monolithic)
- Golay‑3 (3 sub‑apertures)
- Golay‑9 (9 sub‑apertures, non‑redundant)

Engineering constraints:
  - Single aperture diameter = 4 m -> cutoff ~400 kcyc/rad (at 10 µm)
  - Golay‑3 and Golay‑9 share a virtual aperture of 10 m -> cutoff ~1 Mcyc/rad
  - Sub‑aperture size: 0.5 m

Generates a comparison plot saved as 'apertures_comparison.png'
and can be used interactively in Jupyter.
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
# PSF and MTF computation (physical frequency axis)
# ======================================================================

def compute_psf(pupil, pad_factor=1, f_cut=None):
    """PSF (intensity) from pupil. If `f_cut` (cycles/rad) is given,
    the frequency axis is directly labelled in those units."""
    N = pupil.shape[0]
    N_pad = N * pad_factor
    off = (N_pad - N) // 2
    padded = np.zeros((N_pad, N_pad))
    padded[off:off+N, off:off+N] = pupil

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(field)**2
    psf /= psf.sum()                     # normalise to unit energy

    if f_cut is not None:
        freq = np.linspace(-f_cut, f_cut, N_pad, endpoint=False)
    else:
        freq = np.fft.fftshift(np.fft.fftfreq(N_pad, d=1.0))
    return psf, freq

def compute_mtf(psf):
    """MTF from a PSF (absolute value of OTF)."""
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf = np.abs(otf)
    mtf /= mtf.max()
    return mtf

# ======================================================================
# Main comparison plot
# ======================================================================

def plot_apertures(show=True, savepath="apertures_comparison.png"):
    """3×3 figure: rows = Full, Golay‑3, Golay‑9; cols = pupil, PSF(log), MTF.
    Frequency axes are in cycles/rad (Mcyc/rad)."""
    N = 256
    # Cutoff frequencies
    fcut_full  = D_FULL  / WAVELENGTH   # 4e6 cyc/rad = 0.4 Mcyc/rad
    fcut_golay = D_GOLAY / WAVELENGTH   # 1e6 cyc/rad = 1.0 Mcyc/rad
    fcut_max   = fcut_golay             # common axis limit (use the larger one)

    pupils = {
        "Full":    full_aperture(N, D_FULL),
        "Golay‑3": golay3(N, D_GOLAY),
        "Golay‑9": golay9(N, D_GOLAY),
    }
    fcuts = {
        "Full":    fcut_full,
        "Golay‑3": fcut_golay,
        "Golay‑9": fcut_golay,
    }

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    for row, (name, pupil) in enumerate(pupils.items()):
        # ---- Pupil ----
        axp = fig.add_subplot(gs[row, 0])
        diam = D_FULL if name == "Full" else D_GOLAY
        extent_pupil = [-diam/2, diam/2, -diam/2, diam/2]
        axp.imshow(pupil, cmap='gray', origin='lower', extent=extent_pupil)
        axp.set_title(f'{name} pupil\n(cutoff {fcuts[name]/1e6:.2f} Mcyc/rad)')
        axp.set_xlabel('x (m)')
        axp.set_ylabel('y (m)')

        # ---- PSF ----
        psf, freq = compute_psf(pupil, pad_factor=2, f_cut=fcut_max)
        axp = fig.add_subplot(gs[row, 1])
        psf_log = np.log10(np.maximum(psf, 1e-12))
        im = axp.imshow(psf_log, cmap='inferno', origin='lower',
                        extent=[freq[0]/1e6, freq[-1]/1e6, freq[0]/1e6, freq[-1]/1e6])
        axp.set_title(f'{name} PSF (log)')
        axp.set_xlabel('f_x (Mcyc/rad)'); axp.set_ylabel('f_y (Mcyc/rad)')
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

        # ---- MTF ----
        mtf = compute_mtf(psf)
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

    f_cut = diameter / WAVELENGTH
    psf, freq = compute_psf(pupil, pad_factor=2, f_cut=f_cut)
    mtf = compute_mtf(psf)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(pupil, cmap='gray', origin='lower')
    axes[0].set_title('Pupil function')
    axes[0].axis('off')

    psf_log = np.log10(np.maximum(psf, 1e-12))
    im1 = axes[1].imshow(psf_log, cmap='inferno', origin='lower',
                         extent=[freq[0]/1e6, freq[-1]/1e6, freq[0]/1e6, freq[-1]/1e6])
    axes[1].set_title('PSF (log)')
    axes[1].set_xlabel('fx (Mcyc/rad)'); axes[1].set_ylabel('fy (Mcyc/rad)')
    plt.colorbar(im1, ax=axes[1])

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
