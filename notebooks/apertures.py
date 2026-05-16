"""
Optical synthetic aperture pupil functions, PSF and MTF.
- Full aperture (monolithic)
- Golay‑3 (3 sub‑apertures)
- Golay‑9 (9 sub‑apertures, non‑redundant)

Generates a comparison plot saved as 'apertures_comparison.png'
and can be used interactively in Jupyter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ======================================================================
# Pupil generators
# ======================================================================

def full_aperture(N, radius):
    """Monolithic circular pupil (1 = open)."""
    x, y = np.mgrid[-N//2:N//2, -N//2:N//2]
    pupil = (x**2 + y**2) <= radius**2
    return pupil.astype(float)

def golay3(N, radius, spacing):
    """
    Golay‑3: three equal apertures on an equilateral triangle.
    'spacing' = distance from triangle centre to each sub‑aperture centre.
    """
    pupil = np.zeros((N, N))
    cx, cy = N//2, N//2
    angles = np.deg2rad([0, 120, 240])
    x = np.arange(N) - cx
    y = np.arange(N) - cy
    xx, yy = np.meshgrid(x, y)
    for ang in angles:
        x0 = spacing * np.cos(ang)
        y0 = spacing * np.sin(ang)
        mask = (xx - x0)**2 + (yy - y0)**2 <= radius**2
        pupil[mask] = 1.0
    return pupil

def golay9(N, radius, positions):
    """
    Golay‑9: 9 sub‑apertures at given (dx, dy) offsets (in pixels)
    relative to centre.  `positions` is a (9,2) array.
    """
    pupil = np.zeros((N, N))
    cx, cy = N//2, N//2
    x = np.arange(N) - cx
    y = np.arange(N) - cy
    xx, yy = np.meshgrid(x, y)
    for dx, dy in positions:
        mask = (xx - dx)**2 + (yy - dy)**2 <= radius**2
        pupil[mask] = 1.0
    return pupil

# ======================================================================
# PSF and MTF computation
# ======================================================================

def compute_psf(pupil, pad_factor=1):
    """PSF (intensity) from pupil.  Pad to avoid wrap‑around."""
    N = pupil.shape[0]
    N_pad = N * pad_factor
    off = (N_pad - N) // 2
    padded = np.zeros((N_pad, N_pad))
    padded[off:off+N, off:off+N] = pupil

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    psf = np.abs(field)**2
    psf /= psf.sum()                     # normalise to unit energy
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
    """3×3 figure: rows = Full, Golay‑3, Golay‑9; cols = pupil, PSF(log), MTF."""
    N = 256
    r_full = 40
    r_sub = 12
    spacing3 = 30

    # Golay‑9 positions (scaled for N=256)
    scale = 25
    golay9_pos = np.array([
        [0, 0],
        [scale, 0], [-scale, 0],
        [scale/2, scale*np.sqrt(3)/2], [-scale/2, scale*np.sqrt(3)/2],
        [scale/2, -scale*np.sqrt(3)/2], [-scale/2, -scale*np.sqrt(3)/2],
        [0, scale*np.sqrt(3)], [0, -scale*np.sqrt(3)]
    ]) * 0.9  # slightly compressed to fit inside FOV

    pupils = {
        "Full": full_aperture(N, r_full),
        "Golay‑3": golay3(N, r_sub, spacing3),
        "Golay‑9": golay9(N, r_sub, golay9_pos),
    }

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    for row, (name, pupil) in enumerate(pupils.items()):
        # ---- Pupil ----
        axp = fig.add_subplot(gs[row, 0])
        axp.imshow(pupil, cmap='gray', origin='lower')
        axp.set_title(f'{name} pupil')
        axp.axis('off')

        # ---- PSF ----
        psf, freq = compute_psf(pupil, pad_factor=2)
        axp = fig.add_subplot(gs[row, 1])
        psf_log = np.log10(np.maximum(psf, 1e-12))
        im = axp.imshow(psf_log, cmap='inferno', origin='lower',
                        extent=[freq[0], freq[-1], freq[0], freq[-1]])
        axp.set_title(f'{name} PSF (log)')
        axp.set_xlabel('$f_x$ (cy/px)'); axp.set_ylabel('$f_y$ (cy/px)')
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

        # ---- MTF ----
        mtf = compute_mtf(psf)
        axp = fig.add_subplot(gs[row, 2])
        im = axp.imshow(mtf, cmap='gray', origin='lower',
                        extent=[freq[0], freq[-1], freq[0], freq[-1]],
                        vmin=0, vmax=1)
        axp.set_title(f'{name} MTF')
        axp.set_xlabel('$f_x$ (cy/px)'); axp.set_ylabel('$f_y$ (cy/px)')
        plt.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

    fig.suptitle('Optical Synthetic Aperture Comparison', fontsize=14, fontweight='bold')
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_single_aperture(type, N=256, **kwargs):
    """Quickly show pupil, PSF, MTF for one aperture type."""
    if type == 'full':
        pupil = full_aperture(N, kwargs.get('radius', 40))
    elif type == 'golay3':
        pupil = golay3(N, kwargs.get('radius', 12), kwargs.get('spacing', 30))
    elif type == 'golay9':
        scale = kwargs.get('scale', 25)
        pos = np.array([
            [0, 0],
            [scale, 0], [-scale, 0],
            [scale/2, scale*np.sqrt(3)/2], [-scale/2, scale*np.sqrt(3)/2],
            [scale/2, -scale*np.sqrt(3)/2], [-scale/2, -scale*np.sqrt(3)/2],
            [0, scale*np.sqrt(3)], [0, -scale*np.sqrt(3)]
        ]) * 0.9
        pupil = golay9(N, kwargs.get('radius', 12), pos)
    else:
        raise ValueError("Unknown type: choose 'full', 'golay3', 'golay9'")

    psf, freq = compute_psf(pupil, pad_factor=2)
    mtf = compute_mtf(psf)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(pupil, cmap='gray', origin='lower')
    axes[0].set_title('Pupil function')
    axes[0].axis('off')

    psf_log = np.log10(np.maximum(psf, 1e-12))
    im1 = axes[1].imshow(psf_log, cmap='inferno', origin='lower',
                         extent=[freq[0], freq[-1], freq[0], freq[-1]])
    axes[1].set_title('PSF (log)')
    axes[1].set_xlabel('fx (cy/px)'); axes[1].set_ylabel('fy (cy/px)')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(mtf, cmap='gray', origin='lower',
                         extent=[freq[0], freq[-1], freq[0], freq[-1]],
                         vmin=0, vmax=1)
    axes[2].set_title('MTF')
    axes[2].set_xlabel('fx (cy/px)'); axes[2].set_ylabel('fy (cy/px)')
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Drawing aperture comparison...")
    plot_apertures(show=True)
    print("Saved -> apertures_comparison.png")
