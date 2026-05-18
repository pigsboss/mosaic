"""
Generate 2D sea surface fields (SST & SSH) for three typical regimes:
  - calm    (no wind, red-noise SST, low-energy long waves)
  - langmuir (Langmuir circulation: discrete peaks + directional stripes)
  - turbulent (Kolmogorov cascade, broad spectrum)

Also provides anisotropic spectrum synthesis with separable radial and
angular components, and saves fields as .npy for subsequent observation
simulation.

Original functions (generate_multiscale_sst, generate_ssh_from_sst)
are kept for backward compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq, fftshift

# =============================================================================
# 1. Anisotropic field synthesis
# =============================================================================

def generate_anisotropic_field(
    nx=512, ny=512,
    lx=1.0, ly=1.0,
    alpha=2.5,          # radial spectral exponent (power spectrum ~ k^{-alpha})
    theta0=0.0,         # principle direction [rad], 0 = east
    s=1.0,              # directional concentration (0=isotropic, >1 peaked)
    peaks=None,         # list of (k_center, width_fraction, amplitude) for discrete peaks
    seed=42,
    isotropic_component=0.0,  # fraction of energy that is isotropic
):
    """
    Create a 2D field with a specified radial power law and directional
    distribution (cosine power).

    The power spectrum in polar coordinates is:
        P(k, theta) = k^{-alpha} * [ (1-iso)*D(theta) + iso ]
    where D(theta) = cos( (theta - theta0)/2 )^{2s}, normalised to unit mean
    over theta.

    Additional Gaussian peaks can be added to the spectrum.

    Returns
    -------
    field : 2D array (ny, nx) normalised to [0,1]
    """
    rng = np.random.default_rng(seed)

    # wavenumber grids (cycles/km)
    kx = fftfreq(nx, d=lx/nx)
    ky = fftfreq(ny, d=ly/ny)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_rad[0, 0] = 1e-12  # avoid singularity
    theta = np.arctan2(KY, KX)

    # angular distribution
    dtheta = theta - theta0
    D = np.cos(0.5 * dtheta) ** (2 * s)
    D = np.maximum(D, 0)   # clip negative values
    # normalise to mean=1 (preserve total energy)
    D = D / D.mean()

    # isotropic component
    iso_mask = np.ones_like(D)

    # combined directional component
    direction_component = (1 - isotropic_component) * D + isotropic_component * iso_mask

    # base radial spectrum
    P = k_rad ** (-alpha) * direction_component
    # remove DC
    P[0, 0] = 0.0

    # add discrete peaks
    if peaks is not None:
        for k0, width_frac, amp in peaks:
            sigma_k = k0 * width_frac
            gaussian_peak = amp * np.exp(-((k_rad - k0) ** 2) / (2 * sigma_k**2))
            P += gaussian_peak * direction_component

    # amplitude spectrum = sqrt(P)
    amplitude = np.sqrt(P)
    # random phases
    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    S = amplitude * noise

    # inverse FFT
    field = np.real(ifft2(S))

    # normalise to [0,1]
    field = (field - field.min()) / (field.max() - field.min())
    return field


# =============================================================================
# 2. State-specific SST and SSH generators
# =============================================================================

state_params = {
    "calm": {
        "sst": {
            "alpha": 2.5,   # reddish noise
            "theta0": 0.0,
            "s": 0.2,       # almost isotropic
            "isotropic_component": 0.8,
            "peaks": None,
        },
        "ssh": {
            "alpha": 5.0,   # very steep, energy only at lowest wavenumbers
            "theta0": 0.0,
            "s": 0.2,
            "isotropic_component": 0.8,
            "peaks": None,
        },
    },
    "langmuir": {
        "sst": {
            "alpha": 2.0,
            "theta0": 0.0,          # east wind
            "s": 2.0,               # moderately directional stripes
            "isotropic_component": 0.3,
            "peaks": [
                # Langmuir circulation spacing ~30 m => wavenumber ~0.033 cycles/m = 33 cycles/km
                (33.0, 0.05, 10.0),   # fundamental
                (66.0, 0.05, 3.0),    # 1st harmonic
                (99.0, 0.05, 1.0),    # 2nd harmonic (weak)
                (100.0, 0.03, 8.0),   # 10‑m scale peak (NEW)
            ],
        },
        "ssh": {
            "alpha": 3.5,           # less steep than calm, high-frequency tail raises
            "theta0": 0.0,
            "s": 3.0,               # directional wind waves
            "isotropic_component": 0.2,
            "peaks": [
                (33.0, 0.05, 5.0),    # same Langmuir scale but weaker in SSH
                (100.0, 0.03, 4.0),   # 10‑m scale peak (NEW)
            ],
        },
    },
    "turbulent": {
        "sst": {
            "alpha": 1.667,         # -5/3 Kolmogorov spectrum
            "theta0": 0.0,
            "s": 0.5,               # weakly directional
            "isotropic_component": 0.6,
            "peaks": [
                (100.0, 0.04, 5.0),   # 10‑m scale peak (NEW)
            ],
        },
        "ssh": {
            "alpha": 2.5,           # shallower slope, more high-frequency energy
            "theta0": 0.0,
            "s": 1.0,
            "isotropic_component": 0.4,
            "peaks": [
                (100.0, 0.04, 3.0),   # 10‑m scale peak (NEW)
            ],
        },
    },
}


def generate_state_sst_ssh(state="calm", nx=512, ny=512, lx=1.0, ly=1.0, seed=42):
    """
    Generate SST and SSH fields for a given sea state.

    Parameters
    ----------
    state : str
        One of 'calm', 'langmuir', 'turbulent'.
    nx, ny : int
        Grid size (default 1024x1024).
    lx, ly : float
        Domain size in km (default 1 km x 1 km).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sst, ssh : 2D numpy arrays
    """
    if state not in state_params:
        raise ValueError(f"Unknown state '{state}'. Choose from {list(state_params.keys())}")

    cfg_sst = state_params[state]["sst"]
    cfg_ssh = state_params[state]["ssh"]

    rng = np.random.default_rng(seed)

    sst = generate_anisotropic_field(
        nx=nx, ny=ny, lx=lx, ly=ly,
        **cfg_sst,
        seed=rng.integers(1e6),
    )

    ssh = generate_anisotropic_field(
        nx=nx, ny=ny, lx=lx, ly=ly,
        **cfg_ssh,
        seed=rng.integers(1e6),
    )

    # ---- post-processing for SSH: scale to realistic heights ----
    # SSH from parametric model has range ~[0,1]. We scale to about ±0.05 m
    ssh = (ssh - 0.5) * 0.1   # range approx -0.05 to +0.05 m

    return sst, ssh


# =============================================================================
# 3. Original backward‑compatible functions (unchanged)
# =============================================================================

def generate_multiscale_sst(
    nx=512, ny=512,
    lx=1.0, ly=1.0,
    spectral_exponent=2.5,
    seed=42,
):
    """Legacy function. Creates an isotropic SST field with a power‑law spectrum."""
    rng = np.random.default_rng(seed)
    kx = fftfreq(nx, d=lx/nx)
    ky = fftfreq(ny, d=ly/ny)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_rad[0, 0] = 1.0
    amplitude = k_rad ** (-spectral_exponent/2)
    amplitude[0, 0] = 0.0
    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    S = amplitude * noise
    sst = np.real(ifft2(S))
    sst = (sst - sst.min()) / (sst.max() - sst.min())
    return sst


def generate_ssh_from_sst(sst, expansion_scale=0.2, wave_amplitude=0.05, wave_seed=None):
    """Legacy function. Derives SSH from SST via thermal expansion + noise."""
    sst_anomaly = sst - np.mean(sst)
    ssh = expansion_scale * sst_anomaly
    rng = np.random.default_rng(wave_seed) if wave_seed else np.random.default_rng()
    roughness = wave_amplitude * rng.normal(size=sst.shape)
    ssh += roughness
    return ssh


# =============================================================================
# 4. Utility: radial power spectrum, angular anisotropy and 2D PSD
# =============================================================================

def radial_power_spectrum(field, dx, dy):
    """2D radial average power spectrum."""
    ny, nx = field.shape
    F = fft2(field)
    psd2d = np.abs(F)**2
    psd2d_shifted = fftshift(psd2d)

    kx = fftshift(fftfreq(nx, d=dx))
    ky = fftshift(fftfreq(ny, d=dy))
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
            idx = np.digitize(kr, bins) - 1
            if 0 <= idx < n_bins:
                radial_psd[idx] += psd2d_shifted[i, j]
                counts[idx] += 1
    valid = counts > 0
    radial_psd[valid] /= counts[valid]
    k_center = 0.5 * (bins[1:] + bins[:-1])
    return k_center[valid], radial_psd[valid]


# =============================================================================
# NEW: moment‑based anisotropy (structure tensor)
# =============================================================================

def moment_anisotropy(field, dx, dy, n_bins=50):
    """
    Compute spectral moment tensor for each radial bin and derive
    anisotropy degree A = (λ1 - λ2) / (λ1 + λ2) and principal orientation θ.

    Parameters
    ----------
    field : 2D ndarray
    dx, dy : float
        Grid spacing in the same units used for kx, ky (cycles/km when dx in km).
    n_bins : int

    Returns
    -------
    k_centers : 1D array (n_bins,)
    A : 1D array (n_bins,)         anisotropy degree, 0 = isotropic, 1 = fully aligned
    theta : 1D array (n_bins,)     principal orientation [rad], 0 = along x
    """
    ny, nx = field.shape
    F = np.fft.fft2(field)
    psd2d = np.abs(F) ** 2

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    A = np.zeros(n_bins)
    theta = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_rad >= bins[i]) & (k_rad < bins[i+1])
        if not np.any(mask):
            A[i] = 0.0
            theta[i] = 0.0
            continue
        p_vals = psd2d[mask]
        kx_vals = KX[mask]
        ky_vals = KY[mask]
        total = np.sum(p_vals)
        if total == 0:
            A[i] = 0.0
            theta[i] = 0.0
            continue
        # spectral moments (structure tensor)
        m20 = np.sum(p_vals * kx_vals ** 2) / total
        m11 = np.sum(p_vals * kx_vals * ky_vals) / total
        m02 = np.sum(p_vals * ky_vals ** 2) / total

        # eigenvalues of [[m20, m11], [m11, m02]]
        trace = m20 + m02
        det = m20 * m02 - m11 * m11
        disc = np.sqrt(trace ** 2 - 4 * det)
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)

        # avoid division by zero
        if lambda1 + lambda2 > 0:
            A[i] = (lambda1 - lambda2) / (lambda1 + lambda2)
        else:
            A[i] = 0.0

        # principal direction
        if (m20 - m02) == 0 and m11 == 0:
            theta[i] = 0.0
        else:
            theta[i] = 0.5 * np.arctan2(2 * m11, m20 - m02)

    return k_centers, A, theta


# =============================================================================
# NEW: orthogonal (longitudinal / transverse) power spectra
# =============================================================================

def orthogonal_power_spectra(field, dx, dy):
    """
    Compute 1D power spectra by integrating the 2D PSD along the orthogonal axis.
    Returns:
        kx, power_x : along x (longitudinal)
        ky, power_y : along y (transverse)
    Both power arrays are not normalised.
    """
    ny, nx = field.shape
    F = fft2(field)
    psd2d = np.abs(F) ** 2

    kx = fftfreq(nx, d=dx)      # cycles/km
    ky = fftfreq(ny, d=dy)

    # integrate along y → power as function of kx
    power_x = np.sum(psd2d, axis=0)
    # integrate along x → power as function of ky
    power_y = np.sum(psd2d, axis=1)
    return kx, power_x, ky, power_y


# =============================================================================
# (original angular anisotropy kept unchanged)
# =============================================================================

def angular_anisotropy(field, dx, dy, n_bins=50):
    """
    Calculate angular concentration index per radial wavenumber.
    Returns:
        k_centers : 1D array of wavenumber values
        anisotropy : 1D array of concentration values (0 = isotropic, 1 = fully aligned)
    """
    ny, nx = field.shape
    F = fft2(field)
    psd2d = np.abs(F)**2
    kx = fftfreq(nx, d=dx)
    ky = fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    theta = np.arctan2(KY, KX)

    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])
    anisotropy = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (k_rad >= bins[i]) & (k_rad < bins[i+1])
        if np.any(mask):
            p_vals = psd2d[mask]
            t_vals = theta[mask]
            complex_sum = np.sum(p_vals * np.exp(1j * t_vals))
            total_power = np.sum(p_vals)
            if total_power > 0:
                R = np.abs(complex_sum) / total_power
            else:
                R = 0.0
            anisotropy[i] = R

    return k_centers, anisotropy


def plot_2d_psd(field, dx, dy, ax, title=""):
    """Plot 2D log power spectrum (kx, ky) on the given Axes."""
    F = np.fft.fft2(field)
    psd2d = np.abs(F)**2
    psd2d_shifted = np.fft.fftshift(psd2d)
    kx = np.fft.fftshift(np.fft.fftfreq(field.shape[1], d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(field.shape[0], d=dy))
    ax.imshow(np.log10(psd2d_shifted + 1e-12),
              extent=[kx[0], kx[-1], ky[0], ky[-1]],
              origin='lower', cmap='inferno', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('kx (cycles/km)')
    ax.set_ylabel('ky (cycles/km)')


# =============================================================================
# 5. State spectra plotting (three figures)
# =============================================================================

def plot_state_spectra(states=("calm", "langmuir", "turbulent"),
                       lx=1.0, ly=1.0, save=True, show=True):
    """
    Generate three figures:
      - Figure 1: SST fields (1×3)
      - Figure 2: SSH fields (1×3)
      - Figure 3: 2×2 grid of radial power spectra and orthogonal spectra
    Also saves all fields as .npy.
    """
    # ----------------------------------------------------------
    # Generate data & save .npy
    # ----------------------------------------------------------
    data = {}
    for state in states:
        sst, ssh = generate_state_sst_ssh(state, lx=lx, ly=ly)
        np.save(f"{state}_sst.npy", sst)
        np.save(f"{state}_ssh.npy", ssh)
        data[state] = (sst, ssh)

    extent_img = [0, lx, 0, ly]   # km
    # 使用实际生成的场大小计算网格步长
    dx = lx / data[states[0]][0].shape[1]
    dy = ly / data[states[0]][0].shape[0]

    # ----------------------------------------------------------
    # Figure 1: SST comparison
    # ----------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    for idx, state in enumerate(states):
        sst, _ = data[state]
        im = axes1[idx].imshow(sst, origin='lower', cmap='RdYlBu_r',
                               extent=extent_img, vmin=0, vmax=1)
        axes1[idx].set_title(f'{state.capitalize()} SST')
        axes1[idx].set_xlabel('x (km)')
        axes1[idx].set_ylabel('y (km)')
    fig1.colorbar(im, ax=axes1.ravel(), fraction=0.02, pad=0.04, label='Normalized SST')
    fig1.suptitle('Sea Surface Temperature: Three Sea States', fontweight='bold')
    fig1.savefig("state_sst_comparison.png", dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig1)

    # ----------------------------------------------------------
    # Figure 2: SSH comparison
    # ----------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    vmin_all = min(data[s][1].min() for s in states)
    vmax_all = max(data[s][1].max() for s in states)
    for idx, state in enumerate(states):
        _, ssh = data[state]
        im = axes2[idx].imshow(ssh, origin='lower', cmap='coolwarm',
                               extent=extent_img, vmin=vmin_all, vmax=vmax_all)
        axes2[idx].set_title(f'{state.capitalize()} SSH')
        axes2[idx].set_xlabel('x (km)')
        axes2[idx].set_ylabel('y (km)')
        axes2[idx].set_facecolor('gray')
    fig2.colorbar(im, ax=axes2.ravel(), fraction=0.02, pad=0.04, label='Height (m)')
    fig2.suptitle('Sea Surface Height Anomaly: Three Sea States', fontweight='bold')
    fig2.savefig("state_ssh_comparison.png", dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig2)

    # ----------------------------------------------------------
    # Figure 3: 3×2 panel (isotropic, anisotropy, orientation)
    # ----------------------------------------------------------
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 15))
    ((ax_rad_sst, ax_rad_ssh),
     (ax_aniso_sst, ax_aniso_ssh),
     (ax_orient_sst, ax_orient_ssh)) = axes3
    colors = {'calm': 'blue', 'langmuir': 'green', 'turbulent': 'red'}

    # ---- Row 1, left: SST radial power spectrum (isotropic) ----
    for state in states:
        sst, _ = data[state]
        k_sst, psd_sst = radial_power_spectrum(sst, dx, dy)
        ax_rad_sst.loglog(k_sst, psd_sst / psd_sst[0],
                          color=colors[state], label=state)
    ax_rad_sst.set_xlabel('Wavenumber (cycles/km)')
    ax_rad_sst.set_ylabel('Normalized Power')
    ax_rad_sst.set_title('SST Radial Power Spectra')
    ax_rad_sst.legend()
    ax_rad_sst.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- Row 1, right: SSH radial power spectrum ----
    for state in states:
        _, ssh = data[state]
        k_ssh, psd_ssh = radial_power_spectrum(ssh, dx, dy)
        ax_rad_ssh.loglog(k_ssh, psd_ssh / psd_ssh[0],
                          color=colors[state], label=state)
    ax_rad_ssh.set_xlabel('Wavenumber (cycles/km)')
    ax_rad_ssh.set_ylabel('Normalized Power')
    ax_rad_ssh.set_title('SSH Radial Power Spectra')
    ax_rad_ssh.legend()
    ax_rad_ssh.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- Row 2, left: SST anisotropy degree A(k) ----
    for state in states:
        sst, _ = data[state]
        k_mom, A_sst, _ = moment_anisotropy(sst, dx, dy)
        ax_aniso_sst.semilogx(k_mom, A_sst, color=colors[state], label=state)
    ax_aniso_sst.set_xlabel('Wavenumber (cycles/km)')
    ax_aniso_sst.set_ylabel('Anisotropy A = (λ₁-λ₂)/(λ₁+λ₂)')
    ax_aniso_sst.set_title('SST Anisotropy Degree')
    ax_aniso_sst.legend()
    ax_aniso_sst.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- Row 2, right: SSH anisotropy degree A(k) ----
    for state in states:
        _, ssh = data[state]
        k_mom, A_ssh, _ = moment_anisotropy(ssh, dx, dy)
        ax_aniso_ssh.semilogx(k_mom, A_ssh, color=colors[state], label=state)
    ax_aniso_ssh.set_xlabel('Wavenumber (cycles/km)')
    ax_aniso_ssh.set_ylabel('Anisotropy A = (λ₁-λ₂)/(λ₁+λ₂)')
    ax_aniso_ssh.set_title('SSH Anisotropy Degree')
    ax_aniso_ssh.legend()
    ax_aniso_ssh.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- Row 3, left: SST principal orientation θ(k) ----
    for state in states:
        sst, _ = data[state]
        k_mom, _, theta_sst = moment_anisotropy(sst, dx, dy)
        theta_deg = np.degrees(theta_sst)          # convert to degrees
        ax_orient_sst.semilogx(k_mom, theta_deg, color=colors[state], label=state)
    ax_orient_sst.set_xlabel('Wavenumber (cycles/km)')
    ax_orient_sst.set_ylabel('Principal Orientation (deg)')
    ax_orient_sst.set_title('SST Principal Orientation')
    ax_orient_sst.legend()
    ax_orient_sst.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- Row 3, right: SSH principal orientation θ(k) ----
    for state in states:
        _, ssh = data[state]
        k_mom, _, theta_ssh = moment_anisotropy(ssh, dx, dy)
        theta_deg = np.degrees(theta_ssh)
        ax_orient_ssh.semilogx(k_mom, theta_deg, color=colors[state], label=state)
    ax_orient_ssh.set_xlabel('Wavenumber (cycles/km)')
    ax_orient_ssh.set_ylabel('Principal Orientation (deg)')
    ax_orient_ssh.set_title('SSH Principal Orientation')
    ax_orient_ssh.legend()
    ax_orient_ssh.grid(True, which='both', linestyle='--', alpha=0.5)

    fig3.suptitle('Spectral Characteristics (Isotropic, Anisotropy, Orientation)', fontweight='bold')
    fig3.tight_layout()
    fig3.savefig("state_spectra_curves.png", dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig3)


# =============================================================================
# 6. Legacy plotting functions (unchanged)
# =============================================================================

def plot_fields(sst, ssh, lx=1.0, ly=1.0,
                savepath_sst="sst_field.png", savepath_ssh="ssh_field.png",
                show=True):
    """(unchanged)"""
    extent = [0, lx, 0, ly]
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(sst, origin='lower', cmap='RdYlBu_r', extent=extent, vmin=0, vmax=1)
    ax1.set_title('Sea Surface Temperature (multiscale)', fontweight='bold')
    ax1.set_xlabel('x (km)'); ax1.set_ylabel('y (km)')
    plt.colorbar(im1, ax=ax1, label='Normalized SST')
    fig1.tight_layout()
    fig1.savefig(savepath_sst, dpi=200)
    if show: plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(ssh, origin='lower', cmap='coolwarm', extent=extent)
    ax2.set_title('Sea Surface Height anomaly', fontweight='bold')
    ax2.set_xlabel('x (km)'); ax2.set_ylabel('y (km)')
    plt.colorbar(im2, ax=ax2, label='Height (m)')
    fig2.tight_layout()
    fig2.savefig(savepath_ssh, dpi=200)
    if show: plt.show()
    plt.close(fig2)


def plot_power_spectrum(sst, ssh, lx=1.0, ly=1.0,
                        savepath="spectra_comparison.png", show=True):
    """(unchanged)"""
    dx = lx / sst.shape[1]; dy = ly / sst.shape[0]
    k_sst, psd_sst = radial_power_spectrum(sst, dx, dy)
    k_ssh, psd_ssh = radial_power_spectrum(ssh, dx, dy)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_sst, psd_sst / psd_sst[0], 'b-', label='SST')
    ax.loglog(k_ssh, psd_ssh / psd_ssh[0], 'r-', label='SSH')
    ref_k = k_sst[k_sst > 0]
    ax.loglog(ref_k, 1e-3 * ref_k**(-2.0) / 1e-3, 'k--', label=r'$\propto k^{-2}$')
    ax.set_xlabel('Wavenumber (cycles/km)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Spatial Power Spectra of Sea Surface Variables')
    ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    if show: plt.show()
    plt.close(fig)


# =============================================================================
# 7. Main
# =============================================================================
if __name__ == "__main__":
    print("Generating three sea state fields and figures...")
    plot_state_spectra(states=["calm", "langmuir", "turbulent"],
                       lx=1.0, ly=1.0, save=True, show=True)
    print("Figures saved: state_sst_comparison.png, state_ssh_comparison.png, state_spectra_curves.png")
    print("Fields saved as .npy files.")
