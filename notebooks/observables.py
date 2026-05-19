"""
Observation pipeline: Apply synthetic aperture systems to SST and SSH fields
that were pre‑computed and saved as .npy files (by seasurface.py).

Generates three comprehensive comparison figures:
  - all_states_sst_observations.png   (rows: True, Full, Golay‑3, Golay‑9;
                                       cols: calm, langmuir, turbulent)
  - all_states_ssh_observations.png   (same layout for SSH)
  - all_states_spectral_comparison.png (3×2 panels: radial power,
                                       anisotropy degree, orientation;
                                       colours = sea state,
                                       line styles = aperture)

The simulation now uses a visibility‑based (OTF) approach: the scene's Fourier
transform is multiplied by the aperture's optical transfer function computed
directly on the scene's angular‑frequency grid.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import seasurface, apertures
reload(apertures)
reload(seasurface)
from apertures import D_FULL, D_GOLAY, SUBSIZE, WAVELENGTH, GEO_HEIGHT, _GOLAY9_REL

# ----------------------------------------------------------------------
# Radial power spectrum (with optional mask)
# ----------------------------------------------------------------------

def radial_power_spectrum(field, dx, dy, mask=None):
    """Compute 2D radial average power spectrum within the mask region."""
    ny, nx = field.shape
    F = np.fft.fft2(field)
    psd2d = np.abs(F) ** 2

    if mask is not None:
        psd2d = np.where(mask, psd2d, 0.0)

    psd2d_shifted = np.fft.fftshift(psd2d)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    if mask is not None:
        mask_shifted = np.fft.fftshift(mask)
        valid = mask_shifted
    else:
        valid = np.ones_like(k_rad, dtype=bool)

    # 固定使用全图最大波数，保证所有曲线的 bins 一致
    k_max = np.max(k_rad)
    n_bins = 100
    bins = np.linspace(0, k_max, n_bins + 1)
    radial_psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    # 只遍历掩膜内像素
    idx, idy = np.where(valid)
    for i, j in zip(idx, idy):
        kr = k_rad[i, j]
        bin_idx = np.digitize(kr, bins) - 1
        if 0 <= bin_idx < n_bins:
            radial_psd[bin_idx] += psd2d_shifted[i, j]
            counts[bin_idx] += 1

    # 平均；无数据的 bin 保持 0
    nonzero = counts > 0
    radial_psd[nonzero] /= counts[nonzero]

    k_center = 0.5 * (bins[1:] + bins[:-1])
    return k_center, radial_psd


# ----------------------------------------------------------------------
# Local moment_anisotropy (with optional mask)
# ----------------------------------------------------------------------

def moment_anisotropy(field, dx, dy, n_bins=50, mask=None):
    """Compute spectral moment tensor per radial bin with optional mask."""
    ny, nx = field.shape
    F = np.fft.fft2(field)
    psd2d = np.abs(F) ** 2

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX ** 2 + KY ** 2)

    if mask is not None:
        psd2d = np.where(mask, psd2d, 0.0)

    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    A = np.zeros(n_bins)
    theta = np.zeros(n_bins)

    for i in range(n_bins):
        mask_bin = (k_rad >= bins[i]) & (k_rad < bins[i+1])
        if not np.any(mask_bin):
            A[i] = 0.0
            theta[i] = 0.0
            continue
        p_vals = psd2d[mask_bin]
        kx_vals = KX[mask_bin]
        ky_vals = KY[mask_bin]
        total = np.sum(p_vals)
        if total == 0:
            A[i] = 0.0
            theta[i] = 0.0
            continue
        m20 = np.sum(p_vals * kx_vals ** 2) / total
        m11 = np.sum(p_vals * kx_vals * ky_vals) / total
        m02 = np.sum(p_vals * ky_vals ** 2) / total

        trace = m20 + m02
        det = m20 * m02 - m11 * m11
        disc = np.sqrt(trace ** 2 - 4 * det)
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)

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


# ----------------------------------------------------------------------
# OTF construction (visibility‑based simulation)
# ----------------------------------------------------------------------

def compute_otf_for_aperture(aperture_type, shape, dx_m, dy_m, H_m, wavelength_m):
    """
    Build the optical transfer function (OTF) for a given aperture type
    directly on the angular‑frequency grid that matches the scene's
    physical sampling.

    Parameters
    ----------
    aperture_type : str
        'full', 'golay3', or 'golay9'.
    shape : tuple (ny, nx)
        Number of pixels in y and x.
    dx_m, dy_m : float
        Spatial sampling interval in metres.
    H_m : float
        Altitude of the observer (metres).
    wavelength_m : float
        Observation wavelength (metres).

    Returns
    -------
    otf : 2D complex ndarray (ny, nx)
        OTF normalised so that the zero‑frequency value is 1.
        The array is in the order expected by np.fft.fft2 (DC at (0,0)).
    mask : 2D bool ndarray (ny, nx)
        True where |OTF| > 0.01 (relative threshold).
    """
    ny, nx = shape
    # spatial frequencies (cycles/metre)
    fx = np.fft.fftfreq(nx, d=dx_m)
    fy = np.fft.fftfreq(ny, d=dy_m)
    FX, FY = np.meshgrid(fx, fy)
    # angular frequencies (cycles/rad)
    omega_x = FX * H_m
    omega_y = FY * H_m
    omega_rad = np.sqrt(omega_x ** 2 + omega_y ** 2)

    # ------------------------------------------------------------------
    # Coherent transfer function (CTF) mask in angular‑frequency space
    # ------------------------------------------------------------------
    if aperture_type == 'full':
        # For a circular pupil of diameter D, the coherent cutoff is D/(2λ)
        radius_ctf = (D_FULL / 2.0) / wavelength_m
        ctf = (omega_rad <= radius_ctf).astype(np.float64)

    elif aperture_type == 'golay3':
        radius_sub = (SUBSIZE / 2.0) / wavelength_m
        # circumradius = 0.4 * (D_GOLAY/2)
        triangle_radius_angular = (0.4 * D_GOLAY / 2.0) / wavelength_m
        ctf = np.zeros((ny, nx), dtype=np.float64)
        angles = np.deg2rad([0, 120, 240])
        for ang in angles:
            cx = triangle_radius_angular * np.cos(ang)
            cy = triangle_radius_angular * np.sin(ang)
            dist = np.sqrt((omega_x - cx) ** 2 + (omega_y - cy) ** 2)
            ctf += (dist <= radius_sub).astype(np.float64)
        # binary (no overlap)
        ctf = (ctf > 0).astype(np.float64)

    elif aperture_type == 'golay9':
        radius_sub = (SUBSIZE / 2.0) / wavelength_m
        max_rel = np.max(np.sqrt(_GOLAY9_REL[:, 0] ** 2 + _GOLAY9_REL[:, 1] ** 2))
        scale_angular = (0.45 * D_GOLAY / 2.0) / (max_rel * wavelength_m)
        ctf = np.zeros((ny, nx), dtype=np.float64)
        for pos in _GOLAY9_REL:
            cx = pos[0] * scale_angular
            cy = pos[1] * scale_angular
            dist = np.sqrt((omega_x - cx) ** 2 + (omega_y - cy) ** 2)
            ctf += (dist <= radius_sub).astype(np.float64)
        ctf = (ctf > 0).astype(np.float64)

    else:
        raise ValueError(f"Unknown aperture type '{aperture_type}'. "
                         "Choose from 'full', 'golay3', 'golay9'.")

    # ------------------------------------------------------------------
    # OTF = autocorrelation of the CTF
    # ------------------------------------------------------------------
    # autocorrelation via FFT: OTF = ifft(|fft(CTF)|^2)
    ctf_ft = np.fft.fft2(ctf)
    otf = np.fft.ifft2(np.abs(ctf_ft) ** 2).real
    otf /= otf.max()

    # 生成掩膜（OTF 幅度大于阈值的位置）
    mask = np.abs(otf) > 0.01
    return otf, mask


# ----------------------------------------------------------------------
# Visibility‑based observation simulation
# ----------------------------------------------------------------------

def generate_observed(sst, ssh, lx_km, ly_km, noise_level=1e-5):
    """
    Simulate observation through the three apertures by multiplying the
    scene's Fourier transform with the aperture's OTF.

    Parameters
    ----------
    sst : 2D ndarray
        True SST field (already normalised to [0,1]).
    ssh : 2D ndarray
        True SSH field (metres).
    lx_km, ly_km : float
        Physical extent of the scene in km.
    noise_level : float
        Standard deviation of additive Gaussian noise (in the same physical
        units as the field).

    Returns
    -------
    results : dict
        keys 'Full', 'Golay3', 'Golay9', each containing:
          'sst_obs' : 2D ndarray
          'ssh_obs' : 2D ndarray
          'otf_mask': 2D bool ndarray (mask of support)
    """
    ny, nx = sst.shape
    dx_m = (lx_km * 1000.0) / nx
    dy_m = (ly_km * 1000.0) / ny
    H_m = GEO_HEIGHT
    wavelength_m = WAVELENGTH

    aperture_types = ['Full', 'Golay3', 'Golay9']
    results = {}
    rng = np.random.default_rng()   # noisy seed – reproducible via global seed if needed

    for name in aperture_types:
        # nominal type for the helper (lowercase)
        type_code = name.lower()
        otf, mask = compute_otf_for_aperture(type_code, (ny, nx),
                                             dx_m, dy_m, H_m, wavelength_m)

        # SST
        F_sst = np.fft.fft2(sst)
        F_sst_obs = F_sst * otf
        sst_obs = np.real(np.fft.ifft2(F_sst_obs))

        # SSH
        F_ssh = np.fft.fft2(ssh)
        F_ssh_obs = F_ssh * otf
        ssh_obs = np.real(np.fft.ifft2(F_ssh_obs))

        # additive noise
        sst_obs += noise_level * rng.normal(size=sst.shape)
        ssh_obs += noise_level * rng.normal(size=ssh.shape)

        # clip to the dynamic range of the original field (preserve physical meaning)
        sst_obs = np.clip(sst_obs, sst.min(), sst.max())
        ssh_obs = np.clip(ssh_obs, ssh.min(), ssh.max())

        results[name] = {'sst_obs': sst_obs, 'ssh_obs': ssh_obs,
                         'otf_mask': mask}

    return results


# =============================================================================
# Plotting functions (comprehensive comparison figures)
# =============================================================================

def plot_all_states_observations(true_data, obs_data,
                                 lx_km=1.0, ly_km=1.0,
                                 save_sst="all_states_sst_observations.png",
                                 save_ssh="all_states_ssh_observations.png",
                                 show=True):
    """
    true_data: dict {state: (sst, ssh)}
    obs_data:  dict {state: {'Full':{'sst_obs':...,'ssh_obs':...}, ...}}
    Generates two 4×3 figures.
    """
    states = ['calm', 'langmuir', 'turbulent']
    apertures = ['Full', 'Golay3', 'Golay9']
    extent = [0, lx_km, 0, ly_km]

    # ----- SST figure -----
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for col, state in enumerate(states):
        sst, _ = true_data[state]
        # row 0: true
        ax = axes[0, col]
        im = ax.imshow(sst, origin='lower', cmap='RdYlBu_r',
                       extent=extent, vmin=0, vmax=1)
        ax.set_title(f'{state.capitalize()} True')
        ax.set_xlabel('x (km)')
        if col == 0:
            ax.set_ylabel('y (km)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for row, name in enumerate(apertures, start=1):
            ax = axes[row, col]
            sst_obs = obs_data[state][name]['sst_obs']
            im = ax.imshow(sst_obs, origin='lower', cmap='RdYlBu_r',
                           extent=extent, vmin=0, vmax=1)
            ax.set_title(f'{name} observed')
            ax.set_xlabel('x (km)')
            if col == 0:
                ax.set_ylabel('y (km)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('SST Observations: Three States × Four Apertures',
                 fontweight='bold', fontsize=14, y=0.935)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_sst, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    # ----- SSH figure -----
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    # get global vmin/vmax across all states & observations
    all_ssh = [true_data[s][1].ravel() for s in states]
    for s in states:
        for name in apertures:
            all_ssh.append(obs_data[s][name]['ssh_obs'].ravel())
    all_ssh = np.concatenate(all_ssh)
    vmin_ssh, vmax_ssh = all_ssh.min(), all_ssh.max()

    for col, state in enumerate(states):
        _, ssh = true_data[state]
        ax = axes[0, col]
        im = ax.imshow(ssh, origin='lower', cmap='coolwarm',
                       extent=extent, vmin=vmin_ssh, vmax=vmax_ssh)
        ax.set_title(f'{state.capitalize()} True')
        ax.set_xlabel('x (km)')
        if col == 0:
            ax.set_ylabel('y (km)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for row, name in enumerate(apertures, start=1):
            ax = axes[row, col]
            ssh_obs = obs_data[state][name]['ssh_obs']
            im = ax.imshow(ssh_obs, origin='lower', cmap='coolwarm',
                           extent=extent, vmin=vmin_ssh, vmax=vmax_ssh)
            ax.set_title(f'{name} observed')
            ax.set_xlabel('x (km)')
            if col == 0:
                ax.set_ylabel('y (km)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('SSH Observations: Three States × Four Apertures',
                 fontweight='bold', fontsize=14, y=0.935)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_ssh, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_all_states_spectral_comparison(true_data, obs_data,
                                        lx_km=1.0, ly_km=1.0,
                                        savepath="all_states_spectral_comparison.png",
                                        show=True):
    """
    3×2 panel:
      row0: radial power (SST left, SSH right) – observations shifted vertically
      row1: anisotropy A(k) (SST left, SSH right)
      row2: orientation θ(k) (SST left, SSH right)
    Colours = state, linestyles = aperture (True: solid, Full: dashed,
            Golay3: dashdot, Golay9: dotted)
    """
    states = ['calm', 'langmuir', 'turbulent']
    apertures = ['Full', 'Golay3', 'Golay9']
    # aperture -> linestyle
    ls_map = {'True':  '-',
              'Full':  '--',
              'Golay3': '-.',
              'Golay9': ':'}
    state_color = {'calm': 'blue', 'langmuir': 'green', 'turbulent': 'red'}

    # vertical shift factors to separate observed spectra from true ones
    shift_map = {'Full': 10.0, 'Golay3': 100.0, 'Golay9': 1000.0}
    angle_map = {'Full': 360., 'Golay3': 720., 'Golay9': 1080.}

    # get grid spacing from first true SST (all fields share the same shape)
    sample_sst = true_data[states[0]][0]
    ny, nx = sample_sst.shape
    dx = lx_km / nx
    dy = ly_km / ny

    # Compute common maximum wavenumber across all aperture masks
    k_maxes = []
    # use temporary frequency arrays
    kx_temp = np.fft.fftfreq(nx, d=dx)
    ky_temp = np.fft.fftfreq(ny, d=dy)
    KX_temp, KY_temp = np.meshgrid(kx_temp, ky_temp)
    k_rad_temp = np.sqrt(KX_temp**2 + KY_temp**2)
    for state in states:
        for name in apertures:
            m = obs_data[state][name]['otf_mask']
            # only consider positive wavenumbers inside mask
            valid = m & (k_rad_temp > 0)
            if np.any(valid):
                km = np.max(k_rad_temp[valid])
                k_maxes.append(km)
    k_max_common = max(k_maxes) if k_maxes else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    def plot_panel(ax, variable, field_type, log_x=False):
        """
        variable: 'radial', 'anisotropy', 'orientation'
        field_type: 'sst' or 'ssh'
        """
        for state in states:
            # ---- true field ----
            field = true_data[state][0] if field_type == 'sst' else true_data[state][1]
            color = state_color[state]

            if variable == 'radial':
                k, psd = radial_power_spectrum(field, dx, dy)
                y = psd / psd[0]
            elif variable == 'anisotropy':
                k, A, _ = moment_anisotropy(field, dx, dy)
                y = A
            else:  # orientation
                k, _, theta = moment_anisotropy(field, dx, dy)
                y = np.degrees(theta)

            # plot true curve (no shift)
            if log_x:
                ax.semilogx(k, y, color=color, linestyle=ls_map['True'],
                            linewidth=1.5, label=f'{state} True')
            else:
                ax.loglog(k, y, color=color, linestyle=ls_map['True'],
                          linewidth=1.5, label=f'{state} True')

            # ---- observed fields (with shift for radial power) ----
            for name in apertures:
                obs_field = obs_data[state][name][f'{field_type}_obs']
                mask = obs_data[state][name]['otf_mask']

                if variable == 'radial':
                    k_obs, psd_obs = radial_power_spectrum(obs_field, dx, dy, mask=mask)
                    y_obs = psd_obs / psd_obs[0] * shift_map[name]
                elif variable == 'anisotropy':
                    k_obs, A_obs, _ = moment_anisotropy(obs_field, dx, dy, mask=mask)
                    y_obs = A_obs * shift_map[name]
                else:  # orientation
                    k_obs, _, theta_obs = moment_anisotropy(obs_field, dx, dy, mask=mask)
                    y_obs = np.degrees(theta_obs) + angle_map[name]

                if log_x:
                    ax.semilogx(k_obs, y_obs, color=color, linestyle=ls_map[name],
                                linewidth=1.0)
                else:
                    ax.loglog(k_obs, y_obs, color=color, linestyle=ls_map[name],
                              linewidth=1.0)

        # axis labels and scales
        if variable == 'radial':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Normalized Power (offset applied)')
        elif variable == 'anisotropy':
            ax.set_xscale('log')
            ax.set_ylabel('A = (λ₁-λ₂)/(λ₁+λ₂)')
        else:  # orientation
            ax.set_xscale('log')
            ax.set_ylabel('Orientation (deg)')
        ax.set_xlabel('Wavenumber (cyc/km)')
        ax.set_xlim(right=k_max_common)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- row 0: radial power ----
    plot_panel(axes[0, 0], 'radial', 'sst')
    axes[0, 0].set_title('Radial Power Spectrum – SST')
    plot_panel(axes[0, 1], 'radial', 'ssh')
    axes[0, 1].set_title('Radial Power Spectrum – SSH')

    # ---- row 1: anisotropy ----
    plot_panel(axes[1, 0], 'anisotropy', 'sst')
    axes[1, 0].set_title('Anisotropy Degree – SST')
    plot_panel(axes[1, 1], 'anisotropy', 'ssh')
    axes[1, 1].set_title('Anisotropy Degree – SSH')

    # ---- row 2: orientation ----
    plot_panel(axes[2, 0], 'orientation', 'sst', True)
    axes[2, 0].set_title('Principal Orientation – SST')
    plot_panel(axes[2, 1], 'orientation', 'ssh', True)
    axes[2, 1].set_title('Principal Orientation – SSH')

    # ---- unified legend ----
    handles = []
    for state in states:
        for aper in ['True'] + apertures:
            label = f'{state} {aper}'
            handles.append(plt.Line2D([0], [0],
                                      color=state_color[state],
                                      linestyle=ls_map[aper],
                                      lw=1.5, label=label))
    fig.legend(handles=handles, loc='upper center',
               ncol=6, fontsize='small', bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Spectral Feature Comparison (True vs Observed)',
                 fontweight='bold', fontsize=14, y=1.06)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive observation comparison figures "
                    "(all sea states × all apertures)"
    )
    parser.add_argument('--lx_km', type=float, default=1.0,
                        help='Scene width in km (default: 1.0)')
    parser.add_argument('--ly_km', type=float, default=1.0,
                        help='Scene height in km (default: 1.0)')
    parser.add_argument('--noise', type=float, default=0.0001,
                        help='Observation noise level (default: 0.0001)')
    args = parser.parse_args()

    lx_km = args.lx_km
    ly_km = args.ly_km

    states = ['calm', 'langmuir', 'turbulent']
    true_data = {}
    for state in states:
        try:
            sst = np.load(f"{state}_sst.npy")
            ssh = np.load(f"{state}_ssh.npy")
            true_data[state] = (sst, ssh)
            print(f"Loaded {state} fields.")
        except FileNotFoundError:
            print(f"Error: missing {state}_sst.npy / {state}_ssh.npy.")
            print("Run seasurface.py first to generate these files.")
            sys.exit(1)

    # Simulate observations for every state and aperture (visibility‑based)
    obs_data = {}
    for state, (sst, ssh) in true_data.items():
        print(f"Simulating observations for {state}...")
        obs = generate_observed(sst, ssh, lx_km, ly_km,
                                noise_level=args.noise)
        obs_data[state] = obs

    print("Generating comprehensive figures...")
    plot_all_states_observations(true_data, obs_data,
                                 lx_km=lx_km, ly_km=ly_km,
                                 show=True)
    print("   saved: all_states_sst_observations.png")
    print("   saved: all_states_ssh_observations.png")

    plot_all_states_spectral_comparison(true_data, obs_data,
                                        lx_km=lx_km, ly_km=ly_km,
                                        show=True)
    print("   saved: all_states_spectral_comparison.png")
    print("All done.")
