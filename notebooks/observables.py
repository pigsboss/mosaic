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
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from apertures import (full_aperture, golay3, golay9,
                       compute_psf, compute_mtf,
                       D_FULL, D_GOLAY, GEO_HEIGHT)
from seasurface import moment_anisotropy

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


# =============================================================================
# New plotting functions (comprehensive comparison figures)
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
                 fontweight='bold', fontsize=14)
    fig.tight_layout()
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
                 fontweight='bold', fontsize=14)
    fig.tight_layout()
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
      row0: radial power (SST left, SSH right)
      row1: anisotropy A(k) (SST left, SSH right)
      row2: orientation θ(k) (SST left, SSH right)
    Colours = state, linestyles = aperture (True: solid, Full: dashed,
            Golay3: dashdot, Golay9: dotted)
    """
    states = ['calm', 'langmuir', 'turbulent']
    apertures = ['Full', 'Golay3', 'Golay9']
    # aper -> linestyle
    ls_map = {'True':  '-',
              'Full':  '--',
              'Golay3': '-.',
              'Golay9': ':'}
    state_color = {'calm': 'blue', 'langmuir': 'green', 'turbulent': 'red'}

    # one true sst (any state) to get grid spacing (all same size)
    sample_sst = true_data[states[0]][0]
    ny, nx = sample_sst.shape
    dx = lx_km / nx
    dy = ly_km / ny

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # Helper to plot one panel
    def plot_panel(ax, variable, field_type, log_x=False):
        """
        variable: 'radial', 'anisotropy', 'orientation'
        field_type: 'sst' or 'ssh'
        """
        for state in states:
            # True field
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

            if log_x:
                ax.semilogx(k, y, color=color, linestyle=ls_map['True'],
                            linewidth=1.5, label=f'{state} True')
            else:
                ax.loglog(k, y, color=color, linestyle=ls_map['True'],
                          linewidth=1.5, label=f'{state} True')

            for name in apertures:
                obs_field = obs_data[state][name][f'{field_type}_obs']
                if variable == 'radial':
                    k, psd = radial_power_spectrum(obs_field, dx, dy)
                    y = psd / psd[0]
                elif variable == 'anisotropy':
                    k, A, _ = moment_anisotropy(obs_field, dx, dy)
                    y = A
                else:
                    k, _, theta = moment_anisotropy(obs_field, dx, dy)
                    y = np.degrees(theta)

                if log_x:
                    ax.semilogx(k, y, color=color, linestyle=ls_map[name],
                                linewidth=1.0)
                else:
                    ax.loglog(k, y, color=color, linestyle=ls_map[name],
                              linewidth=1.0)

        # set log scale for x always (radial wavenumber)
        if variable == 'radial':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Normalized Power')
        elif variable == 'anisotropy':
            ax.set_xscale('log')
            ax.set_ylabel('A = (λ₁-λ₂)/(λ₁+λ₂)')
        else:
            ax.set_xscale('log')
            ax.set_ylabel('Orientation (deg)')
        ax.set_xlabel('Wavenumber (cyc/km)')
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
    plot_panel(axes[2, 0], 'orientation', 'sst')
    axes[2, 0].set_title('Principal Orientation – SST')
    plot_panel(axes[2, 1], 'orientation', 'ssh')
    axes[2, 1].set_title('Principal Orientation – SSH')

    # Add legend manually (state colour + aperture linestyle)
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
    parser.add_argument('--noise', type=float, default=0.02,
                        help='Observation noise level (default: 0.02)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='MTF threshold for observation (default: 0.1)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize MTF in observation (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
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

    # Simulate observations for every state and aperture
    obs_data = {}
    for state, (sst, ssh) in true_data.items():
        print(f"Simulating observations for {state}...")
        obs = generate_observed(sst, ssh,
                                noise_level=args.noise,
                                lx_km=lx_km, ly_km=ly_km,
                                threshold=args.threshold,
                                normalize=args.normalize)
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
