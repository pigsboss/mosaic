#!/usr/bin/env python3
"""
批量模拟观测：对 seasurface_dynamic.py 生成的时序场（.npy）逐帧施加
合成孔径系统，保存观测数据，并计算时间平均的功率谱、谱矩张量、
各向异性度及主导方向。

用法：
  python observables_batch.py --load timeseries_calm_sst.npy --aperture full
                              --lx 1.0 --ly 1.0 --output calm_sst_full
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# 导入 observables 模块中的 OTF 计算函数和相关常量
from observables import compute_otf_for_aperture, D_FULL, D_GOLAY, SUBSIZE, WAVELENGTH, GEO_HEIGHT, _GOLAY9_REL


def load_time_series(path):
    """加载时序场，返回 (n_frames, ny, nx) 数组。"""
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (time, y, x), got shape {data.shape}")
    return data


def apply_otf_frame(field, otf):
    """
    对单帧场施加 OTF 滤波（频域相乘）。
    field : 2D ndarray
    otf   : 2D ndarray (same shape)
    返回 : 滤波后的实空间场。
    """
    F = np.fft.fft2(field)
    F_obs = F * otf
    obs = np.real(np.fft.ifft2(F_obs))
    return obs


def generate_observed_sequence(frames, aperture_type, lx_km, ly_km):
    """
    逐帧施加孔径 OTF，返回观测帧序列 (n_frames, ny, nx) 及对应的 mask。

    Parameters
    ----------
    frames : ndarray (n_frames, ny, nx)
        原始时序场。
    aperture_type : str
        'full', 'golay3', 'golay9'
    lx_km, ly_km : float
        场景物理尺寸 (km)。

    Returns
    -------
    obs_frames : ndarray (n_frames, ny, nx)
    mask : 2D bool ndarray (ny, nx)
        OTF 有效区域（用于后续频谱分析）。
    """
    n_frames, ny, nx = frames.shape
    # 空间采样间隔（米）
    dx_m = (lx_km * 1000.0) / nx
    dy_m = (ly_km * 1000.0) / ny

    # 计算 OTF
    otf, mask = compute_otf_for_aperture(
        aperture_type, (ny, nx), dx_m, dy_m, GEO_HEIGHT, WAVELENGTH
    )

    obs_frames = np.empty_like(frames)
    for i in range(n_frames):
        obs_frames[i] = apply_otf_frame(frames[i], otf)

    return obs_frames, mask


def time_averaged_psd_2d(frames, mask):
    """
    计算时间平均的 2D 功率谱（仅统计 mask 内像素）。
    frames : (n_frames, ny, nx)
    mask   : (ny, nx) bool
    返回 : (ny, nx) 平均功率谱。
    """
    n_frames, ny, nx = frames.shape
    psd_sum = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_frames):
        F = np.fft.fft2(frames[i])
        psd_sum += np.abs(F) ** 2
    psd_avg = psd_sum / n_frames
    # 将 mask 外的区域置为 0（binning 时会忽略）
    psd_avg[~mask] = 0.0
    return psd_avg


def radial_power_spectrum_from_psd(psd2d, dx, dy, mask):
    """
    从时间平均的 2D PSD 计算径向功率谱（仅 mask 内有效）。
    psd2d : (ny, nx)
    dx, dy : 网格间距（与 PSF 里一致）
    mask   : (ny, nx) bool
    返回 : k_center (1D), radial_psd (1D, masked, 无数据仓位被掩盖)
    """
    ny, nx = psd2d.shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)

    k_max = np.max(k_rad)
    n_bins = 100
    bins = np.linspace(0, k_max, n_bins + 1)
    radial_psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    # 只遍历 mask 内像素
    idx, idy = np.where(mask)
    for i, j in zip(idx, idy):
        kr = k_rad[i, j]
        bin_idx = np.digitize(kr, bins) - 1
        if 0 <= bin_idx < n_bins:
            radial_psd[bin_idx] += psd2d[i, j]
            counts[bin_idx] += 1

    nonzero = counts > 0
    radial_psd[nonzero] /= counts[nonzero]
    k_center = 0.5 * (bins[1:] + bins[:-1])
    # 掩盖无数据的 bin
    radial_psd_masked = np.ma.array(radial_psd, mask=(counts == 0))
    return k_center, radial_psd_masked


def moment_tensor_from_psd(psd2d, dx, dy, mask, n_bins=100):
    """
    从时间平均的 2D PSD 计算谱矩张量分量 m20, m11, m02（仅 mask 内有效）。
    psd2d : (ny, nx)
    dx, dy : 网格间距
    mask : (ny, nx) bool
    n_bins : int
    返回 : k_centers, m20, m11, m02 (每个变量都是 1D，无数据的仓位值为 0)
    """
    ny, nx = psd2d.shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)

    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    m20 = np.zeros(n_bins)
    m11 = np.zeros(n_bins)
    m02 = np.zeros(n_bins)

    for i in range(n_bins):
        mask_bin = (k_rad >= bins[i]) & (k_rad < bins[i+1])
        mask_bin = mask_bin & mask  # 只考虑 mask 内
        if not np.any(mask_bin):
            continue
        p_vals = psd2d[mask_bin]
        kx_vals = KX[mask_bin]
        ky_vals = KY[mask_bin]
        total = np.sum(p_vals)
        if total == 0:
            continue
        m20[i] = np.sum(p_vals * kx_vals**2) / total
        m11[i] = np.sum(p_vals * kx_vals * ky_vals) / total
        m02[i] = np.sum(p_vals * ky_vals**2) / total

    return k_centers, m20, m11, m02


def compute_anisotropy_and_orientation(k_mom, m20, m11, m02):
    """
    由谱矩张量计算各向异性度 A 和主导方向 theta (度)。
    """
    n = len(k_mom)
    A = np.zeros(n)
    theta_deg = np.zeros(n)
    for i in range(n):
        trace = m20[i] + m02[i]
        det = m20[i] * m02[i] - m11[i] * m11[i]
        disc = np.sqrt(trace**2 - 4*det)
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)
        if lambda1 + lambda2 > 0:
            A[i] = (lambda1 - lambda2) / (lambda1 + lambda2)
        else:
            A[i] = 0.0
        if (m20[i] - m02[i]) == 0 and m11[i] == 0:
            theta_rad = 0.0
        else:
            theta_rad = 0.5 * np.arctan2(2*m11[i], m20[i] - m02[i])
        theta_deg[i] = np.degrees(theta_rad)
    return A, theta_deg


def plot_spectra(k_iso, power_iso, k_mom, m20, m11, m02, A, theta_deg, save_fig=None):
    """
    4‑面板图：径向功率谱、矩张量、各向异性度、主导方向。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ((ax_iso, ax_mom), (ax_aniso, ax_orient)) = axes

    # 径向功率谱
    ax_iso.loglog(k_iso, power_iso / power_iso[0], 'k-')
    ax_iso.set_title('Time‑averaged radial power (masked)')
    ax_iso.set_xlabel('Wavenumber (cyc/km)')
    ax_iso.set_ylabel('Normalized Power')
    ax_iso.grid(True, which='both', linestyle='--', alpha=0.5)

    # 谱矩张量
    ax_mom.semilogx(k_mom, m20, label='m20')
    ax_mom.semilogx(k_mom, m11, label='m11')
    ax_mom.semilogx(k_mom, m02, label='m02')
    ax_mom.set_title('Spectral moment tensor components')
    ax_mom.set_xlabel('Wavenumber (cyc/km)')
    ax_mom.set_ylabel('Moment')
    ax_mom.legend()
    ax_mom.grid(True, which='both', linestyle='--', alpha=0.5)

    # 各向异性度
    ax_aniso.semilogx(k_mom, A, 'k-')
    ax_aniso.set_title('Anisotropy A = (λ₁−λ₂)/(λ₁+λ₂)')
    ax_aniso.set_xlabel('Wavenumber (cyc/km)')
    ax_aniso.set_ylabel('A')
    ax_aniso.set_ylim(-0.05, 1.05)
    ax_aniso.grid(True, which='both', linestyle='--', alpha=0.5)

    # 主导方向
    ax_orient.semilogx(k_mom, theta_deg, 'k-')
    ax_orient.set_title('Principal orientation θ(k)')
    ax_orient.set_xlabel('Wavenumber (cyc/km)')
    ax_orient.set_ylabel('θ (degrees)')
    ax_orient.grid(True, which='both', linestyle='--', alpha=0.5)

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=150)
        print(f"Saved figure: {save_fig}")
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Batch observation and spectral analysis of dynamic sea surface fields."
    )
    parser.add_argument('--load', required=True,
                        help='Path to input .npy time series (3D array, frames×ny×nx).')
    parser.add_argument('--aperture', choices=['full', 'golay3', 'golay9'],
                        default='full', help='Aperture type (default: full).')
    parser.add_argument('--lx', type=float, default=1.0, help='Scene width in km (default: 1.0).')
    parser.add_argument('--ly', type=float, default=1.0, help='Scene height in km (default: 1.0).')
    parser.add_argument('--output', required=True, help='Base name for output files.')
    args = parser.parse_args()

    # 1. 加载原始时序
    print(f"Loading time series from {args.load}")
    frames = load_time_series(args.load)
    n_frames, ny, nx = frames.shape
    print(f"  Shape: {n_frames} frames, {ny}×{nx} pixels")

    # 2. 模拟观测
    print(f"Generating observed sequence for aperture '{args.aperture}'…")
    obs_frames, mask = generate_observed_sequence(frames, args.aperture, args.lx, args.ly)
    obs_file = f"{args.output}_observed.npy"
    np.save(obs_file, obs_frames)
    print(f"  Saved observed frames to {obs_file}")

    # 3. 时间平均 2D PSD
    print("Computing time‑averaged 2D PSD…")
    psd2d_avg = time_averaged_psd_2d(obs_frames, mask)

    # 4. 径向功率谱
    dx = args.lx / nx
    dy = args.ly / ny
    k_iso, power_iso = radial_power_spectrum_from_psd(psd2d_avg, dx, dy, mask)
    print("  Radial power spectrum done.")

    # 5. 谱矩张量
    k_mom, m20, m11, m02 = moment_tensor_from_psd(psd2d_avg, dx, dy, mask, n_bins=100)
    print("  Moment tensor computed.")

    # 6. 各向异性 & 方向
    A, theta_deg = compute_anisotropy_and_orientation(k_mom, m20, m11, m02)

    # 7. 保存数据
    data_file = f"{args.output}_spectra.npz"
    np.savez(data_file,
             k_iso=k_iso, power_iso=power_iso,
             k_mom=k_mom, m20=m20, m11=m11, m02=m02,
             A=A, theta_deg=theta_deg)
    print(f"  Saved spectral data to {data_file}")

    # 8. 绘图
    fig_file = f"{args.output}_spectra.png"
    plot_spectra(k_iso, power_iso, k_mom, m20, m11, m02, A, theta_deg, save_fig=fig_file)
    print("Processing complete.")


if __name__ == "__main__":
    main()
