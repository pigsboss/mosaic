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


# =====================================================================
# NEW: per‑frame radial power spectrum sequence (mean + std)
# =====================================================================
def radial_power_spectra_sequence(frames, dx, dy, mask, n_bins=100):
    """
    逐帧计算径向功率谱，返回每个 radial bin 的平均功率和标准差。
    frames : ndarray (n_frames, ny, nx)
    dx, dy : 网格间距（与之前的定义一致）
    mask   : 2D bool 掩膜（只在该区域内统计）
    返回:
      k_centers : 1D array
      mean_power: masked array (n_bins,)
      std_power : masked array (n_bins,)
    """
    n_frames, ny, nx = frames.shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    # 只考虑 mask 内且 k>0 的像素
    valid_mask = mask & (k_rad > 0)
    idx, idy = np.where(valid_mask)
    kr_vals = k_rad[idx, idy]
    bin_indices = np.digitize(kr_vals, bins) - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    bin_indices = bin_indices[valid]
    idx = idx[valid]
    idy = idy[valid]

    # 存储每帧每个 bin 的平均功率
    per_frame_power = np.zeros((n_frames, n_bins))
    for t in range(n_frames):
        F = np.fft.fft2(frames[t])
        psd2d = np.abs(F) ** 2
        vals = psd2d[idx, idy]
        # 对每个 bin 求平均（若没有像素则为 nan）
        for b in range(n_bins):
            mask_b = (bin_indices == b)
            if np.any(mask_b):
                per_frame_power[t, b] = np.mean(vals[mask_b])
            else:
                per_frame_power[t, b] = np.nan

    # 计算均值和标准差
    mean_power = np.zeros(n_bins)
    std_power = np.zeros(n_bins)
    has_data = np.zeros(n_bins, dtype=bool)
    for b in range(n_bins):
        vals = per_frame_power[:, b]
        if not np.all(np.isnan(vals)):
            mean_power[b] = np.nanmean(vals)
            std_power[b] = np.nanstd(vals)
            has_data[b] = True
        else:
            mean_power[b] = 0.0
            std_power[b] = 0.0

    mean_masked = np.ma.array(mean_power, mask=~has_data)
    std_masked = np.ma.array(std_power, mask=~has_data)
    return k_centers, mean_masked, std_masked


# =====================================================================
# NEW: per‑frame moment tensor sequence (mean + std)
# =====================================================================
def moment_tensor_sequence(frames, dx, dy, mask, n_bins=100):
    """
    逐帧计算谱矩张量分量 (m20, m11, m02)，返回每个 radial bin 的均值和标准差。
    参数含义同上。
    返回:
      k_centers,
      mean_m20, std_m20,
      mean_m11, std_m11,
      mean_m02, std_m02
      (均为 masked array)
    """
    n_frames, ny, nx = frames.shape
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)
    k_max = np.max(k_rad)
    bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[1:] + bins[:-1])

    valid_mask = mask & (k_rad > 0)
    idx, idy = np.where(valid_mask)
    kr_vals = k_rad[idx, idy]
    bin_indices = np.digitize(kr_vals, bins) - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    bin_indices = bin_indices[valid]
    idx = idx[valid]
    idy = idy[valid]
    kx_vals = KX[idx, idy]
    ky_vals = KY[idx, idy]

    per_frame_m20 = np.zeros((n_frames, n_bins))
    per_frame_m11 = np.zeros((n_frames, n_bins))
    per_frame_m02 = np.zeros((n_frames, n_bins))

    for t in range(n_frames):
        F = np.fft.fft2(frames[t])
        psd2d = np.abs(F) ** 2
        p_vals = psd2d[idx, idy]
        for b in range(n_bins):
            mask_b = (bin_indices == b)
            if not np.any(mask_b):
                per_frame_m20[t, b] = np.nan
                per_frame_m11[t, b] = np.nan
                per_frame_m02[t, b] = np.nan
                continue
            p = p_vals[mask_b]
            total = np.sum(p)
            if total == 0:
                per_frame_m20[t, b] = np.nan
                per_frame_m11[t, b] = np.nan
                per_frame_m02[t, b] = np.nan
            else:
                per_frame_m20[t, b] = np.sum(p * kx_vals[mask_b]**2) / total
                per_frame_m11[t, b] = np.sum(p * kx_vals[mask_b] * ky_vals[mask_b]) / total
                per_frame_m02[t, b] = np.sum(p * ky_vals[mask_b]**2) / total

    # 辅助函数：计算每列的均值和标准差，并转为 masked array
    def _mean_std(arr_2d):
        mean = np.zeros(n_bins)
        std = np.zeros(n_bins)
        has_data = np.zeros(n_bins, dtype=bool)
        for b in range(n_bins):
            col = arr_2d[:, b]
            if not np.all(np.isnan(col)):
                mean[b] = np.nanmean(col)
                std[b] = np.nanstd(col)
                has_data[b] = True
        return (np.ma.array(mean, mask=~has_data),
                np.ma.array(std, mask=~has_data))

    mean_m20, std_m20 = _mean_std(per_frame_m20)
    mean_m11, std_m11 = _mean_std(per_frame_m11)
    mean_m02, std_m02 = _mean_std(per_frame_m02)

    return (k_centers,
            mean_m20, std_m20,
            mean_m11, std_m11,
            mean_m02, std_m02)


# ----------------------------------------------------------------------
# compute_anisotropy_and_orientation (unchanged)
# ----------------------------------------------------------------------
def compute_anisotropy_and_orientation(k_mom, m20, m11, m02):
    """
    由谱矩张量计算各向异性度 A 和主导方向 theta (度)。
    输入 m20, m11, m02 应为 masked arrays。
    返回的 A 和 theta_deg 也是 masked arrays，无数据的仓位被掩蔽。
    """
    n = len(k_mom)
    A = np.ma.zeros(n)          # 创建 masked array，初始 mask=False
    theta_deg = np.ma.zeros(n)
    # 标记所有仓位为无效，后续赋值时取消掩蔽
    A.mask = True
    theta_deg.mask = True

    for i in range(n):
        if m20.mask[i] or m11.mask[i] or m02.mask[i]:
            continue   # 保持掩蔽
        trace = m20[i] + m02[i]
        det = m20[i] * m02[i] - m11[i] * m11[i]
        disc = np.sqrt(trace**2 - 4*det)
        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)
        if lambda1 + lambda2 > 0:
            A[i] = (lambda1 - lambda2) / (lambda1 + lambda2)
        else:
            A[i] = 0.0
        # 计算方向
        if (m20[i] - m02[i]) == 0 and m11[i] == 0:
            theta_rad = 0.0
        else:
            theta_rad = 0.5 * np.arctan2(2*m11[i], m20[i] - m02[i])
        theta_deg[i] = np.degrees(theta_rad)
        # 成功赋值后取消该位置的掩蔽
        A.mask[i] = False
        theta_deg.mask[i] = False

    return A, theta_deg


# =====================================================================
# UPDATED plot_spectra with error bars
# =====================================================================
def plot_spectra(k_iso, power_iso, std_power_iso,
                 k_mom, m20, std_m20, m11, std_m11, m02, std_m02,
                 A, theta_deg, save_fig=None):
    """
    4‑面板图：径向功率谱、矩张量、各向异性度、主导方向。
    在径向功率谱和矩张量分量上添加误差棒 (1σ)。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ((ax_iso, ax_mom), (ax_aniso, ax_orient)) = axes

    # ---- 径向功率谱归一化 ----
    ref = power_iso[0]
    if np.ma.is_masked(ref):
        valid = ~power_iso.mask
        ix = np.argmax(valid)
        ref = power_iso[ix]
    norm_power = power_iso / ref
    # 误差棒也按相同因子缩放（掩蔽的仓位误差设为0）
    yerr = np.where(std_power_iso.mask, 0.0, std_power_iso / ref)

    ax_iso.errorbar(k_iso, norm_power, yerr=yerr, fmt='k-',
                    ecolor='gray', capsize=2)
    ax_iso.set_title('Time‑averaged radial power (+ std)')
    ax_iso.set_xlabel('Wavenumber (cyc/km)')
    ax_iso.set_ylabel('Normalized Power')
    ax_iso.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_iso.set_xscale('log')
    ax_iso.set_yscale('log')

    # ---- 谱矩张量 ----
    for (m, std, label) in [(m20, std_m20, 'm20'),
                             (m11, std_m11, 'm11'),
                             (m02, std_m02, 'm02')]:
        # 转换为普通数组并处理掩蔽
        x = k_mom
        y = m
        err = np.where(std.mask, 0.0, std)
        ax_mom.errorbar(x, y, yerr=err, label=label, capsize=2)
    ax_mom.set_title('Spectral moment tensor components')
    ax_mom.set_xlabel('Wavenumber (cyc/km)')
    ax_mom.set_ylabel('Moment')
    ax_mom.legend()
    ax_mom.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_mom.set_xscale('log')

    # ---- 各向异性度（无误差棒）----
    ax_aniso.semilogx(k_mom, A, 'k-')
    ax_aniso.set_title('Anisotropy A = (λ₁−λ₂)/(λ₁+λ₂)')
    ax_aniso.set_xlabel('Wavenumber (cyc/km)')
    ax_aniso.set_ylabel('A')
    ax_aniso.set_ylim(-0.05, 1.05)
    ax_aniso.grid(True, which='both', linestyle='--', alpha=0.5)

    # ---- 主导方向（无误差棒）----
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


# =====================================================================
# Main (modified)
# =====================================================================
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

    # 3. 计算径向功率谱的均值与标准差
    dx = args.lx / nx
    dy = args.ly / ny
    print("Computing radial power spectra (frame‑wise)…")
    k_iso, mean_power, std_power = radial_power_spectra_sequence(
        obs_frames, dx, dy, mask)
    print("  Done.")

    # 4. 计算谱矩张量的均值与标准差
    print("Computing spectral moment tensors (frame‑wise)…")
    (k_mom,
     mean_m20, std_m20,
     mean_m11, std_m11,
     mean_m02, std_m02) = moment_tensor_sequence(
        obs_frames, dx, dy, mask)
    print("  Done.")

    # 5. 各向异性与方向 (基于均值的矩张量)
    A, theta_deg = compute_anisotropy_and_orientation(
        k_mom, mean_m20, mean_m11, mean_m02)

    # 6. 保存数据
    data_file = f"{args.output}_spectra.npz"
    np.savez(data_file,
             k_iso=k_iso, power_iso=mean_power, power_iso_std=std_power,
             k_mom=k_mom, m20=mean_m20, m20_std=std_m20,
             m11=mean_m11, m11_std=std_m11,
             m02=mean_m02, m02_std=std_m02,
             A=A, theta_deg=theta_deg)
    print(f"  Saved spectral data to {data_file}")

    # 7. 绘图
    fig_file = f"{args.output}_spectra.png"
    plot_spectra(k_iso, mean_power, std_power,
                 k_mom, mean_m20, std_m20, mean_m11, std_m11, mean_m02, std_m02,
                 A, theta_deg, save_fig=fig_file)
    print("Processing complete.")


if __name__ == "__main__":
    main()
