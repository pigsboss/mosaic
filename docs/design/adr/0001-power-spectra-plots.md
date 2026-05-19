# ADR-0001: Handling of Power Spectra and Masking for Aperture Observations

Date: 2025-03-23

## Status

Accepted

## Context

The project simulates geostationary ocean observations using synthetic apertures (full, Golay‑3, Golay‑9) applied to realistic sea surface fields (SST, SSH).  
After applying an aperture’s optical transfer function (OTF) in the frequency domain, we need to compute and compare spectral diagnostics – radial power spectra, anisotropy degree, and principal orientation – between the true (unobserved) fields and the aperture‑degraded observed fields.

The OTF of an aperture has a finite (and for sparse arrays, non‑circular) support in spatial‑frequency space. Signal outside this support is not recovered; including it contaminates spectral statistics with noise. Moreover, different apertures have different cutoffs, so direct comparison requires a unified frequency coordinate system.

## Decision

We implement the following design choices in `observables.py`:

1. **OTF‑based mask for valid frequency content**  
   Aperture OTF is computed on the same angular‑frequency grid as the scene. A binary mask is derived from the OTF using a relative threshold (|OTF| > 1% of peak). This mask identifies the 2D region where the aperture can actually transfer information.

2. **Unified radial bins across all fields**  
   All radial power spectra, anisotropy, and orientation curves use exactly the same wavenumber bins, defined by the full Nyquist range of the true field's grid. The maximum wavenumber `k_max` is always `np.max(k_rad)` (global Nyquist), independent of any mask. This guarantees that every curve shares the same x‑axis coordinates.

3. **Masked arrays for bins with no data**  
   When an aperture mask is provided, the spectral functions (`radial_power_spectrum`, `moment_anisotropy`) only accumulate statistics from pixels inside the mask. Bins that receive no valid pixels are **masked out** using NumPy's `np.ma.MaskedArray`.  
   This prevents false zero values from appearing on plots (e.g., anisotropy = 0 or orientation = 0° could be misinterpreted as real measurements) and allows matplotlib to automatically skip such points, breaking the line at the cutoff.

4. **Artificial vertical offsets for observation curves**  
   To separate the visual traces of true and multiple apertures on the same panel, observed spectra are shifted vertically:
   - Radial power and anisotropy: multiplied by a constant (10, 100, 1000).
   - Orientation: a constant is added (360°, 720°, 1080°).  
   The true field curves remain unshifted; the offsets are purely for readability and do not affect the physical interpretation.

5. **X‑axis limit set to the common observable range**  
   The x‑axis of the comparison plots is limited to `k_max_common`, the largest wavenumber present in any aperture’s mask (across all states). This clips the plot beyond the highest physically measurable frequency, removing noise‑only high‑frequency tails and making the effective bandwidth immediately visible.

## Consequences

- The spectral analysis is physically rigorous: only actually observed frequencies contribute to the statistics.  
- All curves are plotted on the same wavenumber canvas, enabling direct comparison across apertures and with the true field.  
- The plots are visually clean: lines naturally terminate at the aperture cutoff without artificial gaps or zero‑valued points.  
- The masking approach is safe for logarithmic axes (no zero‑value warnings) and preserves the meaning of genuine zero values (e.g., anisotropy=0).  
- The artificial offsets require attention when interpreting absolute values, but the legend and unchanged true curve make the relative behaviour clear.  
- The ADR ensures that any future contributor can understand and reproduce the spectral analysis methodology.
