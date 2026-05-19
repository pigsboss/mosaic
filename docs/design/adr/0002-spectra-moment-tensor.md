# ADR-0002: Spectral Moment Tensor for Multi‑Scale Analysis of SST and SSH Fields

Date: 2026-05-20

## Status

Accepted

## Context

We simulate geostationary ocean observations of sea surface temperature (SST) and sea surface height anomaly (SSH) under three characteristic sea states (calm, Langmuir circulation, turbulent cascade).  
The traditional radial power spectrum (isotropic) captures the energy distribution across scales but discards directional information. To quantify the **orientation and degree of alignment** of spatial structures at each wavenumber, we need a diagnostic that respects the aperture’s limited frequency support and can be applied uniformly to both true and observed fields.

## Decision

We compute the **spectral moment tensor** from the 2D power spectral density (PSD) and derive two scalar quantities per radial wavenumber bin:

1. **Anisotropy degree** \( A(k) = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2} \), where \(\lambda_1, \lambda_2\) are the eigenvalues of the \(2\times 2\) structure tensor \(\begin{pmatrix} M_{20} & M_{11} \\ M_{11} & M_{02} \end{pmatrix}\) formed by weighted moments of the local frequency vector. \(A=0\) indicates perfect isotropy; \(A=1\) indicates all spectral power aligned along a single direction.

2. **Principal orientation** \( \theta(k) = \frac{1}{2}\arctan\!\left(\frac{2M_{11}}{M_{20} - M_{02}}\right) \), measured counter‑clockwise from the positive \(k_x\)-axis (east–west). This gives the dominant direction of structures at scale \( \sim 1/k \).

### Implementation details in `observables.py`

- The moments are computed on the **unshifted** frequency grid (DC at (0,0)) to keep the phase relationship of the Fourier transform.  
- For each radial bin \( [k_i, k_{i+1}) \), all pixels that fall within the bin (and, when an aperture mask is provided, inside the mask) contribute to the sums.  
- The moment tensor is formed by the weighted average of \(k_x^2\), \(k_x k_y\), and \(k_y^2\):  
  \[
  M_{20} = \frac{\sum P(k)\, k_x^2}{\sum P(k)}, \quad
  M_{11} = \frac{\sum P(k)\, k_x k_y}{\sum P(k)}, \quad
  M_{02} = \frac{\sum P(k)\, k_y^2}{\sum P(k)}.
  \]
- Bins that receive no valid data (e.g., beyond the aperture cutoff) result in **masked values** using NumPy’s `np.ma.MaskedArray`. This prevents misinterpretation of “no measurement” as zero anisotropy or zero orientation.  
- The same fixed wavenumber bins as in the radial power spectrum are used, ensuring identical \(k\)-centers for all curves.  
- The moment‑based anisotropy is **independent of the power scaling** – it measures shape, not amplitude.

### Application to the three sea states

- **Calm**: Very steep SSH spectrum (α=5) → only the lowest \(k\) bins contain significant power; the moment tensor there is expected to show near isotropy (\(A\approx0\)) with little variation in θ. SST is slightly less steep but still red.  
- **Langmuir**: Directional concentration \(s=2\) and discrete peaks at ∼45 and 90 cyc/km — the moment tensor reveals elevated \(A(k)\) at the corresponding wavenumbers and a consistent θ ≈ 0°, reflecting the streaks oriented east–west.  
- **Turbulent**: Broad Kolmogorov cascade (\(α≈1.667\) for SST, 2.5 for SSH) with weak anisotropy (\(s=0.3\)) — \(A(k)\) stays low, and θ may wander as the noise of fully developed turbulence dominates.

### Integration with the aperture mask

- The moment tensor uses the **same OTF‑derived mask** as the radial power spectrum.  
- For observed fields, only Fourier components that survive the aperture contribute to the moments, yielding physically meaningful anisotropy and orientation values up to the aperture cutoff.

## Consequences

- **Physical insight**: The moment tensor separates the shape and orientation of the 2D spectrum from its magnitude, enabling identification of anisotropic features (like Langmuir streaks) at specific scales.  
- **Consistent framework**: The analysis shares the same mask, binning, and masked‑array conventions established in ADR‑0001, guaranteeing comparable diagnostics for both true and observed fields.  
- **Visual clarity**: In the comparison plots, masked bins are automatically skipped by matplotlib, so the curves end naturally at the aperture’s effective bandwidth, avoiding false features.  
- **Robustness**: Using weighted averages (rather than e.g., fitting ellipses to binned data) makes the metric resilient to gaps in the OTF (especially for Golay arrays) and to noisy bins.  
- **Limitation**: The moment tensor assumes that the local power within a radial bin can be characterised by a single elongated ellipsoid. For multi‑modal directional distributions, this single‑mode description may not capture the full complexity; however, it remains a standard and sufficient metric for the regimes considered here.
