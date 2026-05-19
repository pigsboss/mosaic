# Proposal 0002: Hydrodynamic Constraints for Realistic Time Evolution

Date: 2025-03-25

## Status

Draft (debugging in progress)

## Background and Motivation

Building upon the single‑frame sea‑state generation and aperture observation pipeline
(ADR‑0001/ADR‑0002), we required continuous time series of SST and SSH to study
temporal statistics and to produce smooth animations.  
The initial Ornstein‑Uhlenbeck (AR(1)) model produced flickering images that visually
resembled a gas rather than a liquid. This motivated the introduction of physical
constraints that make the synthetic fields behave more like an incompressible
geophysical fluid.

This proposal documents the design decisions that were taken to add fluid‑dynamical
realism while retaining the exact control over the spatial power spectrum that
the original generator provided.

## Overview of the Approach

We keep the **parameterised state script** (calm → Langmuir → turbulent) with smooth
interpolation of spectral exponents, directionality, discrete peaks, etc.  
The Fourier synthesis framework is preserved: the target power spectrum
\(P(k,\theta)\) is enforced by an AR(2) process in Fourier space.  
To obtain liquid‑like motion we add:

- a **second‑order time‑stepping** (AR(2)) for continuous first derivatives,
- **Hermitian half‑plane evolution** to guarantee a real output and phase coherence,
- a **divergence‑free velocity field** synthesised from a streamfunction,
- **semi‑Lagrangian advection** with sub‑stepping to transport the scalar fields,
- a **mean shear** for Langmuir circulation, and
- an **innovation damping factor** that reduces random forcing and lets the
  flow dominate the evolution.

## Core Design Elements

### 1. Second‑order AR(2) Process

Each complex Fourier coefficient \(a_k\) evolves as a discrete second‑order process:

\[
a_{k}^{(n+1)} = 2p_k a_{k}^{(n)} - p_k^2 a_{k}^{(n-1)} + \sigma_k w_t ,
\]

where \(p_k = \exp(-\Delta t / \tau_k)\) and \(\sigma_k\) is chosen so that the
stationary variance of \(a_k\) equals the target power spectrum \(P_k\) at every
time step. This formulation guarantees a **continuous time derivative** and
eliminates the frame‑to‑frame jumps that plagued the AR(1) model.

The correlation time \(\tau_k\) is scale‑dependent:
\[
\tau(k) = \tau_0 \left( \frac{k_{\min}}{k} \right)^{\tau_\alpha},
\]
giving long memory to large scales and shorter memory to small scales.

### 2. Hermitian Half‑Plane Constraint

Only the Fourier coefficients of one half of the frequency plane are evolved;
the other half is set by the complex conjugate of the active half at every
time step. This guarantees a purely real spatial field after inverse FFT
and eliminates the spurious phase discontinuities that occurred when both
half‑planes were updated independently.

### 3. Divergence‑Free Velocity Field

To advect the scalar fields we generate a **streamfunction** \(\psi(x,y,t)\)
with the same AR(2) spectral model that is used for SST and SSH.
The velocity components are obtained via centred differences:

\[
u = \frac{\partial \psi}{\partial y},\qquad
v = -\frac{\partial \psi}{\partial x},
\]

which automatically satisfy \(\nabla \cdot \mathbf{u} = 0\).  
The streamfunction parameters (spectral exponent, directionality,
correlation time) are stored in a `velocity` sub‑dictionary inside
`state_params` and are interpolated between states exactly like the
scalar parameters.

### 4. Semi‑Lagrangian Advection with Sub‑Stepping

Scalar fields are advanced by solving the advection equation
\(\partial Q / \partial t + \mathbf{u} \cdot \nabla Q = 0\) using a
backward‑characteristics scheme. For each time step we perform
**multiple sub‑steps** (currently 3) to improve accuracy and reduce
numerical diffusion. Bicubic interpolation on a periodic grid is used
to evaluate the field at the departure points.

### 5. Mean Shear for Langmuir Circulation

When the active sea state is “langmuir”, a steady east‑west shear
\(u_{\text{shear}} = U_0 \sin(\pi y / L_y)\) is added to the stochastic
velocity field. This produces the characteristic streak drifting and
rolling associated with Langmuir cells, without altering the
streamfunction itself.

### 6. Innovation Damping Factor

To prevent the random AR(2) innovation from acting as an internal volume
source (which makes the fluid appear compressible), the noise amplitude
is multiplied by an **innovation factor** \(\gamma = 0.3\) whenever
advection is active. Consequently the scalar evolution is dominated by
transport, and the field behaves as a nearly passive tracer in an
incompressible flow.

## Parameter Tuning and Visual Quality

- **Memory time parameters**  
  \(\tau_0\) and \(\tau_\alpha\) control the temporal coherence across scales.
  Values \(\tau_0 \sim 3600\text{–}7200\ \text{s}\) and
  \(\tau_\alpha \sim 0.4\text{–}0.55\) give a good balance between slow
  large‑scale drift and lively small‑scale texture.

- **Visual jumpiness**  
  When \(\tau_\alpha\) is too large, high‑wavenumber modes lose memory very
  quickly and the animation flickers. The current implementation avoids this
  by keeping \(\tau(k) \gg \Delta t\) for all resolved wavenumbers.

- **Liquid‑like appearance**  
  The combination of innovation damping, sub‑stepped advection, and
  smooth velocity spectra (velocity spectral exponent \(\ge 4\)) suppresses
  compressible‑like expansions and gives the flow a natural, watery feel.

## Limitations and Future Work

- The model does not yet contain **non‑linear energy cascade**,
  **dissipation**, physical **SST‑SSH coupling**, or **rotation**.
- It remains a kinematic / statistical generator rather than a full
  fluid‑dynamics solver.
- Possible extensions include embedding a 2D quasi‑geostrophic model
  or adding a stochastic vortex method to enhance realistic turbulent
  mixing.

---

*This document records the design rationale for the current
`seasurface_dynamic.py` implementation and will evolve as the model
is refined.*
