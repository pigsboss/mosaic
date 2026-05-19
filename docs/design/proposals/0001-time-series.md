# Proposal 0001: Dynamic Sea Surface Time Series Generation

Date: 2025-03-23

## Status

Draft

## Context

The current `seasurface.py` generates independent, static realizations of SST
and SSH for three discrete sea states (calm, Langmuir, turbulent). To study
temporal evolution, e.g., the transition between regimes and the persistence of
features, we need a time‑continuous model.

The core requirements are:
- Fields should evolve smoothly in time with physically plausible memory.
- Sea states should be able to transition in a controlled way.
- SST and SSH may share partial coherence (coupling).
- The output must be visualisable in Jupyter with HTML5 animations.

## Design Considerations (Three Steps)

### Step 1 – Time‑dependent state parameters
All state descriptors (`alpha`, `s`, `theta0`, `isotropic_component`, `peaks`)
become functions of time \(p(t)\). A “script” defines keyframes (t, state), and
a smooth interpolation fills between them. This eliminates the discrete state
switch and allows the same spectral synthesis framework to be used at each
instant.

### Step 2 – Temporal continuity via Ornstein–Uhlenbeck (O‑U) process
Instead of drawing independent Fourier phases at each time step, each complex
Fourier coefficient \(a(\vec{k},t)\) evolves as:

\[
a(\vec{k}, t+\Delta t) = \phi_k a(\vec{k}, t) + \sigma_k w_t,
\]

where \( \phi_k = \exp(-\Delta t / \tau(k)) \) introduces wavenumber‑dependent
memory, and \( \sigma_k \) is chosen so that the steady‑state variance matches
the target power spectrum \( P(\vec{k},t) \).

- Large scales (small \( k \)) are given longer correlation times.
- When the target spectrum changes (state transition), \( \phi_k \) and
  \( \sigma_k \) are updated, and the process naturally re‑equilibrates.

#### SST–SSH coupling (future step)
A common stochastic driver can be added to link the two fields without
rewriting the whole generator:

\[
a_{sst} \leftarrow \rho\, w_{c} + \sqrt{1-\rho^2}\, w_{sst},\qquad
a_{ssh} \leftarrow \rho\, w_{c} + \sqrt{1-\rho^2}\, w_{ssh}.
\]

The coherence \( \rho \) can depend on \( \vec{k} \) and on the state (e.g., higher
for Langmuir streaks).

### Step 3 – Jupyter HTML5 visualisation
- `matplotlib.animation.FuncAnimation` → `to_html5_video()` yields an
  `<video>` tag that can be embedded in a notebook.
- Companion plots (e.g., time‑varying radial spectrum) can be synchronized with
  the main animation.

## First Action: Temporal Parameter Interpolation + O‑U Generator

We start with Step 1 + Step 2 (single field, no coupling) because it delivers:
- Realistic, continuous SST/SSH sequences that still obey the per‑state
  anisotropic spectra.
- A natural interface (`script`) to define state transitions.
- Data that can be directly fed into `observables.py` for time‑resolved
  spectral analysis.

### Implementation plan (pseudocode)

#### 1. Parameter interpolation utility
A function `interpolate_state_params(state_a, state_b, fraction)` blends
two parameter dictionaries (each containing scalar keys like `alpha`, `s`,
`isotropic_component`, `theta0`, and an optional `peaks` list). Scalars are
linearly interpolated; peaks are faded in/out by adjusting amplitudes while
keeping positions and widths constant.

