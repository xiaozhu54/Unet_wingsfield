# UNet Wing Field Project Summary

This file is intentionally short. Historical implementation notes were replaced by the current comparison-driven route. Use `agents.md` as the main project state document.

## Model

- Input: 12 channels = Mach, AoA, Reynolds number, and 9 grid-metric channels.
- Output: 4 channels = `rho`, `p`, `u`, `v`.
- Baseline architecture: 6-level UNet, `base_filters=32`, about 200M parameters.
- Optional architecture switches: bilinear upsample and Attention UNet gates.

## Current Route

The project target is Cp-first prediction. Full-field prediction is retained as physical context, but optimization must be judged with held-out Cp metrics.

| Model | Role |
|---|---|
| gen3 channel-weighted UNet | best full-field baseline |
| gen5 Cp-loss UNet | first Cp-targeted proof of direction |

## Known Limitation

Cp is currently sampled from approximate near-wall transformed-grid rows. A true wall index or surface mask is required before claiming final aerodynamic Cp accuracy.

## See Also

- `agents.md`: current state, experiment comparison, commands.
- `CONTEXT.md`: domain terms.
- `docs/adr/`: decisions behind the route.
