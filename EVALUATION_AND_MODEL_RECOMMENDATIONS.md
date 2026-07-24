# Evaluation and Optimization Route

## What Matters Now

The final target is accurate and fast `Cp`, not just low full-field `(rho,p,u,v)` error. Full-field accuracy remains useful because Cp depends on the local pressure field and surrounding compressible-flow structure, but model selection should report Cp directly.

## Current Comparison

| Gen | Main change | Test normalized L1 | Mean Cp MAE | Result |
|---|---|---:|---:|---|
| gen1 | strict split + train-only normalizer | 0.0019907 | not measured | clean baseline |
| gen2 | bilinear upsample | 0.0020635 | not measured | worse |
| gen3 | channel weights `rho,p=1.5` | 0.0016954 | 0.0087596 | best full-field |
| gen4 | Attention UNet | 0.0019409 | not measured | worse than gen3 |
| gen5 | Cp loss weight 0.5 | 0.0029229 | 0.0029325 | Cp better, flow worse |
| gen6a | gen3 fine-tune, Cp loss 0.1 | 0.0015482 | 0.0048405 | best full-field |
| gen6b | gen3 fine-tune, Cp loss 0.2 | 0.0016187 | 0.0044083 | best current balance |
| gen7 | continue from gen6b, Cp loss 0.2, LR 5e-5, 160 epochs | 0.0014717 | 0.0031854 | current best balance |
| gen8 | continue from gen7, local Cp focus, negative-AoA lower weight | 0.0014612 | 0.0029982 | current best local/Cp balance |

## Recommendation

Use `results_gen8_cpft_local_w020/best_model.pth` as the current default balanced checkpoint.

Keep `results_gen3_channel_weighted/best_model.pth` as the stable pre-Cp baseline. Use `results_gen5_cp_loss/best_model.pth` only as a high-Cp-weight proof of direction.

## Next Experiments

1. Recover or define true wall/surface indices from the grid generation pipeline.
2. Continue staged fine-tuning from gen7/gen8 with Cp weights around `0.1-0.2`, but monitor extreme negative-AoA lower-surface Cp and worst Cp max error.
3. Add region-aware weighting for the difficult negative-AoA lower-surface cases.
4. Select checkpoints by a Pareto rule: Cp MAE improves while full-field normalized L1 does not exceed gen3 by more than an agreed tolerance.
5. Add region-aware surface weighting around leading edge, pressure recovery/shock region, and trailing edge after the surface contract is fixed.
