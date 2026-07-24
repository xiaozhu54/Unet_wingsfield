# ADR 0008: Shift to Cp-First Optimization

## Status

Accepted.

## Context

The user clarified that the engineering goal is fast and accurate pressure coefficient `Cp` from freestream condition plus airfoil/grid input. The original UNet predicted the full flow field `(rho,p,u,v)`, which is useful but not identical to the final objective.

Observed errors around the leading edge and mid-chord pressure region show that good global flow-field loss does not guarantee best Cp behavior.

## Decision

Keep full-field prediction, but make Cp a first-class evaluation metric and optional training objective.

Use:

- `scripts/evaluate.py` for held-out full-field metrics.
- `scripts/evaluate_cp.py` for held-out Cp metrics.
- `scripts/analyze_flow.py` for visual flow/Cp inspection.

Gen5 adds a near-wall Cp loss on top of gen3 channel weighting.

Gen6 fine-tunes from gen3 with lower Cp weights instead of training the Cp objective from scratch.

## Consequences

Gen5 reduced held-out mean Cp MAE from `0.0087596` to `0.0029325`, proving that Cp-directed training is useful. It also worsened full-field normalized L1 from `0.0016954` to `0.0029229`, so Cp loss must be tuned and grounded in a better surface extraction contract.

The next priority is true wall/surface indexing, then lower Cp weights or staged fine-tuning from gen3.

Follow-up results:

- Gen6a (`cp_loss_weight=0.1`) reached full-field test L1 `0.0015482` and mean Cp MAE `0.0048405`.
- Gen6b (`cp_loss_weight=0.2`) reached full-field test L1 `0.0016187` and mean Cp MAE `0.0044083`.
- Gen7 continued from gen6b with LR `5e-5` for 160 epochs and reached full-field test L1 `0.0014717` and mean Cp MAE `0.0031854`.
- Gen8 continued from gen7 with local Cp weighting and negative-AoA lower-surface weighting, reaching full-field test L1 `0.0014612` and mean Cp MAE `0.0029982`.

Gen8 is the current recommended balance. Extreme negative-AoA lower-surface Cp remains a weak region and should be handled with a true surface contract rather than approximate transformed-grid rows.
