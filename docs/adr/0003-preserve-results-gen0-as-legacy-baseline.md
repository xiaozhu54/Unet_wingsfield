# Preserve results_gen0 as a Legacy Baseline

The `results_gen0` run should be treated as a legacy baseline and not overwritten by future experiments. Its model and normalizer were produced before the current split-manifest and train-only normalizer contract, so keeping it separate prevents old and new evaluation assumptions from being mixed.
