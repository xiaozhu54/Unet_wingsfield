# Persist Split Manifests

Each formal training run should persist a split manifest that records which Flow Samples belong to train, validation, and held-out test splits. Recomputing membership from a directory listing, ratios, and a random seed is fragile once datasets are merged, filtered, renamed, or partially regenerated, while a manifest keeps checkpoints and reported metrics auditable.
