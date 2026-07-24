# Use the Held-Out Test Split for Formal Metrics

Formal model metrics must come from the deterministic held-out test split, and the Normalizer must be fitted only on the training split. The earlier workflow fitted normalization statistics on the full corpus and evaluated the full data directory, which is convenient for smoke checks but leaks validation/test information and makes reported generalization metrics too optimistic.
