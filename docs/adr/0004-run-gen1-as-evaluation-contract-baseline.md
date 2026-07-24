# Run gen1 as the Evaluation-Contract Baseline

The next formal training run should use `results_gen1_evalfix/` and change only the data/evaluation contract: split manifest persistence and train-only normalizer fitting. Model architecture, loss function, optimizer settings, epoch count, and stopping behavior should stay unchanged so the run becomes a clean baseline for judging later model changes.
