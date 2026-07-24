# Run gen3 as the Channel-Weighted Loss Experiment

The `gen3` experiment should return to the `transpose` upsampling path and change only the flow loss channel weights, with higher weight on density and pressure. The initial weights should be `rho=1.5`, `p=1.5`, `u=1.0`, and `v=1.0`. This keeps the data contract, architecture, optimizer, epoch count, and stopping behavior comparable to `gen1`, while testing whether the known weak channels can be improved before introducing a larger architectural change.
