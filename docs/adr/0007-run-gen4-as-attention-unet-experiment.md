# Run gen4 as the Attention UNet Experiment

The `gen4` experiment should use Attention UNet skip gates while keeping the `gen3` data contract, transpose upsampling, channel weights `rho=1.5`, `p=1.5`, `u=1.0`, `v=1.0`, optimizer, epoch count, and stopping behavior. The goal is to test whether attention-gated skip features improve density and pressure prediction near airfoil-adjacent and high-gradient regions beyond the `gen3` channel-weighted baseline.
