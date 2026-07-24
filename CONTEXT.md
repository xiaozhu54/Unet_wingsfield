# UNet Airfoil Flow Prediction

This context names the domain concepts used by the airfoil compressible-flow surrogate model. It exists to keep data, training, evaluation, and analysis discussions precise.

## Language

**Flow Sample**:
A single CFD-derived `.npz` case containing one airfoil geometry, one freestream condition, grid metrics, and the target flow field on a 128x128 tensor.
_Avoid_: file, case, data point when the full 16-channel tensor is meant

**Freestream Condition**:
The scalar inlet condition encoded by Mach number, angle of attack, and Reynolds number for a Flow Sample.
_Avoid_: IC, operating point

**Grid Metrics**:
The nine input channels describing the coordinate transform and airfoil-adapted grid geometry used by the model alongside the freestream condition.
_Avoid_: mesh data, coordinates

**Flow Field Target**:
The four output channels predicted by the model: density, pressure, x-velocity, and y-velocity.
_Avoid_: label, output when the physical field is meant

**Normalizer**:
The per-channel min-max scaling state fitted only to the training split and reused for validation, held-out test evaluation, and inference.
_Avoid_: scaler, normalization file

**Held-Out Test Split**:
The subset produced by the configured deterministic train/validation/test split and reserved as the default source of formal model metrics.
_Avoid_: test set when it means the entire data directory

**Split Manifest**:
A persisted record assigning each Flow Sample to train, validation, or held-out test membership for a specific experiment.
_Avoid_: random split, split seed when the concrete membership is meant

**Flow Error Summary**:
The collection of per-channel MAE, MSE, and relative L2 metrics used to judge prediction quality for Flow Field Targets.
_Avoid_: score, accuracy

**Pressure Coefficient Target**:
The near-wall or surface pressure coefficient `Cp` derived from the pressure channel and freestream Mach number. This is now the primary engineering objective.
_Avoid_: treating pressure-field MAE as a complete substitute for Cp quality

**Surface Sampling Contract**:
The rule used to extract upper/lower Cp curves from a predicted or reference flow field. Current default is an approximate transformed-grid row pair, not a true wall mask.
_Avoid_: calling approximate rows the airfoil surface without qualification

**Cp Error Summary**:
The held-out-test MAE, MSE, and max absolute error of the sampled upper/lower Cp curves.
_Avoid_: judging Cp quality only from selected plots

**Legacy Baseline**:
An experiment result kept for comparison even though its training or evaluation contract is no longer the current project default.
_Avoid_: current result, final model
