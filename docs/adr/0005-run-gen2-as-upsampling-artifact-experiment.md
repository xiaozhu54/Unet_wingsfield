# Run gen2 as the Upsampling Artifact Experiment

The `gen2` experiment should target upsampling artifacts by changing only the decoder upsampling block from transposed convolution to an `Upsample + Conv2d` path. The upsampling method should be a configuration option, with `transpose` as the default for checkpoint compatibility and `bilinear` used for `gen2`. It should keep the `gen1` split manifest, train-only normalizer contract, model width/depth, loss, optimizer, epoch count, and stopping behavior unchanged so any metric or visual change can be attributed mainly to the upsampling method.
