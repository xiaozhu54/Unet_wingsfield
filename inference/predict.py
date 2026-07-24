"""
Inference utilities: load a trained model and predict flow fields.
"""
import os
import numpy as np
import torch

from models.unet import UNet, build_unet
from data.preprocess import Normalizer, split_input_target
from config.train_config import TrainConfig, InferenceConfig


class Predictor:
    """Load a trained model and run inference on new inputs."""

    def __init__(self, model: UNet, normalizer: Normalizer,
                 config: TrainConfig = None, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.config = config or TrainConfig()
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, norm_path: str,
                        config: TrainConfig = None,
                        device: str = "cpu") -> "Predictor":
        """Load predictor from saved model weights and normalizer."""
        cfg = config or TrainConfig()
        model = build_unet(cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        # Support both full checkpoint and state-dict-only saves
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        normalizer = Normalizer.load(norm_path)
        return cls(model, normalizer, cfg, device)

    @torch.no_grad()
    def predict(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Predict flow field for a given input.

        Args:
            raw_data: (16, H, W) or (N, 16, H, W) array

        Returns:
            (N, 4, H, W) predicted flow field (rho, p, u, v)
        """
        if raw_data.ndim == 3:
            raw_data = raw_data[np.newaxis]  # (1, 16, H, W)

        # Normalize
        normed = self.normalizer.normalize(raw_data.astype(np.float32))
        inputs, _ = split_input_target(normed, self.config)

        x = torch.from_numpy(inputs).to(self.device)
        y_pred = self.model(x).cpu().numpy()

        return y_pred

    def predict_and_denormalize(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Predict and denormalize the flow field back to physical scale.

        Returns:
            (N, 4, H, W) denormalized predictions
        """
        pred = self.predict(raw_data)  # (N, 4, H, W)
        # Place predictions in the flow channels of a 16-channel array
        dummy = np.zeros((pred.shape[0], 16, *pred.shape[2:]), dtype=np.float32)
        dummy[:, self.config.flow_channels] = pred
        denormed = self.normalizer.denormalize(dummy, self.config.flow_channels)
        return denormed[:, self.config.flow_channels]
