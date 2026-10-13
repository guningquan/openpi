from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # Setup image saving directory and counter
        self._image_save_dir = pathlib.Path("model_input_images")
        self._image_save_dir.mkdir(parents=True, exist_ok=True)
        self._inference_step = 0

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # Save images before passing to model
        SAVE_IMAGES = False
        if "image" in inputs and SAVE_IMAGES:
            self._inference_step += 1
            for img_key, img_data in inputs["image"].items():
                if hasattr(img_data, "shape"):
                    # Convert to numpy array (handle PyTorch tensors, JAX arrays, etc.)
                    if isinstance(img_data, torch.Tensor):
                        img_array = img_data.detach().cpu().numpy()
                    else:
                        img_array = np.asarray(img_data)
                    
                    # Remove batch dimension if present (shape: [1, H, W, C] -> [H, W, C] or [1, C, H, W] -> [C, H, W])
                    if img_array.ndim == 4 and img_array.shape[0] == 1:
                        img_array = img_array[0]
                    
                    # Handle CHW format (convert to HWC)
                    if img_array.ndim == 3:
                        # Check if it's CHW format (channel first: shape[0] is small like 3)
                        if img_array.shape[0] == 3 and img_array.shape[0] < img_array.shape[1] and img_array.shape[0] < img_array.shape[2]:
                            # Convert from CHW to HWC
                            img_array = np.transpose(img_array, (1, 2, 0))
                    
                    # Convert to uint8 format [0, 255] for saving
                    if img_array.dtype == np.uint8:
                        img_to_save = img_array
                    elif img_array.dtype in (np.float32, np.float64):
                        # Check if values are in [-1, 1] range (normalized) or [0, 1] range
                        if img_array.min() >= -1.1 and img_array.max() <= 1.1:
                            # Convert from [-1, 1] to [0, 255]
                            img_to_save = ((img_array + 1.0) / 2.0 * 255.0).astype(np.uint8)
                        elif img_array.min() >= -0.1 and img_array.max() <= 1.1:
                            # Convert from [0, 1] to [0, 255]
                            img_to_save = (img_array * 255.0).astype(np.uint8)
                        else:
                            # Assume already in [0, 255] range but as float
                            img_to_save = np.clip(img_array, 0, 255).astype(np.uint8)
                    else:
                        # Convert other types to uint8
                        img_to_save = np.clip(img_array, 0, 255).astype(np.uint8)
                    
                    # Ensure image is in [H, W, C] format and save
                    if img_to_save.ndim == 3 and img_to_save.shape[2] == 3:
                        # Save image
                        image = Image.fromarray(img_to_save)
                        save_path = self._image_save_dir / f"step_{self._inference_step:06d}_{img_key}.png"
                        image.save(save_path)
                        logger.info(f"Saved image '{img_key}' to {save_path}")
                    else:
                        logger.warning(f"Image '{img_key}' has unexpected shape {img_to_save.shape}, skipping save")

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
