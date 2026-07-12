from __future__ import annotations

import numpy as np
import torch


class TokenGradCAM:
    """Grad-CAM for transformer token features shaped as [B, N, C]."""

    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module, token_grid_size: int):
        self.model = model
        self.target_module = target_module
        self.token_grid_size = token_grid_size
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = [
            target_module.register_forward_hook(self._save_activations),
            target_module.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output):
        del module, inputs
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        del module, grad_input
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target_index: int | None = None):
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        if target_index is None:
            target_index = int(logits.argmax(dim=1).item())

        score = logits[:, target_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activations = self.activations
        gradients = self.gradients
        if activations.ndim != 3 or gradients.ndim != 3:
            raise ValueError(
                "Expected target module tensors shaped [B, N, C], got "
                f"{tuple(activations.shape)} and {tuple(gradients.shape)}."
            )

        expected_tokens = self.token_grid_size * self.token_grid_size
        if activations.shape[1] != expected_tokens:
            raise ValueError(
                f"Target module produced {activations.shape[1]} tokens, expected "
                f"{expected_tokens} for a {self.token_grid_size}x{self.token_grid_size} grid."
            )

        weights = gradients.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=2))
        cam = cam.view(-1, 1, self.token_grid_size, self.token_grid_size)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        probabilities = torch.softmax(logits.detach(), dim=1)
        return cam.detach().cpu().numpy(), logits.detach().cpu(), probabilities.cpu(), target_index

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()


def denormalize_imagenet(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).detach().cpu().numpy()


def overlay_heatmap(base_image: np.ndarray, cam_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heatmap = cm.get_cmap("jet")(cam_map)[..., :3]
    overlay = (1.0 - alpha) * base_image + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0)
