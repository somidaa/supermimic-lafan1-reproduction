"""Mixed-precision state shared by the world-model and policy trainers.

fp16 autocast is applied to the MLP matmuls (the dominant cost); state integration, forward
kinematics and the analytic reward/termination heads stay fp32, exactly as in the verified
reference reproduction (which uses bf16 for the same reason). A per-optimizer GradScaler keeps
the small normalised-error gradients representable in fp16.
"""

from __future__ import annotations

import torch


class AmpState:
    def __init__(self, enabled: bool = False, device: str = "cuda"):
        self.enabled = bool(enabled)
        self.device = device
        self.dtype = torch.float16
        self.scaler = torch.amp.GradScaler(device, enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.enabled)
