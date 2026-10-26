# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Scaler(OpRunAiOnnxMl):
    def _run(self, x, offset=None, scale=None):
        if x.dtype != np.float64:
            x = x.astype(np.float32)
        dx = x - offset
        return ((dx * scale).astype(np.float32),)
