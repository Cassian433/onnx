# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._common_classifier import (
    compute_probit,
    expit,
    softmax,
    softmax_zero,
)
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LinearRegressor(OpRunAiOnnxMl):
    def _run(
        self, x, coefficients=None, intercepts=None, targets=1, post_transform=None
    ):
        if x.dtype != np.float64:
            x = x.astype(np.float32)
        coefficients = np.array(coefficients, dtype=x.dtype)
        n = coefficients.shape[0] // targets
        coefficients = coefficients.reshape(targets, n).T
        score = np.dot(x, coefficients)
        if intercepts is not None:
            score += np.array(intercepts, dtype=x.dtype)
        if post_transform == "NONE":
            pass
        elif post_transform == "SOFTMAX":
            score = softmax(score.copy())
        elif post_transform == "LOGISTIC":
            score = expit(score)
        elif post_transform == "SOFTMAX_ZERO":
            score = softmax_zero(score.copy())
        elif post_transform == "PROBIT":
            for i in range(score.shape[0]):
                for j in range(score.shape[1]):
                    score[i, j] = compute_probit(score[i, j])
        else:
            raise NotImplementedError(
                f"post_transform: {post_transform!r} is not implemented."
            )
        return (score,)
