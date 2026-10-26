# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class CategoryMapper(OpRunAiOnnxMl):
    def _run(
        self,
        x,
        cats_int64s=None,
        cats_strings=None,
        default_int64=None,
        default_string=None,
    ):
        if np.issubdtype(x.dtype, np.integer):
            classes = dict(zip(cats_int64s, cats_strings, strict=True))
            default = default_string
            dtype = object
        else:
            classes = dict(zip(cats_strings, cats_int64s, strict=True))
            default = default_int64
            dtype = np.int64
        lookup = np.vectorize(lambda value: classes.get(value, default), otypes=[dtype])
        return (lookup(x),)
