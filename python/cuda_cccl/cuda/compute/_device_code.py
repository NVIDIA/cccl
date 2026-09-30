# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Public type for passing compiled device-side operator code into ``Op`` /
``RawOp``. Wraps the bytes together with their format tag so the two cannot
get out of sync as they flow through the binding layer.

Lives in its own module to keep ``op.py`` and the Cython ``_bindings_impl``
free of import-cycle headaches: the Cython side duck-types on
``(op_bytes, kind)`` attributes and never imports the class directly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tag values mirror the C-side ``cccl_op_code_type`` enum.
_VALID_KINDS = ("ltoir", "llvm_ir", "cpp_source")


@dataclass(frozen=True)
class DeviceCode:
    """A compiled-or-source device-code blob ready to hand to ``Op``.

    Args:
        op_bytes: the raw blob (LTO-IR, LLVM bitcode, or C++ source bytes).
        kind: one of ``"ltoir"`` (default), ``"llvm_ir"``, ``"cpp_source"``;
            tells the backend how to interpret ``op_bytes``.

    For most uses you don't construct ``DeviceCode`` directly — the internal
    JIT-compile helpers return one, and the iterator/algorithm machinery
    forwards them. Construct explicitly when feeding a ``RawOp`` from outside
    the default pipeline.
    """

    op_bytes: bytes
    kind: str = "ltoir"

    def __post_init__(self):
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"DeviceCode.kind must be one of {_VALID_KINDS!r}; got {self.kind!r}"
            )
