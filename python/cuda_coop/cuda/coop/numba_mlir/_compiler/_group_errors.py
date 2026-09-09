# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Derived group-planning errors with readable, wrapped diagnostics."""

from textwrap import fill

from ._group_planner_support import GroupRewriteError


def _wrap_diagnostic(message):
    return fill(message, width=80, break_long_words=False, break_on_hyphens=False)


class InvalidGroupSelectorError(ValueError):
    def __init__(self, operation, parameter, choices):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.{operation} {parameter} must be one of: {choices}; use "
                f"a backend-qualified import for backend-only controls",
            )
        )


class CyclicArrayProvenanceError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} could not resolve cyclic array "
                f"provenance to a concrete scalar or array value",
            )
        )


class InconsistentArrayExtentError(GroupRewriteError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.numba_mlir array aliases have inconsistent "
                "items_per_thread extents",
            )
        )


class InconsistentTupleExtentError(GroupRewriteError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.numba_mlir tuple projections have inconsistent "
                "items_per_thread extents",
            )
        )


class UnknownResultExtentError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} could not infer a static "
                f"items_per_thread extent for its non-mutating result",
            )
        )


class NonConstantThreadGroupError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} requires a compile-time "
                f"ThreadGroup from this_*()",
            )
        )


class NonConstantGroupArgumentError(GroupRewriteError):
    def __init__(self, value_name):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir group arguments that shape provider "
                f"specialization must be compile-time constants; got "
                f"{value_name!r}",
            )
        )


class InconsistentLoopTupleExtentError(GroupRewriteError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.numba_mlir loop-carried tuple payloads have "
                "inconsistent items_per_thread extents",
            )
        )


class InconsistentLoopPayloadExtentError(GroupRewriteError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.numba_mlir loop-carried payloads have inconsistent "
                "items_per_thread extents",
            )
        )


class EscapingGroupDescriptorError(GroupRewriteError):
    def __init__(self, names):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir ThreadGroup/ThreadHierarchy values are "
                f"compile-time descriptors and may only feed this_*(), group_by(), "
                f"or group-first primitives; descriptor use involving {names!r} "
                f"would escape to runtime",
            )
        )


class InvalidLoadStoreAlgorithmError(ValueError):
    def __init__(self, operation, choices):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} algorithm must be one of: {choices}",
            )
        )


class UnsupportedLoadStoreAlgorithmError(NotImplementedError):
    def __init__(self, operation, token):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} algorithm {token!r} is not "
                f"executable; only 'direct' is currently supported",
            )
        )


class UnknownBlockDimensionError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} requires an exact block "
                f"dimension before provider selection",
            )
        )


class UnknownLoadStoreExtentError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} requires a static "
                f"items_per_thread extent before provider selection",
            )
        )


class UnknownLoadStoreDtypeError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} could not infer a dtype before "
                f"provider selection",
            )
        )


class MemoryDtypeMismatchError(TypeError):
    def __init__(self, operation, memory_dtype, payload_dtype):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} memory dtype {memory_dtype} "
                f"does not match payload dtype {payload_dtype}",
            )
        )


class NonConstantTempStorageError(GroupRewriteError):
    def __init__(self, operation):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir.{operation} temp_storage must resolve to a "
                f"compile-time TempStorage descriptor",
            )
        )


class UnsupportedLoadStoreTargetError(GroupRewriteError):
    def __init__(self, target):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir Load/Store received an unsupported lowering "
                f"target {target!r}",
            )
        )


class DefaultDtypeMismatchError(TypeError):
    def __init__(self, value_dtype, payload_dtype):
        super().__init__(
            f"cuda.coop.numba_mlir.load runtime oob_default dtype {value_dtype}\n"
            f"does not match payload dtype {payload_dtype}"
        )


class UnknownLoadStoreProviderError(GroupRewriteError):
    def __init__(self, provenance):
        super().__init__(
            _wrap_diagnostic(
                f"cuda.coop.numba_mlir Load/Store received an unknown CUB "
                f"implementation provenance {provenance!r}",
            )
        )


class PortableLoadPayloadError(TypeError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.load requires output to be a fixed-size ThreadData "
                "payload in the portable API; use cuda.coop.numba_mlir for "
                "backend-qualified local-array payload support",
            )
        )


class PortableStorePayloadError(TypeError):
    def __init__(self):
        super().__init__(
            _wrap_diagnostic(
                "cuda.coop.store accepts only a scalar or fixed-size ThreadData "
                "value payload in the portable API; use cuda.coop.numba_mlir for "
                "backend-qualified local-array payload support",
            )
        )


class UnsupportedLoadStoreGroupError(NotImplementedError):
    def __init__(self, operation, reason=None):
        message = (
            f"cuda.coop.numba_mlir.{operation} currently lowers only "
            "this_block() groups through CUB"
        )
        if reason is not None:
            message += f": {reason}"
        super().__init__(_wrap_diagnostic(message))
