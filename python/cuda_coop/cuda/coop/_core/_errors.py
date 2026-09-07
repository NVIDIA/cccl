# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral errors shared by the semantic core and API dispatch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _BackendActivationFailure:
    backend: str
    reason_code: str
    cause: BaseException


class CoopCompilerContextRequiredError(RuntimeError):
    """A root cooperative operation requires an active Python DSL backend."""

    def __init__(
        self,
        feature: str,
        activation_failure: _BackendActivationFailure | None = None,
    ) -> None:
        self.feature = feature
        self.cause = None if activation_failure is None else activation_failure.cause
        self.backend = (
            None if activation_failure is None else activation_failure.backend
        )
        self.reason_code = (
            "compiler-context-required"
            if activation_failure is None
            else activation_failure.reason_code
        )
        activation_details = (
            None if self.cause is None else getattr(self.cause, "details", None)
        )
        self.details = {
            "feature": feature,
            "backend": self.backend,
            "cause_type": None if self.cause is None else type(self.cause).__name__,
            "cause_message": None if self.cause is None else str(self.cause),
            "activation_details": activation_details,
        }
        message = (
            f"cuda.coop.{feature} requires an active compiler backend; "
            "install a compatible backend or import cuda.coop.numba_mlir "
            "before tracing a kernel"
        )
        if activation_failure is not None:
            message += (
                f"; automatic {activation_failure.backend} activation failed "
                f"({activation_failure.reason_code}): "
                f"{activation_failure.cause}"
            )
            self.__cause__ = activation_failure.cause
        super().__init__(message)


__all__ = ["CoopCompilerContextRequiredError"]
