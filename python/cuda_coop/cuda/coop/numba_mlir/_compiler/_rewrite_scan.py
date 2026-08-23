# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan callback and aggregate factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
)


class _ScanRewrite:
    def _finalize_scan_factory_kwargs(
        self,
        *,
        runtime_arg_count: int,
        factory_kwargs: dict[str, object],
    ) -> None:
        from .._stateful_function import StatefulFunction

        prefix_callback = factory_kwargs.get("block_prefix_callback_op")
        if prefix_callback is None:
            prefix_callback = factory_kwargs.get("prefix_op")
        stateful = isinstance(prefix_callback, StatefulFunction)
        expected_count = 3 if stateful else 2
        if runtime_arg_count != expected_count:
            requirement = (
                "requires a third runtime state argument"
                if stateful
                else "accepts only input and output runtime arguments"
            )
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'scan' {requirement}."
            )


__all__ = ["_ScanRewrite"]
