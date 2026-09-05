# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    Any,
    ThreadGroup,
    inspect,
    ir,
)


class _LoadStorePlanning:
    def _lower_load_store(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        if is_common_root:
            if operation == "load":
                if not self._thread_data_operand_state(
                    operation, "output", bound.arguments["output"]
                ):
                    raise TypeError(
                        "cuda.coop.load requires output to be a fixed-size ThreadData payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
            else:
                value = bound.arguments["value"]
                if self._array_operand_state(operation, value) and (
                    not self._thread_data_operand_state(operation, "value", value)
                ):
                    raise TypeError(
                        "cuda.coop.store accepts only a scalar or fixed-size ThreadData value payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
                    )
        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs["algorithm"] = bound.arguments["algorithm"]
        if is_common_root:
            factory_kwargs["_common_root_operation"] = operation
        if not self._is_none(bound.arguments["valid_items"]):
            factory_kwargs["num_valid_items"] = bound.arguments["valid_items"]
        if operation == "load" and (not self._is_none(bound.arguments["oob_default"])):
            factory_kwargs["oob_default"] = bound.arguments["oob_default"]
        if group.kind in {"warp", "threads_within_warp"}:
            factory_kwargs["_physical_warp_tile_origin"] = True
            factory_kwargs["offset"] = (
                0
                if self._is_none(bound.arguments["offset"])
                else bound.arguments["offset"]
            )
        elif not self._is_none(bound.arguments["offset"]):
            factory_kwargs["offset"] = bound.arguments["offset"]
        if operation == "store":
            factory_kwargs["_group_root_store"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            if group.kind != "block":
                raise NotImplementedError(
                    "cuda.coop.numba_mlir Load/Store TempStorage is supported only for block groups"
                )
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]
        if operation == "load":
            runtime_args = [bound.arguments["source"], bound.arguments["output"]]
            return_alias = bound.arguments["output"]
        else:
            runtime_args = [bound.arguments["destination"], bound.arguments["value"]]
            return_alias = None
        return self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=return_alias,
        )


__all__ = ["_LoadStorePlanning"]
