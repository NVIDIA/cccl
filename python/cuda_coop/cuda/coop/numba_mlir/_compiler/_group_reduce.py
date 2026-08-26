# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduction IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    Any,
    Integral,
    ThreadGroup,
    inspect,
    ir,
)


class _ReducePlanning:
    def _lower_reduce(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("reduce", bound)
        broadcast = self._constant(bound.arguments["broadcast"])
        if not isinstance(broadcast, bool):
            raise TypeError(
                "cuda.coop.numba_mlir.reduce broadcast must be a compile-time bool"
            )

        value = bound.arguments["value"]
        if (
            is_common_root
            and self._array_operand_state(operation, value)
            and not (self._thread_data_operand_state(operation, "value", value))
        ):
            raise TypeError(
                f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                "ThreadData value payload in the portable API; use "
                "cuda.coop.numba_mlir for backend-qualified local arrays"
            )

        has_valid = not self._is_none(bound.arguments["valid_items"])
        has_algorithm = not self._is_none(bound.arguments["algorithm"])
        binary_op = self._constant(bound.arguments["binary_op"])
        custom_binary_op = None
        from .._lowering._reduce import (
            _normalize_reduce_operation,
            block_reduce_builtin,
            group_reduce,
            reduce,
            warp_reduce,
            warp_reduce_builtin,
        )

        try:
            normalized_op = _normalize_reduce_operation(binary_op)
        except NotImplementedError:
            if is_common_root or not callable(binary_op):
                raise
            normalized_op = None
            custom_binary_op = binary_op

        if has_valid or has_algorithm or custom_binary_op is not None:
            if group.kind not in {"block", "warp", "threads_within_warp"}:
                raise NotImplementedError(
                    "valid_items, custom callbacks, and explicit CUB algorithms "
                    "are supported only for block, physical-warp, and "
                    "logical-warp groups"
                )
            if broadcast:
                raise NotImplementedError(
                    "direct CUB reduce returns a defined value only at the group "
                    "root; it cannot satisfy broadcast=True"
                )
            if has_valid:
                is_static, valid_items = self._try_constant(
                    bound.arguments["valid_items"]
                )
                if is_static:
                    if isinstance(valid_items, bool) or not isinstance(
                        valid_items, Integral
                    ):
                        raise TypeError(
                            "cuda.coop.numba_mlir.reduce valid_items must be an "
                            "integer, not bool"
                        )
                    group_size = group.static_size
                    assert group_size is not None
                    if not 1 <= int(valid_items) <= group_size:
                        raise ValueError(
                            "cuda.coop.numba_mlir.reduce static valid_items must "
                            f"be between 1 and group size {group_size}"
                        )
            if (
                group.kind == "block"
                and self._array_operand_state(operation, value)
                and not has_algorithm
            ):
                raise ValueError(
                    "cuda.coop.numba_mlir.reduce ThreadData BlockReduce "
                    "requires an explicit algorithm"
                )
            if normalized_op == "sum":
                factory, factory_kwargs = self._scope_factory(group, "sum")
            elif custom_binary_op is not None and group.kind == "block":
                assert group.hierarchy is not None
                factory = reduce
                factory_kwargs = {
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": custom_binary_op,
                }
            elif custom_binary_op is not None:
                assert group.hierarchy is not None
                threads_in_warp = group.static_size
                assert threads_in_warp is not None
                factory = warp_reduce
                factory_kwargs = {
                    "threads_in_warp": threads_in_warp,
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": custom_binary_op,
                }
            elif group.kind == "block":
                assert group.hierarchy is not None
                factory = block_reduce_builtin
                factory_kwargs = {
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            else:
                assert group.hierarchy is not None
                threads_in_warp = group.static_size
                assert threads_in_warp is not None
                factory = warp_reduce_builtin
                factory_kwargs = {
                    "threads_in_warp": threads_in_warp,
                    "threads_per_block": group.hierarchy.block_dim,
                    "binary_op": normalized_op,
                }
            if has_valid:
                parameter = "num_valid" if group.kind == "block" else "valid_items"
                factory_kwargs[parameter] = bound.arguments["valid_items"]
            if has_algorithm:
                if group.kind != "block":
                    raise NotImplementedError(
                        "CUB algorithm selection applies to BlockReduce, not WarpReduce"
                    )
                factory_kwargs["algorithm"] = bound.arguments["algorithm"]
            return self._rewritten_call(
                inst,
                factory=factory,
                args=[value],
                kwargs=factory_kwargs,
                common_root_operation=(operation if is_common_root else None),
            )

        if group.kind == "grid":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.reduce grid groups require a hidden "
                "per-launch provider workspace"
            )
        return self._rewritten_call(
            inst,
            factory=group_reduce,
            args=[value],
            kwargs={
                "group": group,
                "binary_op": bound.arguments["binary_op"],
                "broadcast": broadcast,
                "_compile_context": self._provider_compile_context(),
            },
            common_root_operation=(operation if is_common_root else None),
        )


__all__ = ["_ReducePlanning"]
