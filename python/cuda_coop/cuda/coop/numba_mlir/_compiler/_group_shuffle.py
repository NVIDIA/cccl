# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    Integral,
    ThreadGroup,
    inspect,
    ir,
)


class _ShufflePlanning:
    def _lower_shuffle(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("shuffle", bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle currently lowers only complete physical block groups"
            )
        if not self._is_none(bound.arguments.get("block_prefix")) or not self._is_none(
            bound.arguments.get("block_suffix")
        ):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle root projection currently supports the scalar-return ABI without boundary outputs"
            )
        from .._lowering._shuffle import BlockShuffleType

        mode = self._constant(bound.arguments["mode"])
        if hasattr(mode, "value"):
            mode = mode.value
        try:
            shuffle_type = {
                "offset": BlockShuffleType.Offset,
                "rotate": BlockShuffleType.Rotate,
                "up": BlockShuffleType.Up,
                "down": BlockShuffleType.Down,
            }[mode]
        except KeyError as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.shuffle mode must be offset, rotate, up, or down"
            ) from exc
        factory, factory_kwargs = self._scope_factory(group, "shuffle")
        factory_kwargs["block_shuffle_type"] = shuffle_type
        distance = bound.arguments["distance"]
        normalized_distance = self._constant(distance)
        is_default_up_down_distance = (
            shuffle_type in {BlockShuffleType.Up, BlockShuffleType.Down}
            and normalized_distance == 1
        )
        if not is_default_up_down_distance:
            factory_kwargs["distance"] = distance
        value = bound.arguments["value"]
        array_state = self._is_array_value(value)
        if array_state is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.shuffle could not resolve cyclic array provenance to a concrete scalar or array value"
            )
        is_array_value = array_state
        if is_common_root:
            is_thread_data = self._is_array_value(value, thread_data_only=True)
            if is_thread_data is None:
                raise GroupRewriteError(
                    "cuda.coop.shuffle could not resolve value payload provenance"
                )
            if not is_thread_data:
                raise TypeError(
                    "cuda.coop.shuffle requires a fixed-size ThreadData payload in the portable API; use cuda.coop.numba_mlir for backend-qualified scalar or local-array shuffles"
                )
            if shuffle_type not in {BlockShuffleType.Up, BlockShuffleType.Down}:
                raise ValueError(
                    "cuda.coop.shuffle mode must be 'down' or 'up' in the portable API; use cuda.coop.numba_mlir for backend-qualified scalar offset/rotate shuffles"
                )
            if (
                isinstance(normalized_distance, bool)
                or not isinstance(normalized_distance, Integral)
                or int(normalized_distance) != 1
            ):
                raise ValueError(
                    "cuda.coop.shuffle distance must be exactly 1 in the portable API; use cuda.coop.numba_mlir for backend-qualified scalar shuffles with other distances"
                )
        if is_array_value and shuffle_type not in {
            BlockShuffleType.Up,
            BlockShuffleType.Down,
        }:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.shuffle array values currently support only 'up' and 'down' modes"
            )
        if is_array_value:
            statements: list[Any] = []
            scope = inst.target.scope
            loc = inst.loc
            result_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="shuffle_result",
                prototype=value,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            call_statements = self._rewritten_call(
                inst,
                factory=factory,
                args=[value, result_payload],
                kwargs=factory_kwargs,
                return_alias=result_payload,
                common_root_operation="shuffle" if is_common_root else None,
            )
            call_statements.pop()
            statements.extend(call_statements)
            statements.append(ir.Assign(result_payload, inst.target, loc))
            return statements
        return self._rewritten_call(
            inst,
            factory=factory,
            args=[value],
            kwargs=factory_kwargs,
            common_root_operation="shuffle" if is_common_root else None,
        )


__all__ = ["_ShufflePlanning"]
