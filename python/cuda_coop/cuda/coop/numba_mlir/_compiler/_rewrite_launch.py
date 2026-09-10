# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launch-dimension inference and device-function deferral.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _DeferredCoopRewrite,
    normalize_dim_param,
)


class _LaunchRewrite:
    def _infer_threads_per_block_from_context(
        self,
        *,
        op_name: str,
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if "threads_per_block" not in allowed_factory_kwargs:
            return
        threads_per_block = self._infer_threads_per_block_from_launch_config()
        if threads_per_block is None:
            if (
                "threads_per_block" in seen_factory_kwargs
                and self._can_defer_explicit_launch_dim_reconciliation()
            ):
                raise _DeferredCoopRewrite
            return
        if "threads_per_block" in seen_factory_kwargs:
            explicit_threads_per_block = factory_kwargs["threads_per_block"]
            try:
                explicit_dim = normalize_dim_param(explicit_threads_per_block)
                launch_dim = normalize_dim_param(threads_per_block)
            except (TypeError, ValueError):
                return
            if explicit_dim != launch_dim:
                launch_block = self._launch_block_from_context()
                raise CoopSinglePhaseRewriteError(
                    f"cuda.coop factory '{op_name}' received "
                    f"threads_per_block={explicit_threads_per_block!r}, but the "
                    f"exact kernel launch block is {launch_block!r}. Make "
                    "threads_per_block match the launch block or omit it to infer "
                    "the dimension."
                )
            return
        factory_kwargs["threads_per_block"] = threads_per_block
        seen_factory_kwargs.add("threads_per_block")

    def _can_defer_explicit_launch_dim_reconciliation(self) -> bool:
        metadata = getattr(self._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        # Configured kernel launches carry a tracker until the whole-function
        # planner requests the exact block. Device functions defer to that
        # same planner after inlining into their kernel caller.
        should_defer = self._allow_launch_dim_deferral and (
            bool(targetoptions.get("device", False))
            or metadata.get("launch_config_tracker") is not None
        )
        self._deferred_launch_dim_inference |= should_defer
        return should_defer

    def _launch_block_from_context(self):
        metadata = getattr(self._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        launch_config = targetoptions.get("__launch_config__")
        if not isinstance(launch_config, dict):
            return None
        return launch_config.get("block")

    def _launch_dim_inference_failure_detail(self) -> str:
        metadata = getattr(self._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        if "__launch_config__" not in targetoptions:
            detail = "no __launch_config__ metadata was provided"
        else:
            launch_config = targetoptions["__launch_config__"]
            if not isinstance(launch_config, dict):
                detail = f"__launch_config__ metadata is invalid: {launch_config!r}"
            elif "block" not in launch_config:
                detail = (
                    "__launch_config__ metadata contains no block shape: "
                    f"{launch_config!r}"
                )
            else:
                detail = (
                    f"launch metadata reported invalid block={launch_config['block']!r}"
                )
        if "launch_bounds" in targetoptions:
            detail += (
                f"; launch_bounds={targetoptions['launch_bounds']!r} is only an "
                "upper bound, not an exact launch shape"
            )
        return detail

    def _infer_threads_per_block_from_launch_config(self):
        block = self._launch_block_from_context()
        if isinstance(block, list):
            block = tuple(block)
        try:
            x, y, z = normalize_dim_param(block)
        except (TypeError, ValueError):
            return None
        if z != 1:
            return x, y, z
        if y != 1:
            return x, y
        return x

    def _can_defer_launch_dim_inference(self) -> bool:
        should_defer = (
            self._allow_launch_dim_deferral
            and self._infer_threads_per_block_from_launch_config() is None
        )
        self._deferred_launch_dim_inference |= should_defer
        return should_defer

    @staticmethod
    def _canonicalize_dim_factory_alias(
        *,
        op_name: str,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if "dim" not in seen_factory_kwargs:
            return
        if "threads_per_block" in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                f"cuda.coop factory '{op_name}' received both 'threads_per_block' "
                "and its 'dim' alias; provide only one."
            )
        factory_kwargs["threads_per_block"] = factory_kwargs.pop("dim")
        seen_factory_kwargs.remove("dim")
        seen_factory_kwargs.add("threads_per_block")


__all__ = ["_LaunchRewrite"]
