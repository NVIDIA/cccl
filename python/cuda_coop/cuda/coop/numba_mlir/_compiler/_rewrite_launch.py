# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reconcile provider block dimensions with the configured kernel launch.

Provider specialization needs the exact block shape; a launch bound only
gives an upper limit. These helpers read launch metadata, normalize explicit
shapes, and mark work that must wait for the planner to request a launch
configuration. Device helpers retain unresolved calls until inlining
supplies their caller's launch context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._parameters import normalize_dim_param
from ._rewrite_support import CoopSinglePhaseRewriteError, _DeferredCoopRewrite

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _LaunchRewrite:
    def _infer_threads_per_block_from_context(
        self,
        *,
        op_name: str,
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        """Fill or check the factory block shape against exact launch metadata.

        Only operations accepting ``threads_per_block`` participate. Compare
        normalized three-dimensional shapes when both explicit and launch values
        are available; equal thread counts alone do not imply equal shapes.
        Malformed explicit dimensions are left for later factory validation.
        Launch bounds are never treated as an exact shape.

        An explicit dimension may still need to wait: a device helper or a
        kernel with a launch tracker must reconcile it after inlining or after
        the whole-function planner requests launch metadata.

        Parameters
        ----------
        op_name : str
            Operation name used in mismatch diagnostics.
        allowed_factory_kwargs : set of str
            Keywords accepted by the operation's factory.
        seen_factory_kwargs : set of str
            Resolved keyword names; updated when the block shape is
            inferred.
        factory_kwargs : dict of str to object
            Resolved values; receives an inferred ``threads_per_block`` in
            place.

        Returns
        -------
        None
            Update the inferred inputs or leave them unchanged when
            unavailable.

        Raises
        ------
        CoopSinglePhaseRewriteError
            An explicit shape disagrees with the exact kernel launch shape.
        _DeferredCoopRewrite
            Internal signal that explicit dimensions need pending launch
            metadata. It propagates through argument validation to
            ``CoopSinglePhaseRewrite.match``, which preserves the IR for
            ``_CallRewriting._rewrite_calls`` to request the kernel launch
            shape and retry within ``CoopWholeFunctionPlanner``.
        """

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
                    f"threads_per_block="
                    f"{explicit_threads_per_block!r}, but the "
                    f"exact kernel launch block is {launch_block!r}. Make "
                    "threads_per_block match the "
                    "launch block or omit it to infer "
                    "the dimension."
                )
            return
        factory_kwargs["threads_per_block"] = threads_per_block
        seen_factory_kwargs.add("threads_per_block")

    def _can_defer_explicit_launch_dim_reconciliation(self) -> bool:
        """Record whether explicit dimensions need pending launch metadata.

        Device functions acquire their launch shape from the caller after
        inlining. Configured kernels retain a launch tracker until the planner
        requests an exact shape. Either case permits deferral when enabled on
        this rewrite object; without either signal, this helper returns False.

        This predicate has a side effect: a positive result latches
        ``_deferred_launch_dim_inference`` so matching preserves descriptors and
        the whole-function planner knows that it must retry.

        Returns
        -------
        bool
            Whether the caller should defer reconciliation of explicit
            dimensions.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        metadata = getattr(rewrite._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        # Configured kernel launches carry a tracker until the whole-function
        # planner requests the exact block. Device functions defer to that
        # same planner after inlining into their kernel caller.
        should_defer = rewrite._allow_launch_dim_deferral and (
            bool(targetoptions.get("device", False))
            or metadata.get("launch_config_tracker") is not None
        )
        self._deferred_launch_dim_inference |= should_defer
        return should_defer

    def _launch_block_from_context(self) -> object:
        """Read the raw launch block shape, or ``None`` if unavailable."""

        rewrite = cast("CoopSinglePhaseRewrite", self)
        metadata = getattr(rewrite._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        launch_config = targetoptions.get("__launch_config__")
        if not isinstance(launch_config, dict):
            return None
        return launch_config.get("block")

    def _launch_dim_inference_failure_detail(self) -> str:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        metadata = getattr(rewrite._state, "metadata", {}) or {}
        targetoptions = metadata.get("targetoptions", {}) or {}
        if "__launch_config__" not in targetoptions:
            detail = "no __launch_config__ metadata was provided"
        else:
            launch_config = targetoptions["__launch_config__"]
            if not isinstance(launch_config, dict):
                detail = (
                    f"__launch_config__ metadata is invalid: {launch_config!r}"
                )
            elif "block" not in launch_config:
                detail = (
                    "__launch_config__ metadata contains no block shape: "
                    f"{launch_config!r}"
                )
            else:
                detail = (
                    "launch metadata reported invalid "
                    f"block={launch_config['block']!r}"
                )
        if "launch_bounds" in targetoptions:
            detail += (
                f"; launch_bounds={targetoptions['launch_bounds']!r} "
                "is only an "
                "upper bound, not an exact launch shape"
            )
        return detail

    def _infer_threads_per_block_from_launch_config(
        self,
    ) -> int | tuple[int, int] | tuple[int, int, int] | None:
        """Normalize a launch shape and omit trailing unit dimensions.

        Invalid or missing metadata yields ``None`` so the caller can decide
        whether to request a configured launch or report a diagnostic.
        """

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
        """Flag unresolved launch dimensions for the call rewrite to retry."""

        rewrite = cast("CoopSinglePhaseRewrite", self)
        should_defer = (
            rewrite._allow_launch_dim_deferral
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
                f"cuda.coop factory '{op_name}' received both "
                f"'threads_per_block' "
                "and its 'dim' alias; provide only one."
            )
        factory_kwargs["threads_per_block"] = factory_kwargs.pop("dim")
        seen_factory_kwargs.remove("dim")
        seen_factory_kwargs.add("threads_per_block")


__all__ = ["_LaunchRewrite"]
