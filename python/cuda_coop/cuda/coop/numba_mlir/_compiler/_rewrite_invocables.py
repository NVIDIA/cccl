# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Batch provider compilation and reuse callable specializations.

Whole-function call analysis collects the kernel's distinct provider
specializations before materializing any invocable. On the successful batching
path, their generated C++ shares one translation unit and, on a cache miss,
one NVRTC compilation to LTO IR, avoiding a separate compiler invocation for
each cooperative primitive. Every resulting invocable retains the shared
artifact for linking.

This reduces provider-compilation overhead; it is not a promise of exactly one
NVRTC invocation per kernel. Cache hits can avoid compilation, equivalent
specializations can collapse to one provider, and unavailable or failed
bundling falls back to individual materialization. Scratch layouts come from
the provider compilation and share its cache entry. Extra support images and
the kernel's own compilation remain separate work.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from .._types import (
    _hash_symbol_value,
    collect_specializations,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)
from ._group_planner_support import GroupRewriteError
from ._operations import FactoryOperation
from ._rewrite_support import CoopSinglePhaseRewriteError, _RewriteMatch

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _InvocableRewrite:
    """Reuse compiled providers across matches and planner retries.

    The mixin validates factory results, caches them in compiler state, and
    optionally compiles several specializations together.
    """

    @staticmethod
    def _invocable_cache_key(
        factory: Callable[..., Any],
        factory_metadata: FactoryOperation,
        factory_kwargs: dict[str, object],
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        """Identify a provider specialization within one compiler state.

        Include factory object identity and the registered storage, execution,
        and synchronization contracts so distinct providers cannot share an
        invocable merely because their operation names match. Keyword order is
        irrelevant; each value contributes its Python type and structural
        symbol hash. The object identity makes this unsuitable as a persistent
        cache key across processes.

        Parameters
        ----------
        factory : callable
            Registered host-side provider factory whose identity
            partitions the cache. It is not invoked here.
        factory_metadata : FactoryOperation
            Required registration carrying the operation name, namespace,
            and ABI and scope contracts for ``factory``.
        factory_kwargs : dict of str to object
            Resolved specialization inputs, without lowering-plan metadata.

        Returns
        -------
        tuple
            Provider identity string and sorted keyword-name, value-type,
            and value-digest triples used by the rewrite and
            compiler-state caches.
        """

        def cache_component(name: str, value: object) -> tuple[str, str, str]:
            hasher = hashlib.sha1()
            _hash_symbol_value(hasher, value)
            value_type = f"{type(value).__module__}.{type(value).__qualname__}"
            return (name, value_type, hasher.hexdigest())

        # This cache belongs to one compiler state. Object identity keeps
        # separately registered providers apart even when their operation
        # names and specialization arguments are identical.
        provider_contract = (
            factory_metadata.namespace,
            factory_metadata.storage_abi.value,
            factory_metadata.execution_scope.value,
            factory_metadata.synchronization_scope.value,
        )
        return (
            (
                f"{type(factory).__module__}.{type(factory).__qualname__}:"
                f"{id(factory)}:{factory_metadata.operation}:{provider_contract!r}"
            ),
            tuple(
                sorted(
                    (
                        cache_component(name, value)
                        for name, value in factory_kwargs.items()
                    )
                )
            ),
        )

    @staticmethod
    def _validate_invocable(
        invocable: object, factory_metadata: FactoryOperation
    ) -> None:
        """Check a factory result against its registered provider contract.

        Construction and compiler-cache lookup both use this check before an
        invocable enters the rewrite-local cache. Require a callable exposing
        link files, then compare its storage ABI and thread scopes with the
        registry. This checks declarations, not generated device code.

        Parameters
        ----------
        invocable : object
            Factory result or cached provider to check.
        factory_metadata : FactoryOperation
            The factory's registered storage and synchronization contract.

        Raises
        ------
        CoopSinglePhaseRewriteError
            The result lacks the invocable interface or declares
            incompatible storage, execution, or synchronization metadata.
        """

        op_name = factory_metadata.operation
        if not callable(invocable) or not hasattr(invocable, "files"):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase factory for '{op_name}' did not produce "
                f"a coop invocable; got {type(invocable)!r}."
            )
        expected_contract = {
            "storage_abi": factory_metadata.storage_abi,
            "execution_scope": factory_metadata.execution_scope,
            "synchronization_scope": factory_metadata.synchronization_scope,
        }
        mismatches = []
        for name, expected in expected_contract.items():
            observed = getattr(invocable, name, None)
            try:
                observed = type(expected)(observed)
            except (TypeError, ValueError):
                pass
            if observed != expected:
                mismatches.append(
                    f"{name}={observed!r} (registered {expected.value!r})"
                )
        if mismatches:
            details = ", ".join(mismatches)
            raise CoopSinglePhaseRewriteError(
                f"coop provider '{op_name}' returned incompatible metadata: "
                f"{details}."
            )

    def _prepare_ltoir_bundle_for_matches(
        self, matches: list[_RewriteMatch]
    ) -> None:
        """Prepare a shared LTO IR bundle when distinct matches permit it.

        Function preparation calls this after analyzing calls and validating
        storage uses, before materializing providers. A shared translation
        unit lets NVRTC compile all distinct provider bodies together,
        avoiding per-primitive compiler startup and
        repeated header processing. The resulting invocables share the LTO
        artifact; this does not combine the kernel's own compilation with it.

        Collect factory specializations without immediately building each
        invocable and deduplicate identical matches. Each collected algorithm
        already stores its thread dimensions. Bundling is attempted only for
        at least two unique matches. Launch deferral happens during call
        analysis, so a retry reaches bundling before any provider is emitted.

        This is an optional compilation optimization. Clear the previous
        bundle lookup first; a collection-count mismatch or an import, OS, or
        runtime failure leaves it empty so ``_materialize_invocable`` can call
        factories individually. Other exceptions propagate. Successful
        preparation records specializations by invocable cache key without
        replacing function IR. If those specializations coalesce to one
        algorithm, bundle preparation may produce no shared bundle; retain the
        collected specializations for individual materialization anyway.

        Parameters
        ----------
        matches : list of _RewriteMatch
            Validated provider calls in whole-function scan order.

        Returns
        -------
        None
            ``_prebundled_specializations`` holds the prepared
            specializations, which may share a bundle. Early exits and
            caught failures leave it empty so materialization can invoke
            factories directly.
        """

        self._prebundled_specializations = {}
        if not matches:
            return
        matches_by_specialization: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], _RewriteMatch
        ] = {}
        for match in matches:
            key = self._invocable_cache_key(
                match.factory,
                match.factory_metadata,
                match.factory_kwargs,
            )
            if key not in matches_by_specialization:
                matches_by_specialization[key] = match
        # Repeated calls already reuse one invocable; bundling only helps when
        # there are distinct specializations to compile together.
        if len(matches_by_specialization) < 2:
            return
        try:
            with collect_specializations() as collected:
                for match in matches_by_specialization.values():
                    _ = match.factory(**match.factory_kwargs)
            if len(collected) != len(matches_by_specialization):
                return
            prepare_ltoir_bundle(collected)
            self._prebundled_specializations = dict(
                zip(matches_by_specialization, collected)
            )
        except (ImportError, OSError, RuntimeError):
            self._prebundled_specializations = {}

    def _materialize_invocable(
        self, match: _RewriteMatch
    ) -> tuple[object, bool]:
        """Obtain a callable specialization for one validated match.

        Consult the rewrite-local cache, then the cache in compiler metadata
        so a fresh rewrite during launch retries can reuse prior work. On a
        miss, construct an invocable from a prepared specialization or
        evaluate the factory directly. Newly constructed and compiler-cache
        invocables must agree with the provider's storage and synchronization
        contracts before use. Successful construction populates both caches.

        Parameters
        ----------
        match : _RewriteMatch
            Provider factory, registered contract, and resolved
            specialization keywords. Lowering-plan metadata has already
            been removed.

        Returns
        -------
        invocable : object
            Callable provider with link files and ABI metadata.
        created : bool
            True when this call constructed the invocable, including from
            a prepared specialization; False for either cache hit.

        Raises
        ------
        GroupRewriteError
            A callback reaches unsupported cooperative group planning while
            its provider is materialized. The original diagnostic propagates.
        CoopSinglePhaseRewriteError
            Other construction failures occur or the result violates the
            registered contract.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        key = self._invocable_cache_key(
            match.factory,
            match.factory_metadata,
            match.factory_kwargs,
        )
        if key in rewrite._invocable_cache:
            return (rewrite._invocable_cache[key], False)
        compile_cache = rewrite._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_invocable_cache__", {}
        )
        if key in compile_cache:
            invocable = compile_cache[key]
            self._validate_invocable(invocable, match.factory_metadata)
            rewrite._invocable_cache[key] = invocable
            return (invocable, False)
        try:
            prebundled = self._prebundled_specializations.get(key)
            if prebundled is not None:
                invocable = make_invocable_from_specialization(prebundled)
            else:
                invocable = match.factory(**match.factory_kwargs)
        except GroupRewriteError:
            # A callback can reach cooperative planning while its provider is
            # materialized. Preserve the helper name and actionable diagnostic.
            raise
        except Exception as e:
            raise CoopSinglePhaseRewriteError(
                f"Failed to evaluate coop single-phase factory at compile "
                f"time for '{match.op_name}'."
            ) from e
        self._validate_invocable(invocable, match.factory_metadata)
        rewrite._invocable_cache[key] = invocable
        compile_cache[key] = invocable
        return (invocable, True)


__all__ = ["_InvocableRewrite"]
