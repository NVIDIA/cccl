# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deduplicate cooperative specializations and compile them in a shared LTO
IR bundle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from .._types import (
    _hash_symbol_value,
    algo_coalesce_key,
    collect_specializations,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)
from ._operations import FactoryOperation
from ._rewrite_support import CoopSinglePhaseRewriteError, _RewriteMatch

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class _InvocableRewrite:
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
        irrelevant; each value contributes its Python type and structural symbol
        hash. The object identity makes this unsuitable as a persistent cache
        key across processes.

        Parameters
        ----------
        factory : callable
            Registered host-side provider factory whose identity partitions
            the cache. It is not invoked here.
        factory_metadata : FactoryOperation
            Required registration carrying the operation name, namespace, and
            ABI and scope contracts for ``factory``.
        factory_kwargs : dict of str to object
            Resolved specialization inputs, after removing lowering-plan
            metadata.

        Returns
        -------
        tuple
            Provider identity string and sorted keyword-name, value-type,
            and value-digest triples used by the rewrite and compiler-state
            caches.
        """

        def cache_component(name, value):
            hasher = hashlib.sha1()
            _hash_symbol_value(hasher, value)
            value_type = f"{type(value).__module__}.{type(value).__qualname__}"
            return (name, value_type, hasher.hexdigest())

        # This cache is compiler-state-local. Object identity deliberately keeps
        # separately registered providers apart even when their public operation
        # name and specialization arguments are identical.
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
    def _validate_invocable(invocable, factory_metadata):
        op_name = factory_metadata.operation
        if not callable(invocable) or not hasattr(invocable, "files"):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase factory for '{op_name}' did not produce a "
                f"coop invocable; got {type(invocable)!r}."
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
                f"coop provider '{op_name}' returned "
                f"incompatible metadata: {details}."
            )

    def _prepare_ltoir_bundle_for_matches(
        self, matches: list[_RewriteMatch]
    ) -> None:
        """Prepare one LTO IR bundle for distinct specializations when possible.

        Collect factory specializations without immediately building each
        invocable, deduplicate identical matches, and associate each collected
        algorithm with its thread dimensions. Bundling is attempted only for at
        least two unique matches and before any specialization has been recorded
        as materialized in this compiler state.

        This is an optional compilation optimization. Clear the previous bundle
        lookup first; a collection-count mismatch or an import, OS, or runtime
        failure leaves it empty so ``_materialize_invocable`` can call factories
        individually. Other exceptions propagate. Successful preparation records
        specializations by invocable cache key without replacing function IR.
        If those specializations coalesce to one algorithm, bundle preparation
        may produce no shared bundle; retain the collected specializations for
        individual materialization anyway.

        Parameters
        ----------
        matches : list of _RewriteMatch
            Validated provider calls from the entire function, in scan
            order.

        Returns
        -------
        None
            ``_prebundled_specializations`` holds the prepared specializations,
            which may share a bundle. Early exits and caught failures leave
            it empty so materialization can invoke factories directly.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        self._prebundled_specializations = {}
        if not matches:
            return
        if rewrite._state.metadata.get(
            "__cuda_coop_numba_mlir_materialized_specializations__"
        ):
            return
        unique_matches: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], _RewriteMatch
        ] = {}
        for match in matches:
            key = self._invocable_cache_key(
                match.factory,
                match.factory_metadata,
                match.factory_kwargs,
            )
            if key not in unique_matches:
                unique_matches[key] = match
        if len(unique_matches) < 2:
            return
        try:
            with collect_specializations() as collected:
                for match in unique_matches.values():
                    _ = match.factory(**match.factory_kwargs)
            if len(collected) != len(unique_matches):
                return
            algorithms = []
            threads_by_algo = {}
            block_threads_by_algo = {}
            prebundled = {}
            for key, (algo, threads, block_threads) in zip(
                unique_matches.keys(), collected
            ):
                algorithms.append(algo)
                prebundled[key] = (algo, threads, block_threads)
                if threads is not None:
                    threads_by_algo[id(algo)] = int(threads)
                if block_threads is not None:
                    block_threads_by_algo[id(algo)] = block_threads
            prepare_ltoir_bundle(
                algorithms,
                bundle_name=f"cuda_coop_numba_mlir_bundle_{id(self)}_{id(rewrite._func_ir)}",
                allow_single=False,
                threads_by_algo=threads_by_algo,
                block_threads_by_algo=block_threads_by_algo,
            )
            self._prebundled_specializations = prebundled
        except (ImportError, OSError, RuntimeError):
            self._prebundled_specializations = {}

    def _materialize_invocable(self, match: _RewriteMatch):
        """Obtain the callable specialization for one validated provider match.

        Consult the rewrite-local cache, then the cache in compiler metadata so
        a fresh rewrite during launch retries can reuse prior work. On a miss,
        construct an invocable from a prepared specialization or evaluate the
        factory directly. Newly constructed and compiler-cache invocables must
        agree with the provider's storage and synchronization contracts before
        being used; successful construction populates both caches.

        Parameters
        ----------
        match : _RewriteMatch
            Provider factory, registered contract, and resolved
            specialization keywords. Lowering-plan metadata has already been
            removed.

        Returns
        -------
        invocable : object
            Callable provider object exposing link files and its ABI
            metadata.
        created : bool
            True when this call constructed the invocable, including from a
            prepared specialization; False for either cache hit.

        Raises
        ------
        CoopSinglePhaseRewriteError
            Construction fails or the result violates the registered
            contract.
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
                specialization, threads, block_threads = prebundled
                invocable = make_invocable_from_specialization(
                    specialization, threads=threads, block_threads=block_threads
                )
            else:
                invocable = match.factory(**match.factory_kwargs)
        except Exception as e:
            raise CoopSinglePhaseRewriteError(
                f"Failed to evaluate coop single-phase factory at compile time "
                f"for '{match.op_name}'."
            ) from e
        self._validate_invocable(invocable, match.factory_metadata)
        rewrite._invocable_cache[key] = invocable
        compile_cache[key] = invocable
        return (invocable, True)

    def _record_invocable_specialization(self, invocable):
        rewrite = cast("CoopSinglePhaseRewrite", self)
        specialization = getattr(invocable, "specialization", None)
        link_key = (
            algo_coalesce_key(specialization)
            if specialization is not None
            else None
        )
        materialized_specializations = rewrite._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_materialized_specializations__", []
        )
        if (
            link_key is not None
            and link_key not in materialized_specializations
        ):
            materialized_specializations.append(link_key)


__all__ = ["_InvocableRewrite"]
