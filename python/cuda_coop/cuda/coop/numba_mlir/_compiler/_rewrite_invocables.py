# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Invocable coalescing and LTO bundle materialization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _hash_symbol_value,
    _RewriteMatch,
    algo_coalesce_key,
    collect_specializations,
    hashlib,
    make_invocable_from_specialization,
    prepare_ltoir_bundle,
)


class _InvocableRewrite:
    @staticmethod
    def _invocable_cache_key(
        factory: object,
        op_name: str,
        factory_kwargs: dict[str, object],
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:

        def cache_component(name, value):
            hasher = hashlib.sha1()
            _hash_symbol_value(hasher, value)
            value_type = f"{type(value).__module__}.{type(value).__qualname__}"
            return (name, value_type, hasher.hexdigest())

        # This cache is compiler-state-local. Object identity deliberately keeps
        # separately registered providers apart even when their public operation
        # name and specialization arguments are identical.
        return (
            (
                f"{type(factory).__module__}.{type(factory).__qualname__}:"
                f"{id(factory)}:{op_name}"
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
    def _validate_invocable(invocable, op_name: str):
        if not callable(invocable) or not hasattr(invocable, "files"):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase factory for '{op_name}' did not produce a coop invocable; got {type(invocable)!r}."
            )

    def _prepare_ltoir_bundle_for_matches(self, matches: list[_RewriteMatch]) -> None:
        self._prebundled_specializations = {}
        if not matches:
            return
        if self._state.metadata.get(
            "__cuda_coop_numba_mlir_materialized_specializations__"
        ):
            return
        unique_matches: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], _RewriteMatch
        ] = {}
        for match in matches:
            key = self._invocable_cache_key(
                match.factory,
                match.op_name,
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
                bundle_name=f"cuda_coop_numba_mlir_bundle_{id(self)}_{id(self._func_ir)}",
                allow_single=False,
                threads_by_algo=threads_by_algo,
                block_threads_by_algo=block_threads_by_algo,
            )
            self._prebundled_specializations = prebundled
        except (ImportError, OSError, RuntimeError):
            self._prebundled_specializations = {}

    def _materialize_invocable(self, match: _RewriteMatch):
        key = self._invocable_cache_key(
            match.factory,
            match.op_name,
            match.factory_kwargs,
        )
        if key in self._invocable_cache:
            return (self._invocable_cache[key], False)
        compile_cache = self._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_invocable_cache__", {}
        )
        if key in compile_cache:
            invocable = compile_cache[key]
            self._validate_invocable(invocable, match.op_name)
            self._invocable_cache[key] = invocable
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
                f"Failed to evaluate coop single-phase factory at compile time for '{match.op_name}'."
            ) from e
        self._validate_invocable(invocable, match.op_name)
        self._invocable_cache[key] = invocable
        compile_cache[key] = invocable
        return (invocable, True)

    def _record_invocable_specialization(self, invocable):
        specialization = getattr(invocable, "specialization", None)
        link_key = (
            algo_coalesce_key(specialization) if specialization is not None else None
        )
        materialized_specializations = self._state.metadata.setdefault(
            "__cuda_coop_numba_mlir_materialized_specializations__", []
        )
        if link_key is not None and link_key not in materialized_specializations:
            materialized_specializations.append(link_key)


__all__ = ["_InvocableRewrite"]
