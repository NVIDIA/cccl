# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _FakeInvocable:
    storage_abi = "leading_pointer"
    execution_scope = "block"
    synchronization_scope = "block"

    def __init__(self, name):
        self.name = name
        self.files = (f"{name}.ltoir",)

    def __call__(self, *args):
        del args


def test_invocable_cache_partitions_registered_factory_identities():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from cuda.coop.numba_mlir._compiler._rewrite_invocables import (
        _InvocableRewrite,
    )
    from cuda.coop.numba_mlir._compiler._rewrite_support import _RewriteMatch

    def first_factory(**kwargs):
        del kwargs
        return _FakeInvocable("first")

    def second_factory(**kwargs):
        del kwargs
        return _FakeInvocable("second")

    common_kwargs = {"threads_per_block": (64, 1, 1), "dtype": "int32"}
    provider_metadata = FactoryOperation(
        operation="load",
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    first_key = _InvocableRewrite._invocable_cache_key(
        first_factory,
        provider_metadata,
        common_kwargs,
    )
    second_key = _InvocableRewrite._invocable_cache_key(
        second_factory,
        provider_metadata,
        common_kwargs,
    )
    assert first_key != second_key
    assert first_key == _InvocableRewrite._invocable_cache_key(
        first_factory,
        provider_metadata,
        dict(common_kwargs),
    )

    rewrite = object.__new__(_InvocableRewrite)
    rewrite._invocable_cache = {}
    rewrite._prebundled_specializations = {}
    rewrite._state = SimpleNamespace(metadata={})

    def match(factory):
        return _RewriteMatch(
            op_name="load",
            factory=factory,
            factory_metadata=provider_metadata,
            func_var_name="factory",
            func_var_name_extra=None,
            runtime_args=(),
            runtime_temp_storage_var=None,
            factory_kwargs=dict(common_kwargs),
            factory_kw_value_vars=(),
            loc=None,
        )

    first, first_created = rewrite._materialize_invocable(match(first_factory))
    second, second_created = rewrite._materialize_invocable(match(second_factory))
    assert first_created
    assert second_created
    assert first.name == "first"
    assert second.name == "second"
    assert first is not second

    first_again, first_again_created = rewrite._materialize_invocable(
        match(first_factory)
    )
    assert first_again is first
    assert not first_again_created
