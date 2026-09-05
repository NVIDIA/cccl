# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

import pytest

from ....support.paths import PACKAGE_ROOT
from ..support._subprocess import (
    run_python_with_source as _run_python_with_source,
)
from ..support._subprocess import (
    run_python_with_source_and_site as _run_python_with_source_and_site,
)

SOURCE_ROOT = PACKAGE_ROOT


def test_block_scan_aggregate_preserves_explicit_launch_facts():
    pytest.importorskip("cutlass")
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        import sys
        import types

        from cuda.coop._core import LaunchFactConflict, LaunchFacts
        import cuda.coop.cutlass as cute
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._dsl.block import _scan

        calls = []
        fake_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_scan_provider"
        )
        fake_provider.provider_scan = (
            lambda **kwargs: calls.append(kwargs) or "scanned"
        )
        sys.modules[fake_provider.__name__] = fake_provider
        cute._cub_scan_provider = fake_provider
        original_kernel_facts = _launch.current_kernel_launch_facts
        _launch.current_kernel_launch_facts = lambda: LaunchFacts()
        aggregate = object()
        try:
            result = _scan._inclusive_sum_provider(
                value=object(),
                block_aggregate=aggregate,
                launch_metadata={{"threads_per_block": 64}},
            )
        finally:
            _launch.current_kernel_launch_facts = original_kernel_facts

        assert result == "scanned"
        assert calls[-1]["aggregate_output"] is aggregate
        assert calls[-1]["source"] == "scoped_block"
        assert calls[-1]["launch"].exact_block_dim == (64, 1, 1)
        assert _scan._inclusive_sum_provider._preserves_launch_metadata
        assert _scan._inclusive_sum_provider._uses_planned_temp_storage

        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=32
        )
        call_count = len(calls)
        try:
            _scan._inclusive_sum_provider(
                value=object(),
                block_aggregate=aggregate,
                launch_metadata={{"threads_per_block": 64}},
            )
        except LaunchFactConflict as exc:
            assert "conflicting exact_block_dim" in str(exc)
        else:
            raise AssertionError("conflicting scan launch facts should fail")
        finally:
            _launch.current_kernel_launch_facts = original_kernel_facts
        assert len(calls) == call_count
        """
    )

    result = _run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr


def test_scoped_block_scan_ignores_legacy_temp_storage():
    pytest.importorskip("cutlass")
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        import sys
        import types

        from cuda.coop._core import LaunchFacts
        import cuda.coop.cutlass as cute
        from cuda.coop.cutlass._dsl import _launch

        calls = []
        fake_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_scan_provider"
        )
        fake_provider.provider_scan = (
            lambda **kwargs: calls.append(kwargs) or "scanned"
        )
        sys.modules[fake_provider.__name__] = fake_provider
        cute._cub_scan_provider = fake_provider

        original_kernel_facts = _launch.current_kernel_launch_facts
        _launch.current_kernel_launch_facts = lambda: LaunchFacts()
        storage = cute._block.TempStorage(
            size_in_bytes=1,
            sharing="exclusive",
        )
        try:
            result = cute._block.exclusive_sum(
                object(),
                temp_storage=storage,
                launch_metadata={{"threads_per_block": 64}},
            )
        finally:
            _launch.current_kernel_launch_facts = original_kernel_facts

        assert result == "scanned"
        assert len(calls) == 1
        assert calls[0]["launch"].exact_block_dim == (64, 1, 1)
        assert calls[0]["source"] == "scoped_block"
        assert calls[0]["temp_storage"] is None
        assert storage.uses == ()
        assert storage.required_size_in_bytes == 0
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_scoped_scan_routes_only_supported_forms_to_group_provider():
    pytest.importorskip("cutlass")

    from cuda.coop.cutlass import ThreadData
    from cuda.coop.cutlass._dsl.block import _scan as block_scan
    from cuda.coop.cutlass._dsl.warp import _scan as warp_scan

    assert block_scan._exclusive_sum_provider._supports_native_thread_data
    assert block_scan._exclusive_sum_provider._preserves_launch_metadata
    assert block_scan._exclusive_sum_provider._uses_planned_temp_storage
    assert block_scan._exclusive_sum_provider._supports_deferred_temp_storage

    assert warp_scan._uses_group_scan(
        object(),
        threads_in_warp=32,
        valid_items=None,
    )
    assert not warp_scan._uses_group_scan(
        ThreadData(1),
        threads_in_warp=32,
        valid_items=None,
    )
    assert not warp_scan._uses_group_scan(
        object(),
        threads_in_warp=16,
        valid_items=None,
    )
    assert not warp_scan._uses_group_scan(
        object(),
        threads_in_warp=32,
        valid_items=17,
    )
    assert warp_scan._exclusive_sum_provider._preserves_launch_metadata


def test_registered_warp_scan_providers_validate_physical_warp_width():
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cuda.coop.cutlass import _group_scan
        from cuda.coop.cutlass._dsl.warp import _dispatch
        from cuda.coop.cutlass._dsl.warp import _scan  # noqa: F401

        def unexpected_group_scan(*args, **kwargs):
            raise AssertionError("invalid width reached group Scan")

        _group_scan._scan = unexpected_group_scan
        for primitive_name in (
            "exclusive_sum",
            "exclusive_scan",
            "inclusive_sum",
            "inclusive_scan",
        ):
            try:
                _dispatch.dispatch_primitive(
                    primitive_name,
                    kwargs={{
                        "value": 1,
                        "args": (),
                        "threads_in_warp": 32.0,
                    }},
                )
            except TypeError as exc:
                assert str(exc) == (
                    "cuda.coop.cutlass._warp."
                    f"{{primitive_name}} threads_in_warp must be an int"
                )
            else:
                raise AssertionError(
                    f"{{primitive_name}} accepted a non-integral warp width"
                )
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_group_scan_artifact_canonicalization_and_operand_identity():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts, ScanValueKind
    from cuda.coop._core.block import BlockScanAlgorithm
    from cuda.coop.cutlass import _group_scan
    from cuda.coop.cutlass._dsl import _cub_scan_provider as provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    def request(value_kind, algorithm):
        plan = _group_scan._make_group_scan_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            dtype=Int32,
            value_kind=value_kind,
            items_per_thread=1,
            mode="exclusive",
            op="sum",
            algorithm=algorithm,
        ).require_supported()
        return provider._CubScanRequest(
            plan=plan,
            op="sum",
            value_type=Int32,
        )

    omitted = request(ScanValueKind.SCALAR, None)
    explicit = request(ScanValueKind.SCALAR, BlockScanAlgorithm.RAKING)
    array_x1 = request(ScanValueKind.ARRAY, None)

    assert omitted == explicit
    assert omitted.semantic_key == explicit.semantic_key
    assert omitted.symbol_name == explicit.symbol_name
    assert omitted != array_x1
    assert omitted.symbol_name != array_x1.symbol_name
    assert "_scalar_" in omitted.symbol_name
    assert "_x1_" in array_x1.symbol_name

    session = provider_support.BundleSession()
    session.add(omitted)
    session.add(explicit)
    session.add(array_x1)
    assert set(session.request_list()) == {omitted, array_x1}
