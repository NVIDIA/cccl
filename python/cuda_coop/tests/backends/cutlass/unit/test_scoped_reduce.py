# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
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


def test_block_reduce_builtin_delegates_to_group_frontend():
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cuda.coop.cutlass import _group_reduce
        from cuda.coop.cutlass._dsl.block import _reduce

        calls = []

        def capture(group, value, /, *args, **kwargs):
            calls.append((group, value, args, kwargs))
            return "reduced"

        _group_reduce._reduce = capture
        value = object()

        assert _reduce._sum_provider(
            value=value,
            launch_metadata={{"threads_per_block": 64}},
        ) == "reduced"
        group, captured_value, args, kwargs = calls[-1]
        assert group.kind == "block"
        assert group.is_current
        assert captured_value is value
        assert args == ()
        assert kwargs == {{
            "broadcast": True,
            "valid_items": None,
            "algorithm": None,
            "launch_metadata": {{"threads_per_block": 64}},
        }}

        assert _reduce._reduce_provider(
            value=value,
            binary_op="max",
            valid_items=17,
            algorithm="raking",
            launch_metadata={{"threads_per_block": 64}},
        ) == "reduced"
        group, captured_value, args, kwargs = calls[-1]
        assert group.kind == "block"
        assert captured_value is value
        assert args == ()
        assert kwargs == {{
            "binary_op": "max",
            "broadcast": False,
            "valid_items": 17,
            "algorithm": "raking",
            "launch_metadata": {{"threads_per_block": 64}},
        }}

        assert _reduce._sum_provider._supports_native_thread_data
        assert _reduce._sum_provider._preserves_launch_metadata
        assert _reduce._sum_provider._uses_planned_temp_storage
        assert _reduce._reduce_provider._supports_native_thread_data
        assert _reduce._reduce_provider._preserves_launch_metadata
        assert _reduce._reduce_provider._uses_planned_temp_storage

        try:
            _reduce._reduce_provider(
                value=value,
                num_valid=3,
                valid_items=4,
            )
        except TypeError as exc:
            assert "got both num_valid and valid_items" in str(exc)
        else:
            raise AssertionError("conflicting valid-count aliases should fail")
        """
    )

    result = _run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr


def test_registered_warp_reduce_providers_validate_physical_warp_width():
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cuda.coop.cutlass import _group_reduce
        from cuda.coop.cutlass._dsl.warp import _dispatch
        from cuda.coop.cutlass._dsl.warp import _reduce  # noqa: F401

        def unexpected_group_reduce(*args, **kwargs):
            raise AssertionError("invalid width reached group Reduce")

        _group_reduce._reduce = unexpected_group_reduce
        for primitive_name in ("sum", "reduce", "min", "max"):
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


def test_thread_group_reduce_validates_without_provider_import():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        group = coop.this_block()

        try:
            coop.reduce(group, 1, binary_op="not_an_op")
        except NotImplementedError as exc:
            assert "supports sum" in str(exc)
        else:
            raise AssertionError("unknown cudax reduce op should fail early")

        try:
            coop.reduce(group, 1, broadcast="all")
        except TypeError as exc:
            assert "broadcast must be a bool" in str(exc)
        else:
            raise AssertionError("non-boolean broadcast should fail early")

        try:
            coop.reduce("block", 1)
        except TypeError as exc:
            assert "ThreadGroup" in str(exc)
        else:
            raise AssertionError("non-ThreadGroup reduce group should fail")

        try:
            coop.reduce(1)
        except TypeError as exc:
            assert "value" in str(exc)
        else:
            raise AssertionError("group-less reduce should fail early")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cudax_reduce_request_erases_irrelevant_hierarchy_facts():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    coop_source_path = SOURCE_ROOT / "cuda" / "coop"
    script = textwrap.dedent(
        f"""
        import cuda.coop
        cuda.coop.__path__ = [
            {str(coop_source_path)!r},
            *cuda.coop.__path__,
        ]

        from cutlass.base_dsl.typing import Int32

        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._dsl import _cudax_reduce_provider as provider

        block_only = provider._CudaxReduceRequest(
            group=coop.ThreadGroup(
                kind="block",
                hierarchy=coop.ThreadHierarchy._resolved(block_dim=64),
            ),
            op="sum",
            value_type=Int32,
        )
        extra_hierarchy = provider._CudaxReduceRequest(
            group=coop.ThreadGroup(
                kind="block",
                hierarchy=coop.ThreadHierarchy._resolved(
                    block_dim=64,
                    grid_dim=2,
                    cluster_dim=1,
                ),
            ),
            op="sum",
            value_type=Int32,
        )

        assert extra_hierarchy == block_only
        assert hash(extra_hierarchy) == hash(block_only)
        assert extra_hierarchy.symbol_name == block_only.symbol_name
        assert extra_hierarchy.group.hierarchy.grid_dim is None
        assert extra_hierarchy.group.hierarchy.cluster_dim is None
        """
    )

    result = _run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr


def test_cudax_reduce_request_is_plan_owned_and_fails_before_side_effects(
    monkeypatch,
):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts, ReduceValueKind
    from cuda.coop.cutlass import _group_reduce
    from cuda.coop.cutlass._dsl import _cudax_reduce_provider as provider

    plan = _group_reduce._make_group_reduce_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        value_kind=ReduceValueKind.ARRAY,
        items_per_thread=2,
        op="sum",
        broadcast=True,
        source="root",
    ).require_supported()
    request = provider._CudaxReduceRequest(
        plan=plan,
        op="sum",
        value_type=Int32,
    )
    assert request.plan is plan
    assert request.plan.target is GroupLoweringTarget.CUDAX_GROUP
    assert request.plan.call.source == "root"
    assert request.plan.artifact_key == request.semantic_key

    with pytest.raises(ValueError, match="operator does not match"):
        provider._CudaxReduceRequest(
            plan=plan,
            op="max",
            value_type=Int32,
        )

    side_effects = []
    monkeypatch.setattr(
        provider,
        "_register_request",
        lambda request: side_effects.append("request"),
    )
    monkeypatch.setattr(
        provider,
        "ffi",
        lambda **kwargs: side_effects.append("ffi"),
    )
    with pytest.raises(NotImplementedError, match="exact block dimensions") as exc:
        provider.provider_reduce(
            group=coop.this_block(),
            launch=LaunchFacts(max_block_dim=64),
            value=Int32(1),
        )
    assert "upper bound" in str(exc.value)
    assert side_effects == []


def test_one_item_thread_data_reduce_artifacts_are_distinct_from_scalar():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts, ReduceValueKind
    from cuda.coop._core.block import BlockReduceAlgorithm
    from cuda.coop.cutlass import _group_reduce
    from cuda.coop.cutlass._dsl import _cudax_reduce_provider as provider

    def plan(value_kind, *, algorithm=None):
        return _group_reduce._make_group_reduce_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            dtype=Int32,
            value_kind=value_kind,
            items_per_thread=1,
            op="sum",
            broadcast=algorithm is None,
            algorithm=algorithm,
        ).require_supported()

    scalar_cudax = provider._CudaxReduceRequest(
        plan=plan(ReduceValueKind.SCALAR),
        op="sum",
        value_type=Int32,
    )
    array_cudax = provider._CudaxReduceRequest(
        plan=plan(ReduceValueKind.ARRAY),
        op="sum",
        value_type=Int32,
    )
    assert scalar_cudax != array_cudax
    assert scalar_cudax.symbol_name != array_cudax.symbol_name
    assert not scalar_cudax.symbol_name.endswith("_x1")
    assert array_cudax.symbol_name.endswith("_x1")

    scalar_cub = provider._CubReduceRequest(
        plan=plan(
            ReduceValueKind.SCALAR,
            algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
        op="sum",
        value_type=Int32,
    )
    array_cub = provider._CubReduceRequest(
        plan=plan(
            ReduceValueKind.ARRAY,
            algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
        op="sum",
        value_type=Int32,
    )
    assert scalar_cub != array_cub
    assert scalar_cub.symbol_name != array_cub.symbol_name
    assert "_x1_" not in scalar_cub.symbol_name
    assert "_i32_x1_warp_reductions_full" in array_cub.symbol_name


def test_omitted_and_explicit_default_cub_reduce_share_one_artifact():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import ArgumentBinding, LaunchFacts, ReduceValueKind
    from cuda.coop._core.block import BlockReduceAlgorithm
    from cuda.coop.cutlass import _group_reduce
    from cuda.coop.cutlass._dsl import _cudax_reduce_provider as provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    def request(algorithm):
        plan = _group_reduce._make_group_reduce_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            dtype=Int32,
            value_kind=ReduceValueKind.SCALAR,
            items_per_thread=1,
            op="sum",
            broadcast=False,
            valid_items=ArgumentBinding.static(17),
            algorithm=algorithm,
        ).require_supported()
        return provider._CubReduceRequest(
            plan=plan,
            op="sum",
            value_type=Int32,
        )

    omitted = request(None)
    explicit = request(BlockReduceAlgorithm.WARP_REDUCTIONS)
    assert omitted == explicit
    assert hash(omitted) == hash(explicit)
    assert omitted.symbol_name == explicit.symbol_name

    session = provider_support.BundleSession()
    session.add(omitted)
    session.add(explicit)
    assert session.request_list() == [omitted]


def test_thread_group_reduce_resolves_current_launch_dimensions():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop
        from cuda.coop._core import LaunchFactConflict
        from cuda.coop.cutlass import _group_reduce
        from cuda.coop.cutlass._dsl import _launch

        class KernelOp:
            def __init__(self, attributes, parent_op=None):
                self.attributes = attributes
                self.parent_op = parent_op

        assert _launch.block_dim_from_nvvm_thread_attr(
            "attr : 8, 4, 2>"
        ) == (8, 4, 2)
        assert _launch.block_dim_from_nvvm_thread_attr("attr : 8, 4, 2, 1>") is None
        assert _launch.block_dim_from_nvvm_thread_attr("attr : 8, -1, 2>") is None
        assert _launch.block_dim_from_launch_metadata(
            {"block_dim_x": 8, "block_dim_y": 4, "block_dim_z": 2}
        ) == (8, 4, 2)
        assert (
            _launch.block_dim_from_launch_metadata(
                {"block_dim_x": 8, "block_dim_y": 4}
            )
            is None
        )

        required = KernelOp({"nvvm.reqntid": "attr : 8, 4, 2>"})
        bounded = KernelOp({"nvvm.maxntid": "attr : 256>"}, required)
        assert _launch.block_dim_from_kernel_op(
            bounded,
            allow_maxntid=False,
        ) == (8, 4, 2)
        assert _launch.block_dim_from_kernel_op(
            bounded,
            allow_maxntid=True,
        ) == (8, 4, 2)

        max_only = KernelOp({"nvvm.maxntid": "attr : 256>"})
        assert (
            _launch.block_dim_from_kernel_op(max_only, allow_maxntid=False)
            is None
        )
        assert _launch.block_dim_from_kernel_op(
            max_only,
            allow_maxntid=True,
        ) == (256, 1, 1)

        explicit_metadata = _group_reduce._resolve_group_for_reduce(
            coop.this_block(),
            {"launch_metadata": {"block": (8, 4, 1)}},
        )
        assert explicit_metadata.is_static
        assert explicit_metadata.block_dim == (8, 4, 1)
        assert explicit_metadata.source == "inferred_launch"

        original_kernel_facts = _launch.current_kernel_launch_facts
        _launch.current_kernel_launch_facts = lambda: _launch.LaunchFacts(
            exact_block_dim=64
        )
        try:
            _group_reduce._resolve_group_for_reduce(
                coop.this_block(),
                {"launch_metadata": {"threads_per_block": 32}},
            )
        except LaunchFactConflict as exc:
            assert "conflicting exact_block_dim" in str(exc)
        else:
            raise AssertionError("call metadata and reqntid conflicts should fail")
        finally:
            _launch.current_kernel_launch_facts = original_kernel_facts

        original = _launch.current_kernel_launch_facts
        _launch.current_kernel_launch_facts = lambda: _launch.LaunchFacts(
            exact_block_dim=(16, 2, 1)
        )
        try:
            current_launch = _group_reduce._resolve_group_for_reduce(
                coop.this_block(),
                {},
            )
        finally:
            _launch.current_kernel_launch_facts = original

        assert current_launch.is_static
        assert current_launch.block_dim == (16, 2, 1)
        assert current_launch.symbol_suffix == "block_b16x2"

        saw_allow_maxntid = []
        _launch.current_kernel_launch_facts = (
            lambda: saw_allow_maxntid.append(False)
            or _launch.LaunchFacts(max_block_dim=(256, 1, 1))
        )
        try:
            _group_reduce._resolve_group_for_reduce(coop.this_block(), {})
        except NotImplementedError as exc:
            assert "could not infer static block dimensions" in str(exc)
        else:
            raise AssertionError("maxntid alone must not specialize a reduction")
        finally:
            _launch.current_kernel_launch_facts = original
        assert saw_allow_maxntid == [False]

        try:
            _group_reduce._resolve_group_for_reduce(
                coop.this_block(),
                {"launch_metadata": {"block_dim_x": 16, "block_dim_y": 2}},
            )
        except NotImplementedError as exc:
            assert "could not infer static block dimensions" in str(exc)
        else:
            raise AssertionError("partial launch metadata should be rejected")

        try:
            _group_reduce._resolve_group_for_reduce(
                coop.this_block(),
                {
                    "launch_metadata": {"threads_per_block": 32},
                    "launch": {"threads_per_block": 32},
                },
            )
        except TypeError as exc:
            assert "multiple launch metadata aliases" in str(exc)
        else:
            raise AssertionError("duplicate launch metadata aliases should fail")

        try:
            _group_reduce._resolve_group_for_reduce(
                coop.this_warp(),
                {"launch_metadata": {"threads_per_block": 48}},
            )
        except NotImplementedError as exc:
            assert "every physical warp" in str(exc)
            assert "48 block threads" in str(exc)
        else:
            raise AssertionError("partial physical-warp groups must fail closed")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_required_for_multi_warp_group():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        try:
            coop._block.sum(1, launch_metadata={"threads_per_block": 64})
        except ValueError as exc:
            assert "requires TempStorage" in str(exc)
            assert "64 threads" in str(exc)
        else:
            raise AssertionError("multi-warp group without TempStorage should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_shared_reduce_provider_session_rolls_back_on_cold_failure(monkeypatch):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")

    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl import _cudax_reduce_provider
    from cuda.coop.cutlass._dsl import _provider as provider_support
    from cuda.coop.cutlass._dsl.block import _dispatch, _reduce

    class CompileOptions:
        pass

    class Dsl:
        compile_options = CompileOptions()

    dsl = Dsl()
    monkeypatch.setattr(provider_support, "_get_cute_dsl", lambda: dsl)
    monkeypatch.setattr(
        provider_support,
        "_ensure_trace_hook_registered",
        lambda: None,
    )

    def failing_ffi(**kwargs):
        def call(*args):
            raise TypeError("provider failed after request registration")

        return call

    monkeypatch.setattr(_cudax_reduce_provider, "ffi", failing_ffi)
    monkeypatch.setitem(_dispatch._IMPLS, "sum", _reduce._sum_provider)
    monkeypatch.delitem(
        sys.modules,
        "cuda.coop.cutlass._dsl.block._provider",
        raising=False,
    )

    with pytest.raises(TypeError, match="provider failed after request registration"):
        coop._block.sum(
            Int32(1),
            launch_metadata={"threads_per_block": 32},
        )

    assert provider_support.lookup_bundle_session(dsl.compile_options) is None
