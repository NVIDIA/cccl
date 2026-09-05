# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import (
    COMMON_PROFILE_DTYPE_FAMILIES,
    COMMON_PROFILE_MATRIX,
    GROUP_CONSTRUCTOR_SIGNATURES,
    PORTABLE_GROUP_FIRST_EXPORTS,
    PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
)
from ._imports import SOURCE_ROOT, run_python_with_source


def test_root_api_exposes_common_group_first_contract() -> None:
    expected = {
        "TempStorage",
        "TempStorageLike",
        "ThreadData",
        "ThreadDataLike",
        *PORTABLE_GROUP_FIRST_EXPORTS,
    }

    assert set(root_api.__all__) == expected
    assert not hasattr(root_api, "_DEFAULT_BACKEND_MODULE")
    assert not hasattr(root_api, "available_backends")
    assert not hasattr(root_api, "load_backend")
    for name in expected - {
        "Hierarchy",
        "TempStorageLike",
        "ThreadDataLike",
        "ThreadGroup",
        "ThreadHierarchy",
    }:
        member = getattr(root_api, name)
        assert member.__cuda_coop_backend_member__ == name


def test_common_profile_matrix_covers_every_promoted_operation() -> None:
    from cuda.coop._core.dtype_policy import (
        COMMON_V1_INTEGER_KEY_DTYPE_NAMES,
        COMMON_V1_INTEGER_VALUE_DTYPE_NAMES,
        COMMON_V1_NUMERIC_DTYPE_NAMES,
    )

    production_dtype_families = {
        "numeric": COMMON_V1_NUMERIC_DTYPE_NAMES,
        "integer_value": COMMON_V1_INTEGER_VALUE_DTYPE_NAMES,
        "integer_key": COMMON_V1_INTEGER_KEY_DTYPE_NAMES,
    }
    assert tuple(production_dtype_families) == tuple(COMMON_PROFILE_DTYPE_FAMILIES)
    for family, dtype_names in production_dtype_families.items():
        assert tuple(dtype_names) == tuple(
            COMMON_PROFILE_DTYPE_FAMILIES[family]["canonical_dtypes"]
        )
    assert tuple(COMMON_PROFILE_MATRIX) == tuple(PORTABLE_GROUP_PRIMITIVE_SIGNATURES)
    assert tuple(root_api._COMMON_V1_OPERATION_GROUPS) == tuple(
        PORTABLE_GROUP_PRIMITIVE_SIGNATURES
    )
    for name, profile in COMMON_PROFILE_MATRIX.items():
        assert profile["signature"] == PORTABLE_GROUP_PRIMITIVE_SIGNATURES[name]
        assert (
            tuple(
                root_api._common_v1_group_name(kind)
                for kind in root_api._COMMON_V1_OPERATION_GROUPS[name]
            )
            == profile["supported_groups"]
        )
        assert profile["supported_groups"]
        assert profile["result_layout"]
        assert profile["mutation_rule"]
        assert isinstance(profile["preconditions"], tuple)
        assert profile["dtype_contract"]
        assert set(profile["dtype_contract"].values()) <= set(
            COMMON_PROFILE_DTYPE_FAMILIES
        )
        assert profile["certified_backends"] == ("cutlass", "numba_mlir")
        assert profile["required_evidence"] == {
            "core": ("api", "semantics"),
            "cutlass": ("lowering", "compile", "runtime", "link"),
            "numba_mlir": ("lowering", "compile", "runtime", "link"),
        }
        assert profile["evidence_enforcement"] == {
            "core": "required",
            "cutlass": "required",
            "numba_mlir": "required",
        }
        assert set(profile["evidence_collection_paths"]) == {
            "core",
            "cutlass",
            "numba_mlir",
        }
        assert {
            role: tuple(lanes)
            for role, lanes in profile["evidence_collection_paths"].items()
        } == {
            "core": ("api", "semantics"),
            "cutlass": ("lowering", "compile", "runtime", "link"),
            "numba_mlir": ("lowering", "compile", "runtime", "link"),
        }

    assert COMMON_PROFILE_MATRIX["load"]["mutation_rule"] == (
        "populates and returns output"
    )
    assert COMMON_PROFILE_MATRIX["store"]["result_layout"] == "None"
    assert COMMON_PROFILE_MATRIX["discontinuity"]["result_layout"] == (
        "shape-preserving int32 flags"
    )
    assert COMMON_PROFILE_MATRIX["histogram"]["result_layout"] == (
        "striped counters by rank plus i times group size; out-of-range slots are zero"
    )
    assert COMMON_PROFILE_MATRIX["run_length_decode"]["result_layout"] == (
        "decoded_items_per_thread values per member in blocked window order; "
        "out-of-range positions are zero"
    )
    assert COMMON_PROFILE_MATRIX["topk_max_keys"]["result_layout"] == (
        "shape- and dtype-preserving payload; exactly the first k blocked "
        "positions are defined and unordered; remaining positions are unspecified"
    )
    assert (
        COMMON_PROFILE_MATRIX["topk_min_keys"]["result_layout"]
        == (COMMON_PROFILE_MATRIX["topk_max_keys"]["result_layout"])
    )


@pytest.mark.evidence_for("group.load", backend="core", evidence="api")
@pytest.mark.evidence_for("group.store", backend="core", evidence="api")
@pytest.mark.evidence_for("group.reduce", backend="core", evidence="api")
@pytest.mark.evidence_for("group.sum", backend="core", evidence="api")
@pytest.mark.evidence_for("group.scan", backend="core", evidence="api")
@pytest.mark.evidence_for("group.exclusive_sum", backend="core", evidence="api")
@pytest.mark.evidence_for("group.inclusive_sum", backend="core", evidence="api")
@pytest.mark.evidence_for("group.exclusive_scan", backend="core", evidence="api")
@pytest.mark.evidence_for("group.inclusive_scan", backend="core", evidence="api")
@pytest.mark.evidence_for("group.exchange", backend="core", evidence="api")
@pytest.mark.evidence_for("group.adjacent_difference", backend="core", evidence="api")
@pytest.mark.evidence_for("group.discontinuity", backend="core", evidence="api")
@pytest.mark.evidence_for("group.shuffle", backend="core", evidence="api")
@pytest.mark.evidence_for("group.histogram", backend="core", evidence="api")
@pytest.mark.evidence_for("group.run_length_decode", backend="core", evidence="api")
@pytest.mark.evidence_for("group.merge_sort_keys", backend="core", evidence="api")
@pytest.mark.evidence_for("group.merge_sort_pairs", backend="core", evidence="api")
@pytest.mark.evidence_for("group.radix_rank", backend="core", evidence="api")
@pytest.mark.evidence_for("group.radix_sort_keys", backend="core", evidence="api")
@pytest.mark.evidence_for("group.radix_sort_pairs", backend="core", evidence="api")
@pytest.mark.evidence_for("group.topk_max_keys", backend="core", evidence="api")
@pytest.mark.evidence_for("group.topk_max_pairs", backend="core", evidence="api")
@pytest.mark.evidence_for("group.topk_min_keys", backend="core", evidence="api")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="core", evidence="api")
def test_root_api_preserves_common_call_signatures() -> None:
    expected = {
        "TempStorage": (
            "(size_in_bytes: 'Any' = None, alignment: 'Any' = None, "
            "auto_sync: 'Any' = None, sharing: 'str' = 'shared') -> "
            "'TempStorageLike'"
        ),
        "ThreadData": (
            "(items_per_thread: 'int', dtype: 'Any' = None) -> 'ThreadDataLike[Any]'"
        ),
        **GROUP_CONSTRUCTOR_SIGNATURES,
        **PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
    }

    assert {
        name: str(inspect.signature(getattr(root_api, name))) for name in expected
    } == expected


def test_root_group_primitives_have_backend_neutral_docstrings() -> None:
    for name in PORTABLE_GROUP_PRIMITIVE_SIGNATURES:
        docstring = inspect.getdoc(getattr(root_api, name))

        assert docstring is not None
        assert "compiler-selected backend" in docstring
        assert "cuda.coop.<backend>" in docstring
        assert "CUTLASS" not in docstring


def test_from_cuda_import_coop_can_disable_auto_registration() -> None:
    script = textwrap.dedent(
        """
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        from cuda import coop
        from cuda.coop._core import root_api

        forbidden = (
            "numba_cuda_mlir",
            "cutlass",
            "cuda.coop.numba_mlir",
            "cuda.coop.cutlass",
        )
        assert not any(
            name == root or name.startswith(root + ".")
            for root in forbidden
            for name in sys.modules
        ), sorted(sys.modules)
        assert coop.reduce is root_api.reduce
        assert coop.this_block is root_api.this_block
        assert set(coop.__all__) == {"__version__", *root_api.__all__}
        assert coop.reduce.__cuda_coop_backend_member__ == "reduce"
        assert coop.this_block.__cuda_coop_backend_member__ == "this_block"
        portable_operations = {
            "adjacent_difference",
            "discontinuity",
            "exchange",
            "exclusive_scan",
            "exclusive_sum",
            "histogram",
            "inclusive_scan",
            "inclusive_sum",
            "load",
            "merge_sort_keys",
            "merge_sort_pairs",
            "radix_rank",
            "radix_sort_keys",
            "radix_sort_pairs",
            "reduce",
            "run_length_decode",
            "scan",
            "shuffle",
            "store",
            "sum",
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
        }
        assert portable_operations <= set(coop.__all__)
        """
    )

    result = run_python_with_source(
        script,
        roots=(SOURCE_ROOT,),
        inherit_pythonpath=False,
    )
    assert result.returncode == 0, result.stderr


def test_common_root_remains_available_without_cutlass_runtime() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class BlockCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}",
                        name=fullname,
                    )
                return None


        sys.meta_path.insert(0, BlockCutlass())

        from cuda import coop
        from cuda.coop._core import root_api

        assert not any(
            name == "cutlass" or name.startswith("cutlass.")
            for name in sys.modules
        )
        group = coop.this_block()
        assert type(group) is root_api.ThreadGroup
        assert group.kind == "block"
        assert group.static_size is None
        """
    )

    result = run_python_with_source(
        script,
        roots=(SOURCE_ROOT,),
        inherit_pythonpath=False,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_coop_distribution_owns_the_root_initializer() -> None:
    root_initializer = SOURCE_ROOT / "cuda/coop/__init__.py"

    assert root_initializer.is_file()
    assert root_initializer.read_text(encoding="utf-8").startswith(
        "# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED."
    )


def test_root_version_comes_only_from_cuda_coop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuda.coop as coop

    queried: list[str] = []

    def version(distribution_name: str) -> str:
        queried.append(distribution_name)
        return "1.2.3"

    monkeypatch.setattr(coop.importlib.metadata, "version", version)

    assert coop._package_version() == "1.2.3"
    assert queried == ["cuda-coop"]


def test_compiler_scope_routes_to_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, ...]] = []
    modules: list[str] = []

    backend = SimpleNamespace(
        ThreadData=lambda *args, **kwargs: (
            observed.append(("ThreadData", args, kwargs)) or "payload"
        ),
        this_block=lambda *args, **kwargs: (
            observed.append(("this_block", args, kwargs)) or "block"
        ),
        reduce=lambda *args, **kwargs: observed.append(("reduce", args, kwargs)) or 42,
    )

    def import_backend(name: str):
        modules.append(name)
        return backend

    monkeypatch.setattr(root_api, "import_module", import_backend)

    backend_module = "cuda.coop.testing"
    with root_api._compiler_scope(backend_module):
        block = root_api.this_block()
        payload = root_api.ThreadData(2, dtype=int)
        result = root_api.reduce(block, payload, broadcast=False)

    assert result == 42
    assert modules == [backend_module] * 3
    assert observed == [
        ("this_block", (), {}),
        ("ThreadData", (2,), {"dtype": int}),
        (
            "reduce",
            ("block", "payload"),
            {
                "algorithm": None,
                "binary_op": None,
                "broadcast": False,
                "valid_items": None,
            },
        ),
    ]


def test_qualified_backend_routes_calls_without_compiler_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, tuple[object, ...]]] = []
    backend = SimpleNamespace(
        this_block=lambda *args, **kwargs: (
            observed.append(("this_block", args)) or "block"
        ),
        sum=lambda *args, **kwargs: observed.append(("sum", args)) or 42,
    )

    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "import_module", lambda name: backend)

    root_api._register_qualified_backend("cuda.coop.testing")
    block = root_api.this_block()
    result = root_api.sum(block, 7)

    assert result == 42
    assert observed == [
        ("this_block", ()),
        ("sum", ("block", 7)),
    ]


def test_compiler_scope_overrides_and_restores_qualified_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda name: SimpleNamespace(TempStorage=lambda **kwargs: name),
    )
    root_api._register_qualified_backend("cuda.coop.qualified")

    assert root_api.TempStorage() == "cuda.coop.qualified"
    with root_api._compiler_scope("cuda.coop.compiler_owned"):
        assert root_api.TempStorage() == "cuda.coop.compiler_owned"
    assert root_api.TempStorage() == "cuda.coop.qualified"


def test_qualified_backend_registration_is_idempotent_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)

    root_api._register_qualified_backend("cuda.coop.cutlass")
    root_api._register_qualified_backend("cuda.coop.cutlass")

    with pytest.raises(RuntimeError, match="already activated"):
        root_api._register_qualified_backend("cuda.coop.other")


@pytest.mark.parametrize("backend_module", (None, 1, "", "   "))
def test_qualified_backend_rejects_invalid_module_names(backend_module) -> None:
    with pytest.raises((TypeError, ValueError)):
        root_api._register_qualified_backend(backend_module)


def test_common_root_dispatch_marks_and_restores_operation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str | None]] = []

    def thread_data(*args, **kwargs):
        del args, kwargs
        observed.append(("ThreadData", root_api._common_root_operation_name()))
        return "payload"

    def reduce(*args, **kwargs):
        del args, kwargs
        observed.append(("reduce", root_api._common_root_operation_name()))
        raise RuntimeError("stop")

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda name: SimpleNamespace(ThreadData=thread_data, reduce=reduce),
    )

    assert root_api._common_root_operation_name() is None
    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.ThreadData(1) == "payload"
        assert root_api._common_root_operation_name() is None
        with pytest.raises(RuntimeError, match="stop"):
            root_api.reduce("group", "value")
        assert root_api._common_root_operation_name() is None
    assert root_api._common_root_operation_name() is None
    assert observed == [("ThreadData", "ThreadData"), ("reduce", "reduce")]


def test_all_portable_calls_route_through_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    modules: list[str] = []
    names = (
        "this_thread",
        "this_warp",
        "this_block",
        "this_cluster",
        "this_grid",
        "load",
        "reduce",
        "store",
    )

    def make_member(name: str):
        def member(*args, **kwargs):
            del args, kwargs
            observed.append(name)
            return name

        return member

    backend = SimpleNamespace(**{name: make_member(name) for name in names})

    def import_backend(name: str):
        modules.append(name)
        return backend

    monkeypatch.setattr(root_api, "import_module", import_backend)

    backend_module = "cuda.coop.testing"
    with root_api._compiler_scope(backend_module):
        root_api.this_thread()
        root_api.this_warp()
        root_api.this_block()
        root_api.this_cluster()
        root_api.this_grid()
        root_api.load("group", "source", "output")
        root_api.reduce("group", "value")
        root_api.store("group", "destination", "value")

    assert observed == list(names)
    assert modules == [backend_module] * len(names)


def test_unscoped_group_constructors_return_backend_neutral_descriptors() -> None:
    hierarchy = root_api.ThreadHierarchy()
    groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_block(),
        root_api.this_cluster(),
        root_api.this_grid(),
    )

    assert [group.kind for group in groups] == [
        "thread",
        "warp",
        "block",
        "cluster",
        "grid",
    ]
    assert all(type(group) is root_api.ThreadGroup for group in groups)
    assert all(group.hierarchy == hierarchy for group in groups)


def test_compiler_scope_nests_and_restores_missing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules: list[str] = []

    def import_backend(name: str):
        modules.append(name)
        return SimpleNamespace(TempStorage=lambda **kwargs: (name, kwargs))

    monkeypatch.setattr(root_api, "import_module", import_backend)

    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.TempStorage()
    with root_api._compiler_scope("cuda.coop.outer"):
        assert root_api.TempStorage()[0] == "cuda.coop.outer"
        with pytest.raises(RuntimeError, match="stop"):
            with root_api._compiler_scope("cuda.coop.inner"):
                assert root_api.TempStorage()[0] == "cuda.coop.inner"
                raise RuntimeError("stop")
        assert root_api.TempStorage()[0] == "cuda.coop.outer"
    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.TempStorage()

    assert modules == [
        "cuda.coop.outer",
        "cuda.coop.inner",
        "cuda.coop.outer",
    ]


def test_backend_dependent_root_values_require_compiler_scope() -> None:
    block = root_api.this_block()

    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.reduce(block, 1)
    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.ThreadData(1)
    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.TempStorage()


@pytest.mark.parametrize(
    ("method_name", "args"),
    (
        ("rank", ()),
        ("count", ()),
        ("rank_as", (int,)),
        ("count_as", (int,)),
        ("sync", ()),
        ("sync_aligned", ()),
        ("is_member", ()),
    ),
)
def test_common_thread_group_methods_require_compiler_context(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    group = root_api.this_block()
    expected = (
        f"cuda.coop.ThreadGroup.{method_name} requires a Python DSL compiler "
        "context (compiler-owned activation) or a qualified backend import before "
        "compilation; for example, import "
        "cuda.coop.cutlass or cuda.coop.numba_mlir"
    )

    with pytest.raises(root_api.CoopCompilerContextRequiredError) as error_info:
        getattr(group, method_name)(*args)

    assert str(error_info.value) == expected


def test_portable_root_rejects_backend_specific_arguments_before_dispatch() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'payload'"):
        root_api.load("group", "source", "output", payload="prims")
    with pytest.raises(TypeError, match="unexpected keyword argument 'dtype'"):
        root_api.store("group", "destination", "value", dtype="int32")
    with pytest.raises(TypeError, match="positional"):
        root_api.reduce("group", "value", "backend-only")


@pytest.mark.parametrize(
    ("operation", "kwargs", "expected"),
    (
        (
            "exchange",
            {"mode": "blocked_to_warp_striped"},
            "cuda.coop.exchange mode must be one of: blocked_to_striped, "
            "striped_to_blocked; use a backend-qualified import for backend-only "
            "controls",
        ),
        (
            "reduce",
            {"algorithm": "warp_reductions_nondeterministic"},
            "cuda.coop.reduce algorithm must be one of: raking, "
            "raking_commutative_only, warp_reductions; use a backend-qualified "
            "import for backend-only controls",
        ),
        (
            "discontinuity",
            {"mode": "heads_and_tails"},
            "cuda.coop.discontinuity mode must be one of: heads, tails; use a "
            "backend-qualified import for backend-only controls",
        ),
    ),
)
def test_portable_root_rejects_backend_only_selectors_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    kwargs: dict[str, str],
    expected: str,
) -> None:
    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda name: pytest.fail(f"unexpected backend import: {name}"),
    )

    with root_api._compiler_scope("cuda.coop.testing"):
        with pytest.raises(ValueError) as error_info:
            getattr(root_api, operation)("group", "value", **kwargs)

    assert str(error_info.value) == expected


def test_portable_root_normalizes_hyphenated_selector_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []
    backend = SimpleNamespace(
        exchange=lambda group, value, /, **kwargs: (
            observed.append((group, value, kwargs)) or "result"
        )
    )
    monkeypatch.setattr(root_api, "import_module", lambda name: backend)

    with root_api._compiler_scope("cuda.coop.testing"):
        result = root_api.exchange(
            "group",
            "value",
            mode="blocked-to-striped",
        )

    assert result == "result"
    assert observed == [("group", "value", {"mode": "blocked_to_striped"})]


def _common_v1_groups() -> dict[str, root_api.ThreadGroup]:
    warp = root_api.this_warp()
    block = root_api.this_block()
    return {
        "thread": root_api.this_thread(),
        "warp": warp,
        "threads_within_warp": warp.group_by(8),
        "block": block,
        "warps_within_block": block.group_by(1),
        "cluster": root_api.this_cluster(),
        "grid": root_api.this_grid(),
    }


def _call_with_required_arguments(function, group: root_api.ThreadGroup) -> None:
    args = []
    kwargs = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = group if parameter.name == "group" else object()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
        else:  # pragma: no cover - the exact signature contract forbids this.
            raise AssertionError(f"unexpected required parameter kind: {parameter}")
    function(*args, **kwargs)


@pytest.mark.parametrize(
    "operation",
    tuple(root_api._COMMON_V1_OPERATION_GROUPS),
)
def test_portable_root_rejects_every_unsupported_group_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    groups = _common_v1_groups()
    supported = root_api._COMMON_V1_OPERATION_GROUPS[operation]
    supported_names = ", ".join(
        root_api._common_v1_group_name(kind) for kind in supported
    )
    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda name: pytest.fail(f"unexpected backend import: {name}"),
    )

    for group_kind, group in groups.items():
        if group_kind in supported:
            continue
        group_name = root_api._common_v1_group_name(group_kind)
        expected = (
            f"cuda.coop.{operation} does not support group kind {group_name!r} "
            f"in common V1; supported group kinds: {supported_names}; use a "
            "backend-qualified import for backend-specific group support"
        )
        with root_api._compiler_scope("cuda.coop.testing"):
            with pytest.raises(NotImplementedError) as error_info:
                _call_with_required_arguments(getattr(root_api, operation), group)

        assert str(error_info.value) == expected


def test_common_v1_reduction_includes_only_the_certified_static_mapping() -> None:
    groups = _common_v1_groups()

    for operation in ("reduce", "sum"):
        root_api._validate_common_v1_operation_group(
            operation,
            groups["threads_within_warp"],
        )
        with pytest.raises(NotImplementedError, match="warps_within_block"):
            root_api._validate_common_v1_operation_group(
                operation,
                groups["warps_within_block"],
            )


def test_unscoped_grid_descriptor_remains_backend_neutral() -> None:
    group = root_api.this_grid()

    assert type(group) is root_api.ThreadGroup
    assert group.kind == "grid"
    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.reduce(group, 1)


def test_missing_backend_operation_has_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    backend_module = "cuda.coop.testing"
    with root_api._compiler_scope(backend_module):
        with pytest.raises(root_api.UnsupportedCoopBackendOperationError) as error_info:
            root_api.reduce("group", 1)

    error = error_info.value
    assert error.backend_module == backend_module
    assert error.operation == "reduce"
    assert error.reason_code == "cuda-coop-backend-operation-unavailable"


@pytest.mark.parametrize("backend_module", (None, 1, "", "   "))
def test_compiler_scope_rejects_invalid_module_names(backend_module) -> None:
    with pytest.raises((TypeError, ValueError)):
        with root_api._compiler_scope(backend_module):
            pass
