# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import textwrap

import pytest

from ._imports import (
    SOURCE_ROOT,
    optional_dependencies,
)
from ._imports import (
    run_python_with_source as _run_python_with_source,
)

_FAKE_CUTLASS_RUNTIME = textwrap.dedent(
    """
    import sys as _sys
    import types as _types

    _cutlass = _types.ModuleType("cutlass")
    _cutlass.__path__ = []
    _cutlass_dsl = _types.ModuleType("cutlass.cutlass_dsl")

    class _CuTeDSL:
        _instance = None

        @classmethod
        def _get_dsl(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def __init__(self):
            self.trace_context_factories = []

        def register_trace_context_factory(self, factory):
            if factory not in self.trace_context_factories:
                self.trace_context_factories.append(factory)

        def register_trace_finalize_hook(self, hook):
            self.trace_finalize_hook = hook

    _cutlass_dsl.CuTeDSL = _CuTeDSL
    _cute = _types.ModuleType("cutlass.cute")
    _cute._get_launch_facts = lambda: {}
    _base_dsl = _types.ModuleType("cutlass.base_dsl")
    _base_dsl.__path__ = []
    _compiler = _types.ModuleType("cutlass.base_dsl.compiler")
    _compiler.LinkLibraries = type(
        "LinkLibraries",
        (),
        {"_option_name": "link-libraries"},
    )
    _compiler.GPUArch = type("GPUArch", (), {})
    _cutlass.cutlass_dsl = _cutlass_dsl
    _cutlass.cute = _cute
    _cutlass.base_dsl = _base_dsl
    _base_dsl.compiler = _compiler
    _sys.modules["cutlass"] = _cutlass
    _sys.modules["cutlass.cutlass_dsl"] = _cutlass_dsl
    _sys.modules["cutlass.cute"] = _cute
    _sys.modules["cutlass.base_dsl"] = _base_dsl
    _sys.modules["cutlass.base_dsl.compiler"] = _compiler
    """
)


def _with_fake_cutlass_runtime(script: str) -> str:
    return f"{_FAKE_CUTLASS_RUNTIME}\n{textwrap.dedent(script)}"


@pytest.mark.parametrize(
    "relative_path",
    (
        "cutlass/__init__.py",
        "cutlass/_block/__init__.py",
        "cutlass/_compiler.py",
        "cutlass/_runtime_dependency.py",
        "cutlass/_warp/__init__.py",
        "cutlass/_dsl/__init__.py",
    ),
)
def test_cutlass_modules_ship_in_the_source_tree(relative_path):
    assert (SOURCE_ROOT / "cuda" / "coop" / relative_path).is_file()


def test_cutlass_public_scope_package_paths_are_retired():
    root = SOURCE_ROOT / "cuda" / "coop" / "cutlass"

    assert not (root / "block").exists()
    assert not (root / "warp").exists()


def test_cutlass_extras_select_compatible_runtime_versions():
    extras = optional_dependencies()

    assert extras["cutlass"] == [
        "cuda-coop[cu13]",
        "nvidia-cutlass-dsl>=4.8,<5",
    ]
    assert extras["cutlass-cu12"] == [
        "cuda-coop[cu12]",
        "nvidia-cutlass-dsl>=4.8,<5",
    ]
    assert extras["cutlass-cu13"] == extras["cutlass"]


def test_root_import_is_cold_and_qualified_cutlass_import_reports_missing_runtime():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCutlassRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ImportError("blocked CUTLASS runtime", name=fullname)
                return None

        sys.meta_path.insert(0, BlockCutlassRuntime())

        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert coop.reduce.__module__ == "cuda.coop._core.root_api"
        assert not any(
            name == "cutlass" or name.startswith("cutlass.")
            for name in sys.modules
        )

        try:
            import cuda.coop.cutlass  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "cutlass"
            assert exc.reason_code == "backend-runtime-missing"
            assert exc.details["missing"] == "cutlass"
            assert isinstance(exc.__cause__, ImportError)
            message = str(exc).lower()
            assert "cutlass" in message
            assert "install" in message
            assert "nvidia-cutlass-dsl>=4.8" in message
            assert "cuda-coop[cutlass]" in message
        else:
            raise AssertionError("qualified CUTLASS import unexpectedly succeeded")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("setup", "reason_code", "details"),
    [
        pytest.param(
            """
cutlass = types.ModuleType("cutlass")
cutlass.__path__ = []
sys.modules["cutlass"] = cutlass
""",
            "conflicting-backend-runtime",
            {"missing": "cutlass.cutlass_dsl"},
            id="conflicting-package",
        ),
        pytest.param(
            """
class BrokenCutlassRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "cutlass":
            raise ImportError("broken transitive import", name="cutlass_dependency")
        return None

sys.meta_path.insert(0, BrokenCutlassRuntime())
""",
            "transitive-runtime-import-failed",
            {"missing": "cutlass_dependency"},
            id="transitive-import",
        ),
    ],
)
def test_qualified_cutlass_import_errors_are_structured(
    setup,
    reason_code,
    details,
):
    indented_setup = textwrap.indent(textwrap.dedent(setup).strip(), "        ")
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys
        import types

{indented_setup}

        try:
            import cuda.coop.cutlass  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "cutlass"
            assert exc.reason_code == {reason_code!r}
            for key, value in {details!r}.items():
                assert exc.details[key] == value
            assert isinstance(exc.__cause__, BaseException)
            assert "cutlass" in str(exc).lower()
        else:
            raise AssertionError("qualified CUTLASS import unexpectedly succeeded")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_qualified_cutlass_import_reports_capabilities_with_version_diagnostics():
    script = textwrap.dedent(
        """
        import importlib.metadata

        importlib.metadata.version = lambda distribution: "99.7"
        _compiler.GPUArch = None

        try:
            import cuda.coop.cutlass  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "cutlass"
            assert exc.reason_code == "backend-runtime-incompatible"
            assert exc.details["missing_capabilities"] == (
                "cutlass.base_dsl.compiler.GPUArch",
            )
            message = str(exc)
            assert "cutlass.base_dsl.compiler.GPUArch" in message
            assert "detected nvidia-cutlass-dsl==99.7" in message
        else:
            raise AssertionError("incompatible CUTLASS import unexpectedly succeeded")
        """
    )

    result = _run_python_with_source(_with_fake_cutlass_runtime(script))

    assert result.returncode == 0, result.stderr


_EXPECTED_BLOCK_PRIMITIVES = [
    "BlockAdjacentDifferenceType",
    "BlockDiscontinuityType",
    "BlockExchangeType",
    "BlockRunLengthDecode",
    "BlockShuffleType",
    "TempStorage",
    "adjacent_difference",
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "discontinuity",
    "discontinuity_flag_heads",
    "discontinuity_flag_heads_and_tails",
    "discontinuity_flag_tails",
    "exchange",
    "exchange_blocked_to_striped",
    "exchange_blocked_to_warp_striped",
    "exchange_scatter_to_blocked",
    "exchange_scatter_to_striped",
    "exchange_scatter_to_striped_flagged",
    "exchange_scatter_to_striped_guarded",
    "exchange_striped_to_blocked",
    "exchange_warp_striped_to_blocked",
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
    "radix_sort_keys_descending",
    "radix_sort_pairs",
    "radix_sort_pairs_descending",
    "reduce",
    "row_sum",
    "run_length",
    "run_length_decode",
    "scan",
    "shuffle",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "shuffle_up",
    "store",
    "sum",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]

_EXPECTED_WARP_PRIMITIVES = [
    "TempStorage",
    "WarpExchangeType",
    "exchange",
    "exchange_blocked_to_striped",
    "exchange_scatter_to_striped",
    "exchange_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "max",
    "merge_sort_keys",
    "merge_sort_pairs",
    "min",
    "reduce",
    "scan",
    "store",
    "sum",
]

_EXPECTED_BLOCK_FACTORIES = [
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_histogram",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_merge_sort_keys",
    "make_merge_sort_pairs",
    "make_radix_rank",
    "make_radix_sort_keys",
    "make_radix_sort_keys_descending",
    "make_radix_sort_pairs",
    "make_radix_sort_pairs_descending",
    "make_reduce",
    "make_run_length",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
    "make_topk_max_keys",
    "make_topk_max_pairs",
    "make_topk_min_keys",
    "make_topk_min_pairs",
]

_EXPECTED_WARP_FACTORIES = [
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_max",
    "make_merge_sort_keys",
    "make_merge_sort_pairs",
    "make_min",
    "make_reduce",
    "make_store",
    "make_sum",
]

EXPECTED_CUTE_BLOCK_EXPORTS = sorted(
    [*_EXPECTED_BLOCK_PRIMITIVES, *_EXPECTED_BLOCK_FACTORIES]
)
EXPECTED_CUTE_WARP_EXPORTS = sorted(
    [*_EXPECTED_WARP_PRIMITIVES, *_EXPECTED_WARP_FACTORIES]
)


def test_cutlass_public_root_exports_are_group_first_after_runtime_validation():
    script = textwrap.dedent(
        """
        import importlib
        import sys
        import types

        import cuda.coop.cutlass as coop

        expected = [
            "Hierarchy", "Payload", "TempStorage",
            "ThreadData", "ThreadDataLoadSource", "ThreadDataSource",
            "ThreadDataTensorMetadata", "ThreadGroup", "ThreadHierarchy",
            "adjacent_difference", "aot", "discontinuity",
            "exchange", "exclusive_scan", "exclusive_sum", "histogram",
            "inclusive_scan", "inclusive_sum", "load", "merge_sort_keys",
            "merge_sort_pairs", "radix_rank", "radix_sort_keys",
            "radix_sort_pairs", "reduce", "run_length_decode", "scan",
            "shuffle", "store", "sum", "this_block", "this_cluster",
            "this_grid", "this_thread", "this_warp", "topk_max_keys",
            "topk_max_pairs", "topk_min_keys", "topk_min_pairs",
        ]
        assert coop.__all__ == expected
        assert sorted(name for name in dir(coop) if not name.startswith("_")) == expected
        assert "cuda.coop.cutlass.block" not in sys.modules
        assert "cuda.coop.cutlass.warp" not in sys.modules
        assert not hasattr(coop, "block")
        assert not hasattr(coop, "warp")
        assert "numpy" not in sys.modules
        assert "torch" not in sys.modules

        root_aot = coop.aot
        assert root_aot.__name__ == "cuda.coop.cutlass.aot"
        assert "cuda.bindings.nvrtc" not in sys.modules
        assert "cuda.bindings.nvjitlink" not in sys.modules
        assert "torch" not in sys.modules

        root_temp_storage = coop.TempStorage
        assert root_temp_storage.__module__ == "cuda.coop.cutlass._temp_storage"
        assert "cuda.coop.cutlass.block" not in sys.modules

        for retired in (
            "cuda.coop.cutlass.block",
            "cuda.coop.cutlass.warp",
        ):
            try:
                importlib.import_module(retired)
            except ModuleNotFoundError as exc:
                assert exc.name == retired
            else:
                raise AssertionError(f"retired public scope {retired!r} imported")

        root_block = importlib.import_module("cuda.coop.cutlass._block")
        root_warp = importlib.import_module("cuda.coop.cutlass._warp")
        implementation_block = importlib.import_module("cuda.coop.cutlass._dsl.block")
        implementation_warp = importlib.import_module("cuda.coop.cutlass._dsl.warp")

        assert coop._block is root_block
        assert coop._warp is root_warp
        assert isinstance(coop._block, types.ModuleType)
        assert isinstance(coop._warp, types.ModuleType)
        assert coop._block.__all__ == implementation_block.__all__
        assert coop._warp.__all__ == implementation_warp.__all__
        assert root_temp_storage is coop._block.TempStorage
        assert (
            coop._block.radix_sort_keys.__wrapped__
            is implementation_block.radix_sort_keys
        )
        assert (
            coop._warp.exclusive_sum.__wrapped__
            is implementation_warp.exclusive_sum
        )
        assert coop._block.make_load.__module__ == "cuda.coop.cutlass._block"
        assert coop._warp.make_store.__module__ == "cuda.coop.cutlass._warp"
        assert list(coop.Payload) == [coop.Payload.PRIMS]
        assert coop.Payload.PRIMS == "prims"
        assert "torch" not in sys.modules
        """
    )

    result = _run_python_with_source(_with_fake_cutlass_runtime(script))

    assert result.returncode == 0, result.stderr


def test_cutlass_private_dsl_import_is_lazy():
    script = textwrap.dedent(
        """
        import sys

        import cuda.coop.cutlass._dsl as dsl

        assert dsl.__all__ == ()
        assert "cuda.coop.cutlass._dsl.block" not in sys.modules
        assert "cuda.coop.cutlass._dsl.warp" not in sys.modules
        """
    )
    result = _run_python_with_source(_with_fake_cutlass_runtime(script))
    assert result.returncode == 0, result.stderr


def test_cute_internal_scoped_callables_have_docstrings():
    script = textwrap.dedent(
        f"""
        import inspect

        import cuda.coop.cutlass as coop

        scoped_exports = {{
            "_block": {EXPECTED_CUTE_BLOCK_EXPORTS!r},
            "_warp": {EXPECTED_CUTE_WARP_EXPORTS!r},
        }}

        for scope_name, exported_names in scoped_exports.items():
            scope = getattr(coop, scope_name)
            for name in exported_names:
                obj = getattr(scope, name)
                if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                    continue
                doc = inspect.getdoc(obj)
                assert doc, f"{{scope_name}}.{{name}} is missing a docstring"
        """
    )

    result = _run_python_with_source(_with_fake_cutlass_runtime(script))

    assert result.returncode == 0, result.stderr


def test_cutlass_public_exports_are_group_first_and_scopes_are_private():
    script = textwrap.dedent(
        """
        import importlib.util
        import inspect

        import cuda.coop.cutlass as coop

        assert coop.ThreadData.__module__ == "cuda.coop.cutlass._internal._thread_data"
        assert (
            coop.ThreadDataLoadSource.__module__
            == "cuda.coop.cutlass._internal._thread_data"
        )
        assert (
            coop.ThreadDataSource.__module__
            == "cuda.coop.cutlass._internal._thread_data"
        )
        assert (
            coop.ThreadDataTensorMetadata.__module__
            == "cuda.coop.cutlass._internal._thread_data"
        )
        assert coop.reduce.__module__ == "cuda.coop.cutlass._group_reduce"
        assert coop.scan.__module__ == "cuda.coop.cutlass._group_scan"
        assert coop.exchange.__module__ == "cuda.coop.cutlass._group_exchange"
        assert coop.load.__module__ == "cuda.coop.cutlass._group_load_store"
        assert coop.store.__module__ == "cuda.coop.cutlass._group_load_store"
        assert coop.this_block.__module__ == "cuda.coop.cutlass._thread_group"
        assert not hasattr(coop, "block")
        assert not hasattr(coop, "warp")
        assert importlib.util.find_spec("cuda.coop.cutlass.block") is None
        assert importlib.util.find_spec("cuda.coop.cutlass.warp") is None
        assert coop._block.__name__ == "cuda.coop.cutlass._block"
        assert coop._warp.__name__ == "cuda.coop.cutlass._warp"
        assert not hasattr(coop._block, "ThreadData")
        assert not hasattr(coop._warp, "ThreadData")
        assert importlib.util.find_spec("cuda.coop.cutlass._dsl._factory") is not None
        assert importlib.util.find_spec("cuda.coop.cutlass._dsl._provider") is not None
        assert importlib.util.find_spec("cuda.coop.cutlass._dsl.block._provider") is not None
        assert importlib.util.find_spec("cuda.coop.cutlass._dsl.warp._provider") is not None

        for primitive in (
            coop.load,
            coop.reduce,
            coop.scan,
            coop.exchange,
            coop.store,
        ):
            parameters = tuple(inspect.signature(primitive).parameters.values())
            assert parameters[0].name == "group"
            assert parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY
            assert parameters[0].default is inspect.Parameter.empty

        for primitive in (coop.reduce, coop.scan, coop.exchange):
            parameters = tuple(inspect.signature(primitive).parameters.values())
            assert [parameter.name for parameter in parameters[:2]] == [
                "group",
                "value",
            ]
            assert all(
                parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                for parameter in parameters[:2]
            )
            assert all(
                parameter.default is inspect.Parameter.empty
                for parameter in parameters[:2]
            )
        assert inspect.signature(coop.reduce).parameters["broadcast"].default is True
        """
    )

    result = _run_python_with_source(_with_fake_cutlass_runtime(script))

    assert result.returncode == 0, result.stderr
