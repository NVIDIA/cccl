# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


def _cute_example_paths() -> list[Path]:
    paths = sorted((SOURCE_ROOT / "examples" / "cutlass").glob("cute_*.py"))
    assert paths, "no CuTe examples found under examples/cutlass"
    return paths


def _cutlass_example_paths() -> list[Path]:
    paths = sorted(
        path
        for path in (SOURCE_ROOT / "examples" / "cutlass").glob("*.py")
        if not path.name.startswith("_") and path.name != "__init__.py"
    )
    assert paths, "no CUTLASS examples found under examples/cutlass"
    return paths


def _run_python_with_source(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{python_path}" if python_path else str(SOURCE_ROOT)
    )

    return subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _load_example_runtime(monkeypatch):
    path = SOURCE_ROOT / "examples" / "cutlass" / "_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "cuda_coop_example_runtime_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _clear_loaded_cutlass_modules(monkeypatch, runtime) -> None:
    for module_name in tuple(sys.modules):
        if module_name == runtime.ROOT_SCOPE or module_name.startswith(
            f"{runtime.ROOT_SCOPE}."
        ):
            monkeypatch.delitem(sys.modules, module_name)


def _coop_parent(*paths: Path, module_file: Path | None = None):
    rendered = [str(path) for path in paths]
    return SimpleNamespace(
        __file__=None if module_file is None else str(module_file),
        __path__=rendered,
        __spec__=SimpleNamespace(submodule_search_locations=list(rendered)),
    )


def _assert_uses_int32_runtime(text: str) -> None:
    assert re.search(r"require_runtime\(\s*include_int32=True\s*\)", text)


def _words(text: str) -> str:
    return " ".join(text.split())


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _assert_no_payload_calls(text: str) -> None:
    tree = ast.parse(text)
    offenders = sorted(
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "payload" for keyword in node.keywords)
    )
    assert not offenders, f"unexpected payload selectors at: {offenders}"


def _assert_no_public_scoped_calls(text: str) -> None:
    for scope in ("block", "warp"):
        assert f"coop.{scope}." not in text


def test_readme_lists_public_cutlass_example_commands():
    readme = (SOURCE_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [
        f"python -m examples.cutlass.{path.stem}"
        for path in _cutlass_example_paths()
        if f"python -m examples.cutlass.{path.stem}" not in readme
    ]

    assert missing == []


def test_cutlass_examples_package_all_matches_public_modules():
    expected = [path.stem for path in _cutlass_example_paths()]
    script = textwrap.dedent(
        f"""
        from examples import cutlass

        assert cutlass.__all__ == {expected!r}
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cute_examples_and_benchmarks_import_without_runtime_dependencies():
    script = textwrap.dedent(
        """
        import sys

        from benchmarks.cute import bench
        from examples.cutlass import (
            cute_kmeans_assign_gemm_argmin,
            cute_kmeans_assign_topk,
            cute_legacy_reduce_compare,
            cute_mma_amax_sm100,
            cute_mma_topk,
            cute_mma_topk_sm100,
            cute_run_length_decode_window,
            cute_scheduler_prefix,
            cute_sort_register_fragment,
            cute_sort_and_segment,
            cute_sort_and_segment_thread_data,
            cute_thread_group_descriptor_reduce,
            cute_thread_group_query,
            cute_thread_group_reduce,
            cute_thread_hierarchy_reduce,
            cute_topk_score_window,
            cute_warp_merge_sort,
            cute_warp_prefix_reduce,
            mixed_payload_factory_sort_topk,
            mixed_payload_sort_topk,
            mixed_tensor_vector_scan,
            portable_root_sum,
            prims_vector_block_exchange,
            prims_vector_block_prefix_segment,
            prims_vector_histogram_run_length,
            prims_vector_pair_sort_topk,
            prims_vector_rank_merge,
            prims_vector_sort_topk,
            prims_vector_warp_merge_sort,
            prims_vector_warp_prefix,
        )

        assert callable(cute_kmeans_assign_gemm_argmin.run_example)
        assert callable(cute_kmeans_assign_topk.run_example)
        assert callable(cute_kmeans_assign_topk.prepare_batched_example)
        assert callable(cute_kmeans_assign_topk.prepare_feature_split_batched_example)
        assert callable(
            cute_kmeans_assign_topk.prepare_feature_split_score_batched_example
        )
        assert callable(
            cute_kmeans_assign_topk.prepare_feature_split_top1_score_batched_example
        )
        assert callable(
            cute_kmeans_assign_topk.prepare_feature_split_top1_score_warp_batched_example
        )
        assert callable(cute_kmeans_assign_topk.prepare_wide_batched_example)
        assert callable(cute_legacy_reduce_compare.run_example)
        assert callable(cute_mma_amax_sm100.compile_example)
        assert callable(cute_mma_amax_sm100.run_example)
        assert callable(cute_mma_amax_sm100.benchmark_example)
        assert callable(cute_mma_amax_sm100.prepare_benchmark)
        assert callable(cute_mma_topk.run_example)
        assert callable(cute_mma_topk_sm100.compile_example)
        assert callable(cute_mma_topk_sm100.run_example)
        assert callable(cute_mma_topk_sm100.benchmark_example)
        assert callable(cute_mma_topk_sm100.prepare_block_merge_benchmark)
        assert callable(cute_mma_topk_sm100.prepare_warp_merge_benchmark)
        assert callable(cute_run_length_decode_window.run_example)
        assert callable(cute_scheduler_prefix.run_example)
        assert callable(cute_sort_register_fragment.run_example)
        assert callable(cute_sort_and_segment.run_example)
        assert callable(cute_sort_and_segment_thread_data.run_example)
        assert callable(cute_thread_group_descriptor_reduce.run_example)
        assert callable(cute_thread_group_query.run_example)
        assert callable(cute_thread_group_reduce.run_example)
        assert callable(cute_thread_hierarchy_reduce.run_example)
        assert callable(cute_topk_score_window.run_example)
        assert callable(cute_warp_merge_sort.run_example)
        assert callable(cute_warp_prefix_reduce.run_example)
        assert callable(mixed_payload_factory_sort_topk.run_example)
        assert callable(mixed_payload_sort_topk.run_example)
        assert callable(mixed_tensor_vector_scan.run_example)
        assert callable(portable_root_sum.run_example)
        assert callable(prims_vector_block_exchange.run_example)
        assert callable(prims_vector_block_prefix_segment.run_example)
        assert callable(prims_vector_histogram_run_length.run_example)
        assert callable(prims_vector_pair_sort_topk.run_example)
        assert callable(prims_vector_rank_merge.run_example)
        assert callable(prims_vector_sort_topk.run_example)
        assert callable(prims_vector_warp_merge_sort.run_example)
        assert callable(prims_vector_warp_prefix.run_example)
        default_scenarios = (
            "cute_kmeans_assign_topk",
            "cute_kmeans_assign_topk_batched",
            "cute_kmeans_assign_topk_feature_split_batched",
            "cute_kmeans_assign_topk_feature_split_score_batched",
            "cute_kmeans_assign_topk_feature_split_top1_score_batched",
            "cute_kmeans_assign_topk_feature_split_top1_score_warp_batched",
            "cute_kmeans_assign_topk_wide_batched",
            "cute_kmeans_assign_gemm_argmin",
            "kmeans_assign_torch_gemm_argmin_reference",
            "kmeans_assign_cute_gemm_coop_argmin_reference",
            "cute_legacy_reduce_compare",
            "cute_run_length_decode_window",
            "cute_scheduler_prefix",
            "cute_sort_register_fragment",
            "cute_sort_and_segment",
            "cute_sort_and_segment_thread_data",
            "cute_thread_group_descriptor_reduce",
            "cute_thread_group_query",
            "cute_thread_group_reduce",
            "cute_thread_hierarchy_reduce",
            "cute_topk_score_window",
            "cute_warp_merge_sort",
            "cute_warp_prefix_reduce",
        )
        optional_prims_scenarios = (
            "mixed_payload_factory_sort_topk",
            "mixed_payload_sort_topk",
            "mixed_tensor_vector_scan",
            "prims_vector_block_exchange",
            "prims_vector_block_prefix_segment",
            "prims_vector_histogram_run_length",
            "prims_vector_pair_sort_topk",
            "prims_vector_rank_merge",
            "prims_vector_sort_topk",
            "prims_vector_warp_merge_sort",
            "prims_vector_warp_prefix",
        )
        optional_sm100_scenarios = (
            "cute_mma_amax_sm100_batched",
            "cute_mma_topk_sm100_block_merge_batched",
            "cute_mma_topk_sm100_warp_merge_batched",
        )
        assert bench.default_scenarios() == default_scenarios
        assert bench.available_scenarios() == (
            *default_scenarios,
            *optional_sm100_scenarios,
            *optional_prims_scenarios,
        )
        result = bench.ScenarioResult(
            name="probe",
            first_launch_us=1.0,
            steady_launch_us=2.0,
            steady_kernel_us=0.5,
        ).to_dict()
        assert "first_launch_us" in result
        assert "steady_launch_us" in result
        assert result["steady_kernel_us"] == 0.5

        try:
            bench.run_sanity_suite(scenarios=["cute_kmeans_assign_topk"], timer="bogus")
        except ValueError as exc:
            assert "timer must be" in str(exc)
        else:
            raise AssertionError("invalid timer did not raise")

        assert "cutlass" not in sys.modules
        assert "torch" not in sys.modules

        forwarded = []

        def fake_measure(name, *, warmup_iters, measure_iters, timer):
            forwarded.append((name, warmup_iters, measure_iters, timer))
            return bench.ScenarioResult(
                name=name,
                first_launch_us=3.0,
                steady_launch_us=4.0,
                steady_kernel_us=5.0 if timer == "cupti" else None,
            )

        original_measure_scenario = bench._measure_scenario
        try:
            bench._measure_scenario = fake_measure
            default_result = bench.run_sanity_suite(
                warmup_iters=1,
                measure_iters=2,
                timer="wall",
            )
            assert forwarded == [
                (name, 1, 2, "wall") for name in bench.default_scenarios()
            ]
            assert default_result == [
                {
                    "name": name,
                    "first_launch_us": 3.0,
                    "steady_launch_us": 4.0,
                }
                for name in bench.default_scenarios()
            ]
            forwarded.clear()
            forwarded_result = bench.run_sanity_suite(
                scenarios=["mixed_payload_sort_topk"],
                warmup_iters=1,
                measure_iters=2,
                timer="cupti",
            )
        finally:
            bench._measure_scenario = original_measure_scenario
        assert forwarded == [("mixed_payload_sort_topk", 1, 2, "cupti")]
        assert forwarded_result == [
            {
                "name": "mixed_payload_sort_topk",
                "first_launch_us": 3.0,
                "steady_launch_us": 4.0,
                "steady_kernel_us": 5.0,
            }
        ]

        import types

        cutlass_module = types.ModuleType("cutlass")
        cutlass_module.__path__ = []
        cute_module = types.ModuleType("cutlass.cute")
        cute_module.__path__ = []
        testing_module = types.ModuleType("cutlass.cute.testing")

        class JitArguments:
            pass

        called = []

        def benchmark(
            step,
            *,
            warmup_iterations,
            iterations,
            kernel_arguments,
            use_cupti,
        ):
            called.append((warmup_iterations, iterations, use_cupti))
            step()
            assert isinstance(kernel_arguments, JitArguments)
            return 6.5

        testing_module.JitArguments = JitArguments
        testing_module.benchmark = benchmark
        cutlass_module.cute = cute_module
        cute_module.testing = testing_module
        module_names = ("cutlass", "cutlass.cute", "cutlass.cute.testing")
        saved_modules = {name: sys.modules.get(name) for name in module_names}
        try:
            sys.modules["cutlass"] = cutlass_module
            sys.modules["cutlass.cute"] = cute_module
            sys.modules["cutlass.cute.testing"] = testing_module

            stepped = []
            assert (
                bench._measure_kernel_activity_us(
                    lambda: stepped.append(True),
                    warmup_iters=0,
                    measure_iters=2,
                )
                == 6.5
            )
            assert called == [(1, 2, True)]
            assert stepped == [True]
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_example_runtime_prefers_source_path_by_default(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    wheel_path = tmp_path / "wheel" / "cuda" / "coop"
    parent = _coop_parent(wheel_path, runtime.SOURCE_COOP_PATH)
    monkeypatch.setattr(runtime.sys, "path", [str(tmp_path / "wheel")])
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda module_name: parent,
    )

    runtime._prepend_source_cuda_coop_path()

    assert parent.__path__ == [
        str(runtime.SOURCE_COOP_PATH),
        str(wheel_path),
    ]
    assert parent.__spec__.submodule_search_locations == parent.__path__
    assert runtime.sys.path[0] == str(runtime.SOURCE_ROOT)


def test_cutlass_example_runtime_rejects_invalid_installed_mode(monkeypatch):
    runtime = _load_example_runtime(monkeypatch)
    monkeypatch.setenv(runtime._INSTALLED_COOP_ENV, "auto")

    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        runtime._import_cuda_coop_cutlass()


def test_cutlass_example_runtime_filters_parent_before_installed_import(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    _clear_loaded_cutlass_modules(monkeypatch, runtime)
    environment_prefix = tmp_path / "env"
    installed_coop = (
        environment_prefix / "lib" / "python3.14" / "site-packages" / "cuda" / "coop"
    )
    installed_module_path = installed_coop / "cutlass" / "__init__.py"
    installed_module_path.parent.mkdir(parents=True)
    installed_module_path.touch()
    installed_parent_path = installed_coop / "__init__.py"
    installed_parent_path.touch()
    source_coop = tmp_path / "checkout" / "cuda" / "coop"
    source_coop.mkdir(parents=True)
    parent = _coop_parent(
        source_coop,
        installed_coop,
        module_file=installed_parent_path,
    )
    installed_module = SimpleNamespace(__file__=str(installed_module_path))
    imports = []
    retained_path = tmp_path / "retained"
    monkeypatch.setattr(
        runtime.sys,
        "path",
        [str(runtime.SOURCE_ROOT), str(retained_path)],
    )

    def import_module(module_name):
        imports.append(module_name)
        assert runtime.sys.path == [str(retained_path)]
        if module_name == runtime.COOP_PARENT_SCOPE:
            return parent
        assert module_name == runtime.ROOT_SCOPE
        assert parent.__path__ == [str(installed_coop)]
        assert parent.__spec__.submodule_search_locations == [str(installed_coop)]
        return installed_module

    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(runtime.sys, "meta_path", [])
    monkeypatch.setenv(runtime._INSTALLED_COOP_ENV, "1")
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime.importlib, "invalidate_caches", lambda: None)

    assert runtime._import_cuda_coop_cutlass() is installed_module
    assert imports == [runtime.COOP_PARENT_SCOPE, runtime.ROOT_SCOPE]


@pytest.mark.parametrize("use_symlink", [False, True])
def test_cutlass_example_runtime_requires_installed_parent_path(
    monkeypatch,
    tmp_path,
    use_symlink,
):
    runtime = _load_example_runtime(monkeypatch)
    environment_prefix = tmp_path / "env"
    source_coop = tmp_path / "checkout" / "cuda" / "coop"
    source_coop.mkdir(parents=True)
    parent_path = source_coop
    if use_symlink:
        parent_path = environment_prefix / "lib" / "cuda" / "coop"
        parent_path.parent.mkdir(parents=True)
        try:
            parent_path.symlink_to(source_coop, target_is_directory=True)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                pytest.skip(
                    "Windows symlink creation requires Developer Mode or "
                    "SeCreateSymbolicLinkPrivilege"
                )
            raise
    parent = _coop_parent(
        parent_path,
        module_file=environment_prefix / "site-packages" / "cuda" / "coop.py",
    )

    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(runtime.sys, "meta_path", [])
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda module_name: parent,
    )

    with pytest.raises(RuntimeError, match="has no package path"):
        runtime._prepare_installed_cuda_coop_parent()


def test_cutlass_example_runtime_rejects_source_parent_before_child_import(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    _clear_loaded_cutlass_modules(monkeypatch, runtime)
    environment_prefix = tmp_path / "env"
    installed_coop = environment_prefix / "site-packages" / "cuda" / "coop"
    installed_coop.mkdir(parents=True)
    source_parent = _coop_parent(
        installed_coop,
        module_file=tmp_path / "checkout" / "cuda" / "coop" / "__init__.py",
    )
    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(runtime.sys, "meta_path", [])
    imports = []

    def import_module(module_name):
        imports.append(module_name)
        assert module_name == runtime.COOP_PARENT_SCOPE
        return source_parent

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="did not resolve"):
        runtime._import_installed_cuda_coop_cutlass()

    assert imports == [runtime.COOP_PARENT_SCOPE]


@pytest.mark.parametrize("module_suffix", ["", ".warp"])
def test_cutlass_example_runtime_rejects_preloaded_source_before_child_import(
    monkeypatch,
    tmp_path,
    module_suffix,
):
    runtime = _load_example_runtime(monkeypatch)
    _clear_loaded_cutlass_modules(monkeypatch, runtime)
    environment_prefix = tmp_path / "env"
    installed_coop = environment_prefix / "site-packages" / "cuda" / "coop"
    installed_coop.mkdir(parents=True)
    parent = _coop_parent(
        installed_coop,
        module_file=installed_coop / "__init__.py",
    )
    source_module = SimpleNamespace(
        __file__=str(tmp_path / "checkout" / "cuda" / "coop" / "cutlass.py")
    )
    loaded_name = f"{runtime.ROOT_SCOPE}{module_suffix}"
    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(runtime.sys, "meta_path", [])
    monkeypatch.setitem(sys.modules, loaded_name, source_module)
    imports = []

    def import_module(module_name):
        imports.append(module_name)
        assert module_name == runtime.COOP_PARENT_SCOPE
        return parent

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="already loaded outside"):
        runtime._import_installed_cuda_coop_cutlass()

    assert imports == [runtime.COOP_PARENT_SCOPE]


def test_cutlass_example_runtime_rejects_editable_finder(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    finder_type = type("Finder", (), {})
    finder_type.__module__ = "_cuda_coop_editable"
    monkeypatch.setattr(runtime.sys, "meta_path", [finder_type()])
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path / "env"))

    with pytest.raises(RuntimeError, match="rejects cuda-coop editable"):
        runtime._prepare_installed_cuda_coop_parent()


def test_cutlass_example_runtime_rejects_editable_with_preloaded_wheel(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    _clear_loaded_cutlass_modules(monkeypatch, runtime)
    environment_prefix = tmp_path / "env"
    installed_module_path = (
        environment_prefix
        / "site-packages"
        / "cuda"
        / "coop"
        / "cutlass"
        / "__init__.py"
    )
    installed_module = SimpleNamespace(__file__=str(installed_module_path))
    finder_type = type("Finder", (), {})
    finder_type.__module__ = "_cuda_coop_editable"
    monkeypatch.setattr(runtime.sys, "meta_path", [finder_type()])
    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setitem(sys.modules, runtime.ROOT_SCOPE, installed_module)

    with pytest.raises(RuntimeError, match="rejects cuda-coop editable"):
        runtime._import_installed_cuda_coop_cutlass()


def test_cutlass_example_runtime_rejects_out_of_prefix_import(
    monkeypatch,
    tmp_path,
):
    runtime = _load_example_runtime(monkeypatch)
    _clear_loaded_cutlass_modules(monkeypatch, runtime)
    environment_prefix = tmp_path / "env"
    installed_coop = environment_prefix / "site-packages" / "cuda" / "coop"
    installed_coop.mkdir(parents=True)
    parent = _coop_parent(
        installed_coop,
        module_file=installed_coop / "__init__.py",
    )
    source_module = SimpleNamespace(
        __file__=str(tmp_path / "checkout" / "cuda" / "coop" / "cutlass.py")
    )

    def import_module(module_name):
        if module_name == runtime.COOP_PARENT_SCOPE:
            return parent
        assert module_name == runtime.ROOT_SCOPE
        return source_module

    monkeypatch.setattr(runtime.sys, "prefix", str(environment_prefix))
    monkeypatch.setattr(runtime.sys, "meta_path", [])
    monkeypatch.setenv(runtime._INSTALLED_COOP_ENV, "1")
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime.importlib, "invalidate_caches", lambda: None)

    with pytest.raises(RuntimeError, match="did not resolve"):
        runtime._import_cuda_coop_cutlass()


def test_cute_examples_and_benchmarks_compile():
    paths = [
        SOURCE_ROOT / "examples" / "cutlass" / "_runtime.py",
        *_cutlass_example_paths(),
        SOURCE_ROOT / "benchmarks" / "cute" / "bench.py",
    ]
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_public_cutlass_examples_use_only_group_first_collectives():
    for path in _cutlass_example_paths():
        _assert_no_public_scoped_calls(path.read_text(encoding="utf-8"))


def test_cute_mma_topk_example_fuses_selection_with_tensorop_gemm():
    text = (SOURCE_ROOT / "examples" / "cutlass" / "cute_mma_topk.py").read_text(
        encoding="utf-8"
    )

    assert "tensorop_gemm = _load_tensorop_gemm_module()" in text
    assert "gemm = tensorop_gemm.TensorOpGemm(" in text
    assert "epilogue = TileTopKEpilogue()" in text
    assert "tensorop_gemm.bmm," in text
    assert "epilogue_op(tCrC.load())" in text
    assert text.count("coop.topk_max_keys(") == 2
    assert "lambda item: accumulator[" in text
    assert "No full GEMM result is materialized before selection" in text
    assert "torch.topk(" in text


def test_cute_mma_amax_sm100_uses_thread_data_and_cooperative_reduce():
    text = (SOURCE_ROOT / "examples" / "cutlass" / "cute_mma_amax_sm100.py").read_text(
        encoding="utf-8"
    )

    assert "cute/blackwell/kernel/dense_gemm/dense_gemm.py" in text
    assert "gemm = dense_gemm.DenseGemmKernel(" in text
    assert "dense_gemm.bmm," in text
    assert "MMA_TILER_MN = (128, 32)" in text
    assert "BLOCK_THREADS = 128" in text
    assert "SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))" in text
    assert "def _require_supported_compute_capability(" in text
    assert 'TMEM_LOADER_MODE = "tmem_loader"' in text
    assert 'POST_T2R_MODE = "post_t2r"' in text
    assert "class TileAmaxConsumer:" in text
    assert "accumulator = coop.ThreadData.load(accumulator_source)" in text
    assert "magnitude = cute.arch.fmax(value, -value)" in text
    assert "tile_amax = coop.reduce(" in text
    assert "coop.this_block()" in text
    assert 'binary_op="max"' in text
    assert "output.to_tensor_ssa(like=accumulator_source)" in text
    assert "PostT2RAccumulatorSource(accumulator)" in text
    assert "torch.abs(torch.bmm(a.float(), b.float())).amax" in text
    assert '"output_contract": "broadcast_tile_amax"' in text
    assert '"device_compute_capability": device_compute_capability' in text
    assert '"compile_target": compile_target' in text
    assert "BaseDSL._get_dsl().get_arch_enum().name" in text
    assert '"--compile-only"' in text
    assert '"--benchmark"' in text


def test_cute_mma_topk_sm100_supports_tmem_and_post_t2r_source_capabilities():
    text = (SOURCE_ROOT / "examples" / "cutlass" / "cute_mma_topk_sm100.py").read_text(
        encoding="utf-8"
    )

    assert "cute/blackwell/kernel/dense_gemm/dense_gemm.py" in text
    assert "dense_gemm = _load_dense_gemm_module()" in text
    assert "gemm = dense_gemm.DenseGemmKernel(" in text
    assert "dense_gemm.bmm," in text
    assert "USE_2CTA_INSTRS = False" in text
    assert "USE_TMA_STORE = True" in text
    assert "MMA_TILER_MN = (128, 32)" in text
    assert "CLUSTER_SHAPE_MN = (1, 1)" in text
    assert "BLOCK_THREADS = 128" in text
    assert "WARP_THREADS = 32" in text
    assert "CHUNK_ITEMS_PER_THREAD = 8" in text
    assert "SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))" in text
    assert "def _require_supported_compute_capability(" in text
    assert 'TMEM_LOADER_MODE = "tmem_loader"' in text
    assert 'POST_T2R_MODE = "post_t2r"' in text
    assert "class TileTopKConsumer:" in text
    assert "class PostT2RAccumulatorSource:" in text
    assert "def __cuda_coop_thread_data_load__(self) -> Any:" in text
    assert "return self.register_payload" in text
    assert "accumulator_values = coop.ThreadData.load(accumulator_source)" in text
    assert "class PostT2RTopKAdapter:" in text
    assert "PostT2RAccumulatorSource(accumulator)" in text
    assert "accumulator_consumer = topk_consumer" in text
    assert "accumulator_consumer = None" in text
    assert "if accumulator_consumer is not None:" in text
    assert "compiled = cute.compile(*compile_args)" in text
    assert "def dtype(self) -> Any:" in text
    assert text.count("coop.topk_max_keys(") == 2
    assert "CANDIDATE_COUNT = CHUNKS_PER_THREAD * TOPK" in text
    assert "valid_items=cutlass.const_expr(candidate_count)" in text
    assert "lambda item: accumulator_values[" in text
    assert "if candidate_position < cutlass.Int32(candidate_count):" in text
    assert "selected.to_tensor_ssa(like=accumulator_source)" in text
    assert "if flat_item < cutlass.Int32(TOPK): value =" in _words(text)
    assert "selected[item] = value" in text
    assert "torch.topk(" in text
    assert "torch.bmm(a.float(), b.float()).flatten(start_dim=1)" in _words(text)
    assert 'BLOCK_MERGE_SELECTOR = "block_merge"' in text
    assert 'WARP_MERGE_SELECTOR = "warp_merge"' in text
    assert "coop.merge_sort_keys(" in text
    tree = ast.parse(text)
    warp_merge_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "merge_sort_keys"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "coop"
    ]
    assert len(warp_merge_calls) == 1
    assert ast.unparse(warp_merge_calls[0].args[0]) == "coop.this_warp()"
    assert isinstance(warp_merge_calls[0].args[1], ast.Name)
    assert warp_merge_calls[0].args[1].id == "warp_merge_input"
    assert "temp_storage" not in {
        keyword.arg for keyword in warp_merge_calls[0].keywords
    }
    assert "warp_merge_value = cutlass.Float32(NON_CANDIDATE)" in text
    assert "warp_merge_input = coop.ThreadData.from_values(" in text
    assert "warp_top_score = cutlass.Float32(NON_CANDIDATE)" in text
    assert "if tidx < cutlass.Int32(WARP_THREADS):" in text
    assert "warp_top_scores" not in text
    assert '"output_tiles_per_second"' in text
    assert '"transfer_policy_owner": "cutlass_dense_gemm"' in text
    assert '"thread_data_load_source_hook"' in text
    assert '"cutlass_dense_gemm_epilogue"' in text
    assert '"producer_tmem_accumulator"' in text
    assert '"post_t2r_register"' in text
    assert '"device_compute_capability": device_compute_capability' in text
    assert '"compile_target": compile_target' in text
    assert "BaseDSL._get_dsl().get_arch_enum().name" in text
    assert "trigger the exact ``cute.copy`` selected by CUTLASS" in text
    assert "CUTLASS triggers the transfer first" in _words(text)
    assert "ThreadData.from_tmem" not in text
    assert "releases its mainloop shared-memory partition" in text
    assert "releases without that fix cannot final-link this example" in _words(text)
    assert '"--compile-only"' in text
    assert '"--mode"' in text
    assert '"--selector"' in text
    assert '"--batch-count"' in text
    assert '"--benchmark"' in text


def test_cute_mma_topk_sm100_mode_metadata_distinguishes_transfer_trigger():
    script = textwrap.dedent(
        """
        from examples.cutlass import cute_mma_topk_sm100 as example

        tmem_loader = example._metadata(
            example.TMEM_LOADER_MODE,
            example.WARP_MERGE_SELECTOR,
            17,
        )
        assert tmem_loader["mode"] == "tmem_loader"
        assert tmem_loader["selector"] == "warp_merge"
        assert tmem_loader["batch_count"] == 17
        assert tmem_loader["chunk_items_per_thread"] == 8
        assert tmem_loader["chunk_count"] == 4
        assert tmem_loader["candidate_count"] == 32
        assert tmem_loader["temp_storage_bytes"] == 10240
        assert tmem_loader["transfer_policy_owner"] == "cutlass_dense_gemm"
        assert tmem_loader["transfer_trigger"] == "thread_data_load_source_hook"
        assert tmem_loader["thread_data_source"] == "producer_tmem_accumulator"
        assert tmem_loader["device_compute_capability"] is None
        assert tmem_loader["compile_target"] is None

        class FakeCuda:
            def __init__(self, capability):
                self.capability = capability

            def get_device_capability(self):
                return self.capability

        class FakeTorch:
            def __init__(self, capability):
                self.cuda = FakeCuda(capability)

        for capability in ((10, 0), (10, 3)):
            assert (
                example._require_supported_compute_capability(FakeTorch(capability))
                == capability
            )
        try:
            example._require_supported_compute_capability(FakeTorch((12, 0)))
        except RuntimeError as exc:
            assert "supported compute capabilities are 10.0, 10.3" in str(exc)
            assert "found 12.0" in str(exc)
        else:
            raise AssertionError("unsupported compute capability was accepted")

        post_t2r = example._metadata(
            example.POST_T2R_MODE,
            example.BLOCK_MERGE_SELECTOR,
            1,
        )
        assert post_t2r["mode"] == "post_t2r"
        assert post_t2r["selector"] == "block_merge"
        assert post_t2r["batch_count"] == 1
        assert post_t2r["transfer_policy_owner"] == "cutlass_dense_gemm"
        assert post_t2r["transfer_trigger"] == "cutlass_dense_gemm_epilogue"
        assert post_t2r["thread_data_source"] == "post_t2r_register"

        try:
            example._metadata("implicit_tmem", "block_merge", 1)
        except ValueError as exc:
            assert "mode must be one of: tmem_loader, post_t2r" in str(exc)
        else:
            raise AssertionError("invalid accumulator-source mode was accepted")

        try:
            example._metadata("tmem_loader", "full_block", 1)
        except ValueError as exc:
            assert "selector must be one of: block_merge, warp_merge" in str(exc)
        else:
            raise AssertionError("invalid selector was accepted")

        for invalid_batch_count in (True, 1.5):
            try:
                example._metadata("tmem_loader", "block_merge", invalid_batch_count)
            except TypeError as exc:
                assert "batch_count must be an integer" in str(exc)
            else:
                raise AssertionError("non-integral batch count was accepted")

        try:
            example._metadata("tmem_loader", "block_merge", 0)
        except ValueError as exc:
            assert "batch_count must be positive" in str(exc)
        else:
            raise AssertionError("non-positive batch count was accepted")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cute_examples_use_direct_single_phase_primitives():
    factory_call = re.compile(r"\bcoop\.(?:block|warp)\.make_[A-Za-z0-9_]+\s*\(")
    run_length_decode_call = re.compile(r"\bcoop\.block\.run_length_decode\s*\(")
    offenders = []

    for path in _cute_example_paths():
        text = path.read_text(encoding="utf-8")
        for match in factory_call.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(
                f"{path.relative_to(SOURCE_ROOT)}:{line_no}: factory call",
            )
        for match in run_length_decode_call.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(
                f"{path.relative_to(SOURCE_ROOT)}:{line_no}: use run_length().decode()",
            )

    assert offenders == []


def test_cute_sort_and_segment_example_uses_single_item_payloads():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "cute_sort_and_segment.py"
    ).read_text(encoding="utf-8")

    assert "ITEMS_PER_THREAD = 1" in text
    assert "keys = coop.ThreadData(ITEMS_PER_THREAD, Int32)" in text
    assert (
        "coop.load(\n            block,\n            keys_in,\n            keys,"
        in text
    )
    assert "lanes = coop.ThreadData.from_values(" in text
    assert "key = keys_in[tidx]" not in text
    assert "sorted_key_out[tidx]" not in text
    assert "sorted_lane_out[tidx]" not in text
    assert "head_out[tidx]" not in text
    assert "run_id_out[tidx]" not in text
    assert "sorted_keys, sorted_lanes = coop.radix_sort_pairs(" in text
    assert "heads = coop.discontinuity(" in text
    assert "run_ids = coop.exclusive_sum(" in text
    for destination in ("sorted_key_out", "sorted_lane_out", "head_out", "run_id_out"):
        assert f"coop.store(\n            block,\n            {destination}," in text


def test_cute_sort_and_segment_thread_data_example_uses_multi_item_payloads():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "cute_sort_and_segment_thread_data.py"
    ).read_text(encoding="utf-8")

    assert "ITEMS_PER_THREAD = 2" in text
    assert "items_per_thread: cutlass.Constexpr" in text
    assert "keys = coop.ThreadData(items_per_thread, Int32)" in text
    assert (
        "coop.load(\n            block,\n            keys_in,\n            keys,"
        in text
    )
    assert "lanes = coop.ThreadData.from_fn(" in text
    assert "lambda item: base_lane + Int32(item)" in text
    assert "lanes[item]" not in text
    assert "keys[item] = keys_in" not in text
    assert "sorted_key_out[out_idx]" not in text
    for destination in ("sorted_key_out", "sorted_lane_out", "head_out", "run_id_out"):
        assert f"coop.store(\n            block,\n            {destination}," in text
    assert "ITEMS_PER_THREAD,\n        ).launch" in text
    assert "sorted_keys, sorted_lanes = coop.radix_sort_pairs(" in text
    assert "heads = coop.discontinuity(" in text
    assert "run_ids = coop.exclusive_sum(" in text


def test_cute_sort_register_fragment_example_uses_rmem_bridge():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "cute_sort_register_fragment.py"
    ).read_text(encoding="utf-8")

    assert "ITEMS_PER_THREAD = 2" in text
    assert "items_per_thread: cutlass.Constexpr" in text
    assert "key_fragment = cute.make_rmem_tensor(" in text
    assert (
        "coop.radix_sort_keys(\n            block,\n            key_fragment," in text
    )
    assert "keys = coop.ThreadData.from_payload(" not in text
    assert "keys = coop.ThreadData.from_register_tensor(" not in text
    assert "items_per_thread=items_per_thread" not in text
    assert "cutlass.range_constexpr(items_per_thread)" in text
    assert "sorted_keys = coop.radix_sort_keys(" in text
    assert "coop.store(\n            block,\n            sorted_key_out," in text
    assert "sorted_key_out[base + item]" not in text


def test_prims_vector_sort_topk_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_sort_topk.py"
    ).read_text(encoding="utf-8")

    assert "require_runtime()" in text
    assert "items_per_thread: cutlass.Constexpr" in text
    assert "keys_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    _assert_no_payload_calls(text)
    assert (
        "coop.load(\n            block,\n            keys_in,\n            keys_vec,"
        in text
    )
    assert "coop.radix_sort_keys(\n            block,\n            keys_vec," in text
    assert "coop.topk_max_keys(\n            block,\n            keys_vec," in text
    assert "temp_storage=topk_temp_storage" in text
    assert "TOPK_VALID_ITEMS = TOTAL_ITEMS - 9" in text
    assert "keys_host[:TOPK_VALID_ITEMS]" in text


def test_prims_vector_pair_sort_topk_example_uses_group_first_payloads():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_pair_sort_topk.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    assert "block = coop.this_block()" in text
    assert "keys_vec = coop.ThreadData(" in text
    assert "values_vec = coop.ThreadData(" in text
    assert (
        "coop.load(\n            block,\n            keys_in,\n            keys_vec,"
        in text
    )
    assert (
        "coop.load(\n            block,\n            values_in,\n            values_vec,"
        in text
    )
    _assert_no_payload_calls(text)
    assert "coop.radix_sort_pairs(\n            block," in text
    assert "coop.topk_min_pairs(\n            block," in text
    assert "temp_storage=topk_temp_storage" in text
    assert "coop.store(block, sorted_keys_out, sorted_keys)" in text
    assert "coop.store(block, top_pair_values_out, top_pair_values)" in text


def test_prims_vector_rank_merge_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_rank_merge.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    assert "keys_vec = coop.ThreadData(items_per_thread, Int32)" in text
    assert "values_vec = coop.ThreadData(items_per_thread, Int32)" in text
    _assert_no_payload_calls(text)
    assert "coop.radix_rank(\n            block,\n            keys_vec," in text
    assert "coop.merge_sort_pairs(" in text
    assert "coop.merge_sort_keys(" in text
    assert "valid_items=valid_items" in text
    assert "exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)" in text


def test_prims_vector_block_exchange_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_block_exchange.py"
    ).read_text(encoding="utf-8")

    for name in ("blocked_vec", "striped_vec", "reverse_ranks_vec", "valid_flags_vec"):
        assert f"{name} = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    _assert_no_payload_calls(text)
    assert 'mode="striped_to_blocked"' in text
    assert 'mode="blocked_to_striped"' in text
    assert 'mode="scatter_to_striped"' in text
    assert 'mode="scatter_to_striped_flagged"' in text


def test_prims_vector_block_prefix_segment_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_block_prefix_segment.py"
    ).read_text(encoding="utf-8")

    assert "values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    assert "segments_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    _assert_no_payload_calls(text)
    assert "coop.exclusive_sum(" in text
    assert "coop.inclusive_scan(" in text
    assert "coop.adjacent_difference(" in text
    assert "coop.discontinuity(" in text
    assert "coop.shuffle(" in text
    assert "aggregate_output=sum_aggregate" in text
    assert "block_prefix=shuffle_prefix" in text


def test_prims_vector_histogram_run_length_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_histogram_run_length.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    assert "samples_vec = coop.ThreadData(items_per_thread, Int32)" in text
    assert "run_values_vec = coop.ThreadData(items_per_thread, Int32)" in text
    assert "run_lengths_vec = coop.ThreadData(items_per_thread, Uint32)" in text
    _assert_no_payload_calls(text)
    assert "coop.histogram(\n            block," in text
    assert "decoded = coop.run_length_decode(" in text
    assert "decoded_window_offset=" in text


def test_cute_run_length_decode_window_example_uses_group_decode_and_stores():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "cute_run_length_decode_window.py"
    ).read_text(encoding="utf-8")

    assert "decoded = coop.run_length_decode(" in text
    assert "block = coop.this_block()" in text
    for destination in ("decoded_out", "offsets_out", "total_out"):
        assert f"coop.store(\n            block,\n            {destination}," in text
    assert "decoded_out[out_base + 0]" not in text
    assert "offsets_out[out_base + 0]" not in text
    assert "total_out[tidx] = total_decoded_size[0]" not in text


def test_cute_warp_prefix_reduce_example_uses_group_first_prefix_store():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "cute_warp_prefix_reduce.py"
    ).read_text(encoding="utf-8")

    assert "prefix_value = coop.ThreadData.from_values(" in text
    assert "coop.exclusive_sum(warp, value)" in text
    assert "coop.store(warp, prefix_out, prefix_value)" in text
    assert "prefix_out[tidx] = coop.exclusive_sum(" not in text
    assert "warp_totals[warp_id] = warp_total" in text


def test_prims_vector_warp_prefix_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_warp_prefix.py"
    ).read_text(encoding="utf-8")

    assert "values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    _assert_no_payload_calls(text)
    assert "prefix_values = coop.exclusive_sum(warp, values_vec)" in text
    assert "warp_totals = coop.sum(warp, values_vec)" in text
    assert 'binary_op="min"' in text
    assert 'binary_op="max"' in text
    assert 'binary_op="bit_xor"' in text
    assert "striped_values = coop.exchange(" in text


def test_prims_vector_warp_merge_sort_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "prims_vector_warp_merge_sort.py"
    ).read_text(encoding="utf-8")

    assert "keys_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    assert "values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)" in text
    _assert_no_payload_calls(text)
    assert "coop.merge_sort_keys(\n            warp," in text
    assert "coop.merge_sort_pairs(\n            warp," in text
    assert "coop.store(\n            warp,\n            desc_keys_out," in text


def test_mixed_payload_sort_topk_example_uses_group_first_payloads():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "mixed_payload_sort_topk.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    assert "vector_keys = coop.ThreadData(items_per_thread, Int32)" in text
    _assert_no_payload_calls(text)
    assert "fragment_keys = cute.make_rmem_tensor(" in text
    assert "coop.radix_sort_keys(\n            block,\n            vector_keys," in text
    assert (
        "coop.radix_sort_keys(\n            block,\n            fragment_keys," in text
    )
    assert "coop.topk_max_keys(\n            block,\n            vector_keys," in text
    assert "temp_storage=vector_topk_temp_storage" in text
    assert "coop.store(block, sorted_fragment_keys_out, sorted_fragment_keys)" in text


def test_historical_factory_example_uses_public_group_routes_only():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "mixed_payload_factory_sort_topk.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    _assert_no_public_scoped_calls(text)
    assert "block = coop.this_block()" in text
    _assert_no_payload_calls(text)
    assert "coop.load(block, vector_keys_in, vector_keys)" in text
    assert "coop.load(block, tensor_keys_in, tensor_keys)" in text
    assert "coop.store(block, sorted_vector_keys_out, sorted_vector_keys)" in text
    assert "coop.store(block, sorted_tensor_keys_out, sorted_tensor_keys)" in text
    assert "temp_storage=vector_topk_temp_storage" in text
    assert "factory_scopes" in text


def test_mixed_tensor_vector_scan_example_uses_group_first_thread_data():
    text = (
        SOURCE_ROOT / "examples" / "cutlass" / "mixed_tensor_vector_scan.py"
    ).read_text(encoding="utf-8")

    _assert_uses_int32_runtime(text)
    assert "block = coop.this_block()" in text
    assert "tensor_values = coop.ThreadData(" in text
    assert "vector_values = coop.ThreadData(" in text
    assert "coop.load(\n            block,\n            tensor_values_in," in text
    assert "coop.load(\n            block,\n            vector_values_in," in text
    _assert_no_payload_calls(text)
    assert "coop.scan(\n            block,\n            tensor_values," in text
    assert "coop.scan(\n            block,\n            vector_values," in text
    assert "coop.store(\n            block,\n            tensor_prefix_out," in text
    assert "coop.store(\n            block,\n            vector_prefix_out," in text


def _direct_scoped_coop_import_offenders(path: Path, tree: ast.AST) -> list[str]:
    offenders = []
    scoped_modules = (
        "cuda.coop.cutlass.block",
        "cuda.coop.cutlass.warp",
    )
    scoped_prefixes = tuple(f"{name}." for name in scoped_modules)
    api_modules = ("cuda.coop.cutlass",)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in scoped_modules or alias.name.startswith(
                    scoped_prefixes
                ):
                    offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in scoped_modules or module.startswith(scoped_prefixes):
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")
            elif module in api_modules and any(
                alias.name in {"block", "warp", "*"} for alias in node.names
            ):
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    return offenders


def test_cutlass_example_import_guard_rejects_scoped_coop_imports():
    path = SOURCE_ROOT / "examples" / "cutlass" / "probe.py"
    rejected_snippets = (
        "import cuda.coop.cutlass.block\n",
        "import cuda.coop.cutlass.block as block\n",
        "import cuda.coop.cutlass.warp\n",
        "from cuda.coop.cutlass import block\n",
        "from cuda.coop.cutlass import warp\n",
        "from cuda.coop.cutlass import *\n",
        "from cuda.coop.cutlass.block import radix_sort_keys\n",
        "from cuda.coop.cutlass.warp import exclusive_sum\n",
    )
    allowed_snippets = (
        "import cuda.coop.cutlass as coop\n",
        "from cuda.coop.cutlass import Payload\n",
        "from cuda.coop.cutlass import ThreadData\n",
    )

    for snippet in rejected_snippets:
        tree = ast.parse(snippet)
        assert _direct_scoped_coop_import_offenders(path, tree), snippet

    for snippet in allowed_snippets:
        tree = ast.parse(snippet)
        assert _direct_scoped_coop_import_offenders(path, tree) == [], snippet


def _dotted_attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value_name = _dotted_attribute_name(node.value)
        return f"{value_name}.{node.attr}" if value_name is not None else None
    return None


def _scoped_thread_data_offenders(path: Path, tree: ast.AST) -> list[str]:
    offenders = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = _dotted_attribute_name(node)
        if name is not None and (
            name in {"block.ThreadData", "warp.ThreadData"}
            or name.endswith((".block.ThreadData", ".warp.ThreadData"))
        ):
            offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    return offenders


def test_cute_examples_use_cutlass_namespace_and_root_thread_data():
    assert not (SOURCE_ROOT / "examples" / "cute").exists()

    runtime_text = (SOURCE_ROOT / "examples" / "cutlass" / "_runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'ROOT_SCOPE = "cuda.coop.cutlass"' in runtime_text
    assert "coop_scope" not in runtime_text
    offenders = []

    for path in _cutlass_example_paths():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for import_offender in _direct_scoped_coop_import_offenders(path, tree):
            offenders.append(f"{import_offender}: direct scoped coop import")
        for thread_data_offender in _scoped_thread_data_offenders(path, tree):
            offenders.append(f"{thread_data_offender}: scoped ThreadData")

    assert offenders == []
