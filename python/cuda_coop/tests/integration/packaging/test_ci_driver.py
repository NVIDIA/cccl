# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

from ...support.paths import REPO_ROOT

_NUMBA_MLIR_PLANNER_API = (
    (
        "numba_cuda_mlir.extending",
        (
            "WholeFunctionPlanner",
            "register_planner",
            "require_launch_config",
        ),
    ),
)
_CUTLASS_ROOT_API = (("cuda.coop", ("ThreadData", "this_block", "reduce")),)


@pytest.fixture
def ci_driver(monkeypatch):
    path = REPO_ROOT / "ci" / "util" / "python" / "cuda_coop_test_driver.py"
    spec = importlib.util.spec_from_file_location("cuda_coop_test_driver_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _stage(ci_driver, required_public_symbols=_NUMBA_MLIR_PLANNER_API):
    return ci_driver.Stage(
        modules=(),
        distributions=(),
        batches=(),
        required_public_symbols=required_public_symbols,
    )


def test_all_numba_mlir_stages_require_group_planner_public_api(ci_driver):
    for stage_name in (
        "numba-mlir",
        "numba-mlir-host",
        "numba-mlir-qualification",
        "numba-mlir-cluster-qualification",
    ):
        assert (
            ci_driver.STAGES[stage_name].required_public_symbols
            == _NUMBA_MLIR_PLANNER_API
        )


def test_cutlass_stages_require_common_root_api(ci_driver):
    for stage_name in (
        "contracts",
        "cutlass-host",
        "cutlass-final-link-qualification",
        "cutlass-cluster-qualification",
        "cutlass-sm100-qualification",
    ):
        assert ci_driver.STAGES[stage_name].required_public_symbols == _CUTLASS_ROOT_API

    assert ci_driver.STAGES["cutlass"].required_public_symbols == (
        *_CUTLASS_ROOT_API,
        *_NUMBA_MLIR_PLANNER_API,
    )


def test_cutlass_stages_run_examples_against_the_installed_wheel(
    ci_driver,
    monkeypatch,
):
    expected_stages = {
        "cutlass",
        "cutlass-host",
        "cutlass-final-link-qualification",
        "cutlass-cluster-qualification",
        "cutlass-sm100-qualification",
    }
    assert ci_driver._CUTLASS_EXAMPLE_INSTALLED_STAGES == expected_stages
    for stage_name in expected_stages - {"cutlass"}:
        assert ci_driver.STAGES[stage_name].modules == (
            "cuda.coop",
            "cuda.coop.cutlass",
            "cutlass",
            "torch",
        )
        assert ci_driver.STAGES[stage_name].distributions == (
            "cuda-coop",
            "nvidia-cutlass-dsl",
            "torch",
        )

    assert ci_driver.STAGES["cutlass"].modules == (
        "cuda.coop",
        "cuda.coop.cutlass",
        "cutlass",
        "torch",
        "cuda.coop.numba_mlir",
        "numba_cuda_mlir",
        "numba_cuda_mlir.cuda",
    )
    assert ci_driver.STAGES["cutlass"].distributions == (
        "cuda-coop",
        "nvidia-cutlass-dsl",
        "torch",
        "numba-cuda-mlir",
    )

    monkeypatch.setenv(ci_driver._CUTLASS_EXAMPLES_INSTALLED_ENV, "0")

    for stage_name in expected_stages:
        environment = ci_driver._batch_environment(stage_name)
        assert environment[ci_driver._CUTLASS_EXAMPLES_INSTALLED_ENV] == "1"

    monkeypatch.setenv(ci_driver._CUTLASS_EXAMPLES_INSTALLED_ENV, "1")
    for stage_name in ("contracts", "cutlass-extra"):
        assert (
            ci_driver._CUTLASS_EXAMPLES_INSTALLED_ENV
            not in ci_driver._batch_environment(stage_name)
        )


def test_cutlass_batch_collection_and_execution_use_installed_wheel(
    ci_driver,
    monkeypatch,
    tmp_path,
):
    selector = tmp_path / "test_example.py"
    selector.touch()
    config = tmp_path / "pyproject.toml"
    config.touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("capture_output"):
            return SimpleNamespace(
                returncode=0,
                stdout="test_example.py: 1\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ci_driver.subprocess, "run", run)
    batch = ci_driver.Batch("examples", (), workers=0)

    ci_driver._run_batch(
        tmp_path,
        config,
        "cutlass",
        batch,
        [selector],
        dry_run=False,
    )

    assert len(calls) == 2
    for _, kwargs in calls:
        assert kwargs["env"][ci_driver._CUTLASS_EXAMPLES_INSTALLED_ENV] == "1"


def test_cutlass_stage_gates_mixed_backend_activation(ci_driver):
    batch = next(
        batch
        for batch in ci_driver.STAGES["cutlass"].batches
        if batch.name == "mixed-activation"
    )

    assert batch.selectors == ("python/cuda_coop/tests/integration/compiler",)
    assert batch.workers == 0
    assert batch.needs_gpu
    assert batch.forbid_skips


def test_compiler_qualification_stages_fail_closed(ci_driver):
    for stage_name in (
        "numba-mlir-qualification",
        "numba-mlir-cluster-qualification",
        "cutlass-final-link-qualification",
        "cutlass-cluster-qualification",
    ):
        stage = ci_driver.STAGES[stage_name]
        assert stage.batches
        assert all(batch.needs_gpu for batch in stage.batches)
        assert all(batch.forbid_skips for batch in stage.batches)

    numba_final_link = ci_driver.STAGES["numba-mlir-qualification"].batches[0]
    assert numba_final_link.selectors == (
        "python/cuda_coop/tests/providers/qualification/numba_mlir",
    )

    cutlass_final_link = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "final-link"
    )
    assert cutlass_final_link.selectors == (
        "python/cuda_coop/tests/providers/cutlass/test_ltoir_inlining.py",
    )
    assert cutlass_final_link.selection_args == ("-m", "not requires_sm100")
    cutlass_load_store = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "load-store-final-link"
    )
    assert cutlass_load_store.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_load_store_final_link.py",
    )
    assert cutlass_load_store.selection_args == ()
    cutlass_reduce_sum_scan = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "reduce-sum-scan-final-link"
    )
    assert cutlass_reduce_sum_scan.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_sum_scan_final_link.py",
    )
    assert cutlass_reduce_sum_scan.selection_args == ()
    cutlass_exchange = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "exchange-final-link"
    )
    assert cutlass_exchange.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_exchange_final_link.py",
    )
    assert cutlass_exchange.selection_args == ()
    cutlass_adjacent_discontinuity = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "adjacent-discontinuity-final-link"
    )
    assert cutlass_adjacent_discontinuity.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_adjacent_discontinuity_final_link.py",
    )
    assert cutlass_adjacent_discontinuity.selection_args == ()
    cutlass_shuffle = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "shuffle-final-link"
    )
    assert cutlass_shuffle.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_shuffle_final_link.py",
    )
    assert cutlass_shuffle.selection_args == ()
    cutlass_histogram = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "histogram-final-link"
    )
    assert cutlass_histogram.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_histogram_final_link.py",
    )
    assert cutlass_histogram.selection_args == ()
    cutlass_run_length_decode = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "run-length-decode-final-link"
    )
    assert cutlass_run_length_decode.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_run_length_decode_final_link.py",
    )
    assert cutlass_run_length_decode.selection_args == ()
    cutlass_merge_sort = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "merge-sort-final-link"
    )
    assert cutlass_merge_sort.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_merge_sort_final_link.py",
    )
    assert cutlass_merge_sort.selection_args == ()
    cutlass_radix_sort = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "radix-sort-final-link"
    )
    assert cutlass_radix_sort.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_radix_sort_final_link.py",
    )
    assert cutlass_radix_sort.selection_args == ()
    cutlass_radix_rank = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "radix-rank-final-link"
    )
    assert cutlass_radix_rank.selectors == (
        "python/cuda_coop/tests/providers/cutlass/"
        "test_common_root_radix_rank_final_link.py",
    )
    assert cutlass_radix_rank.selection_args == ()
    cutlass_topk = next(
        batch
        for batch in ci_driver.STAGES["cutlass-final-link-qualification"].batches
        if batch.name == "topk-final-link"
    )
    assert cutlass_topk.selectors == (
        "python/cuda_coop/tests/providers/cutlass/test_common_root_topk_final_link.py",
    )
    assert cutlass_topk.selection_args == ()

    sm100_selectors = {
        selector
        for batch in ci_driver.STAGES["cutlass-sm100-qualification"].batches
        for selector in batch.selectors
    }
    assert cutlass_load_store.selectors[0] not in sm100_selectors
    assert cutlass_reduce_sum_scan.selectors[0] not in sm100_selectors
    assert cutlass_exchange.selectors[0] not in sm100_selectors
    assert cutlass_adjacent_discontinuity.selectors[0] not in sm100_selectors
    assert cutlass_shuffle.selectors[0] not in sm100_selectors
    assert cutlass_histogram.selectors[0] not in sm100_selectors
    assert cutlass_run_length_decode.selectors[0] not in sm100_selectors
    assert cutlass_merge_sort.selectors[0] not in sm100_selectors
    assert cutlass_radix_sort.selectors[0] not in sm100_selectors
    assert cutlass_radix_rank.selectors[0] not in sm100_selectors
    assert cutlass_topk.selectors[0] not in sm100_selectors

    runtime_evidence_paths = {
        "cutlass": (
            "python/cuda_coop/tests/backends/cutlass/runtime/"
            "test_common_root_load_store.py"
        ),
        "numba_mlir": (
            "python/cuda_coop/tests/backends/numba_mlir/runtime/"
            "test_common_root_load_store.py"
        ),
    }
    for backend, stage_name in (
        ("cutlass", "cutlass-cluster-qualification"),
        ("numba_mlir", "numba-mlir-cluster-qualification"),
    ):
        cluster_selectors = {
            selector
            for batch in ci_driver.STAGES[stage_name].batches
            for selector in batch.selectors
        }
        assert runtime_evidence_paths[backend] not in cluster_selectors


def test_numba_host_stages_do_not_collect_gpu_compile_evidence(ci_driver):
    for stage_name in ("numba-mlir", "numba-mlir-host"):
        host = next(
            batch
            for batch in ci_driver.STAGES[stage_name].batches
            if batch.name == "host"
        )
        assert host.selectors == ("python/cuda_coop/tests/backends/numba_mlir/unit",)
        assert host.selection_args == ()

        diagnostics = next(
            batch
            for batch in ci_driver.STAGES[stage_name].batches
            if batch.name == "host-compile-diagnostics"
        )
        assert diagnostics.selectors == ci_driver._NUMBA_MLIR_HOST_COMPILE_DIAGNOSTICS
        assert diagnostics.selection_args == ()
        assert all(
            "test_common_root_load_store.py" not in selector
            for selector in diagnostics.selectors
        )

    gpu_compile = next(
        batch
        for batch in ci_driver.STAGES["numba-mlir"].batches
        if batch.name == "gpu-compile-diagnostics"
    )
    assert gpu_compile.selectors == (
        "python/cuda_coop/tests/backends/numba_mlir/compile",
    )
    assert gpu_compile.selection_args == ("-m", "gpu")


def test_provenance_rejects_missing_public_planner_symbol(
    ci_driver, monkeypatch, tmp_path
):
    public_api = SimpleNamespace(
        WholeFunctionPlanner=object,
        register_planner=lambda planner: planner,
    )
    monkeypatch.setattr(
        ci_driver.importlib,
        "import_module",
        lambda module_name: public_api,
    )

    with pytest.raises(RuntimeError) as exc_info:
        ci_driver._print_provenance(tmp_path, _stage(ci_driver))

    assert str(exc_info.value) == (
        "required public API module 'numba_cuda_mlir.extending' "
        "is missing symbols: require_launch_config"
    )


def test_provenance_accepts_complete_public_planner_contract(
    ci_driver, monkeypatch, tmp_path, capsys
):
    public_api = SimpleNamespace(
        WholeFunctionPlanner=object,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: state,
    )
    monkeypatch.setattr(
        ci_driver.importlib,
        "import_module",
        lambda module_name: public_api,
    )

    ci_driver._print_provenance(tmp_path, _stage(ci_driver))

    assert (
        "public API numba_cuda_mlir.extending: "
        "WholeFunctionPlanner, register_planner, require_launch_config"
    ) in capsys.readouterr().out


def test_provenance_reports_public_module_import_failure(
    ci_driver, monkeypatch, tmp_path
):
    def fail_import(module_name):
        raise ImportError(f"cannot import {module_name}")

    monkeypatch.setattr(ci_driver.importlib, "import_module", fail_import)
    stage = _stage(
        ci_driver,
        (("numba_cuda_mlir.extending", ("WholeFunctionPlanner",)),),
    )

    with pytest.raises(RuntimeError) as exc_info:
        ci_driver._print_provenance(tmp_path, stage)

    assert str(exc_info.value) == (
        "required public API module 'numba_cuda_mlir.extending' "
        "could not be imported: cannot import numba_cuda_mlir.extending"
    )
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_ci_driver_rejects_legacy_cuda_cccl(ci_driver, monkeypatch):
    monkeypatch.setattr(
        ci_driver.importlib.metadata,
        "version",
        lambda distribution: "3.5.0.dev846",
    )

    with pytest.raises(RuntimeError) as exc_info:
        ci_driver._reject_legacy_cuda_cccl()

    assert str(exc_info.value) == (
        "cuda-coop CI requires an independent-wheel environment, but found "
        "cuda-cccl==3.5.0.dev846"
    )


def test_ci_driver_accepts_environment_without_cuda_cccl(ci_driver, monkeypatch):
    def missing(distribution):
        raise ci_driver.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(ci_driver.importlib.metadata, "version", missing)

    ci_driver._reject_legacy_cuda_cccl()


def test_ci_wheel_producer_builds_one_cuda_coop_artifact():
    matrix = (REPO_ROOT / "ci" / "matrix.yaml").read_text(encoding="utf-8")
    producer = (REPO_ROOT / "ci" / "build_cuda_coop_python.sh").read_text(
        encoding="utf-8"
    )
    consumer = (REPO_ROOT / "ci" / "test_cuda_coop_python.sh").read_text(
        encoding="utf-8"
    )

    assert "needs: 'build_py_coop_wheel'" in matrix
    assert "needs: 'build_py_wheel'" not in "\n".join(
        line for line in matrix.splitlines() if "test_py_coop_" in line
    )
    assert '"$repo_root/python/cuda_coop"' in producer
    assert 'get_cuda_coop_wheel_artifact_name.sh" wheel' in producer
    assert "get_producer_id.sh" in consumer
    assert "build_cuda_cccl" not in producer
    assert "build_cuda_cccl" not in consumer
    assert 'add_coop_extra "cu13"' in consumer
    assert 'add_coop_extra "cu${cuda_major_version}"' in consumer
    assert "needs_numba_mlir_runtime=true" in consumer
    assert consumer.count("needs_numba_mlir_runtime=true") >= 2
    assert "CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE" in consumer
    assert "CUDA_COOP_CUTLASS_REQUIREMENTS_FILE" in consumer
    assert consumer.count("is not a readable file") == 2
    assert "ci/requirements/cuda-coop-cutlass.txt" not in consumer
    assert 'add_coop_extra "minimal-cu' not in consumer
    assert 'add_coop_extra "test-numba-mlir-cu' not in consumer
    assert 'add_coop_extra "test-numba-cuda-mlir-cu' not in consumer

    for project in ("contracts", "numba_mlir", "cutlass"):
        for prefix in ("build_cuda_coop", "test_cuda_coop"):
            wrapper = REPO_ROOT / "ci" / f"{prefix}_cuda_coop_{project}.sh"
            assert wrapper.is_file()
            assert os.access(wrapper, os.X_OK)


def test_cutlass_qualification_rejects_the_incompatible_legacy_lock():
    assert not (REPO_ROOT / "ci" / "requirements" / "cuda-coop-cutlass.txt").exists()
