# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end pytest/xdist checks for earned conformance evidence."""

from __future__ import annotations

import os
import subprocess
import sys

from ...support.paths import PACKAGE_ROOT, REPO_ROOT

_FIXTURE = "tests/contracts/core/evidence_outcome_fixture.py"
_COLLECTION_SKIP_FIXTURE = (
    "tests/contracts/core/evidence_collection_skip_fixture.py::test_never_collected"
)
_LOAD_STORE_SEMANTICS = "tests/contracts/core/test_common_group_memory_semantics.py"
_SUM_SCAN_SEMANTICS = "tests/contracts/core/test_common_group_sum_scan_semantics.py"
_EXCHANGE_SEMANTICS = "tests/contracts/core/test_common_group_exchange_semantics.py"
_ADJACENT_DISCONTINUITY_SEMANTICS = (
    "tests/contracts/core/test_common_root_adjacent_discontinuity.py"
)
_SHUFFLE_SEMANTICS = "tests/contracts/core/test_common_group_shuffle_semantics.py"
_HISTOGRAM_SEMANTICS = "tests/contracts/core/test_common_group_histogram_semantics.py"
_RUN_LENGTH_DECODE_SEMANTICS = (
    "tests/contracts/core/test_common_group_run_length_decode_semantics.py"
)
_MERGE_SORT_SEMANTICS = "tests/contracts/core/test_common_group_merge_sort_semantics.py"
_RADIX_RANK_SEMANTICS = "tests/contracts/core/test_common_group_radix_rank_semantics.py"
_RADIX_SORT_SEMANTICS = "tests/contracts/core/test_common_group_radix_sort_semantics.py"
_TOPK_SEMANTICS = "tests/contracts/core/test_common_group_topk_semantics.py"
_UNMARKED_EXACT_NODE = (
    "tests/contracts/parity/test_coverage_manifest.py"
    "::test_only_fully_passing_tests_earn_conformance_evidence"
)


def _run_pytest(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (os.fspath(PACKAGE_ROOT), python_path) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        check=False,
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def _run_pytest_from_repo_root(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (os.fspath(PACKAGE_ROOT), python_path) if part
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            "python/cuda_coop/pyproject.toml",
            *arguments,
        ],
        check=False,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def _assert_load_store_evidence_failure(
    completed: subprocess.CompletedProcess[str],
) -> None:
    _assert_required_core_semantics_failure(completed, ("load", "store"))


def _assert_required_core_semantics_failure(
    completed: subprocess.CompletedProcess[str],
    operations: tuple[str, ...],
) -> None:
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    for operation in operations:
        assert (
            output.count(
                f"missing 'semantics' conformance evidence for 'group.{operation}'/core"
            )
            == 1
        )
    assert "INTERNALERROR" not in output
    assert "PluggyTeardownRaisedWarning" not in output


def _assert_harness_evidence_failure(
    completed: subprocess.CompletedProcess[str],
) -> None:
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert (
        output.count(
            "missing 'semantics' conformance evidence for "
            "'contracts.evidence_outcomes'/core"
        )
        == 1
    )
    assert "outside its declared collection lanes" not in output
    assert "malformed cuda.coop expected-evidence inventory" not in output
    assert "INTERNALERROR" not in output
    assert "PluggyTeardownRaisedWarning" not in output


def test_xdist_transfers_passed_collection_for_an_all_deselected_lane() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _LOAD_STORE_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_load_store_evidence_failure(completed)


def test_xdist_enforces_every_promoted_reduce_sum_scan_semantics_cell() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _SUM_SCAN_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(
        completed,
        (
            "reduce",
            "sum",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        ),
    )


def test_repo_root_invocation_activates_package_relative_evidence_lane() -> None:
    completed = _run_pytest_from_repo_root(
        "-n",
        "2",
        "python/cuda_coop/tests/contracts/core/test_common_group_memory_semantics.py",
        "-k",
        "definitely_no_such_test",
    )

    _assert_load_store_evidence_failure(completed)


def test_xdist_enforces_promoted_merge_sort_semantics_cell() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _MERGE_SORT_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(completed, ("merge_sort_keys",))


def test_xdist_enforces_promoted_radix_sort_semantics_cell() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _RADIX_SORT_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(completed, ("radix_sort_keys",))


def test_xdist_enforces_promoted_radix_rank_semantics_cell() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _RADIX_RANK_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(completed, ("radix_rank",))


def test_xdist_enforces_promoted_run_length_decode_semantics_cell() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _RUN_LENGTH_DECODE_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(completed, ("run_length_decode",))


def test_xdist_enforces_promoted_topk_semantics_cells() -> None:
    completed = _run_pytest(
        "-n",
        "2",
        _TOPK_SEMANTICS,
        "-k",
        "definitely_no_such_test",
    )

    _assert_required_core_semantics_failure(
        completed,
        ("topk_max_keys", "topk_min_keys"),
    )


def test_ignore_keeps_an_exact_required_lane_inactive() -> None:
    completed = _run_pytest(
        "tests/contracts/core",
        "--ignore",
        _LOAD_STORE_SEMANTICS,
        "--ignore",
        _SUM_SCAN_SEMANTICS,
        "--ignore",
        _EXCHANGE_SEMANTICS,
        "--ignore",
        _ADJACENT_DISCONTINUITY_SEMANTICS,
        "--ignore",
        _SHUFFLE_SEMANTICS,
        "--ignore",
        _HISTOGRAM_SEMANTICS,
        "--ignore",
        _RUN_LENGTH_DECODE_SEMANTICS,
        "--ignore",
        _MERGE_SORT_SEMANTICS,
        "--ignore",
        _RADIX_SORT_SEMANTICS,
        "--ignore",
        _RADIX_RANK_SEMANTICS,
        "--ignore",
        _TOPK_SEMANTICS,
        "-k",
        "test_argument_binding_classifies_omitted_static_and_runtime_values",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_xdist_exact_unmarked_node_does_not_activate_sibling_claims() -> None:
    completed = _run_pytest("-n", "2", _UNMARKED_EXACT_NODE)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_xdist_exact_skipped_evidence_cannot_satisfy_its_cell() -> None:
    completed = _run_pytest("-n", "2", f"{_FIXTURE}::test_skipped_evidence")

    _assert_harness_evidence_failure(completed)


def test_xdist_exact_collection_skip_uses_the_lane_fallback() -> None:
    completed = _run_pytest("-n", "2", _COLLECTION_SKIP_FIXTURE)

    _assert_harness_evidence_failure(completed)


def test_xdist_requires_every_selected_parameter_case_to_pass() -> None:
    completed = _run_pytest("-n", "2", f"{_FIXTURE}::test_parameterized_evidence")

    _assert_harness_evidence_failure(completed)
