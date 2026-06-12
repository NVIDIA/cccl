# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

import cuda.compute as cc

from host_benchmark_cases import (
    CASES,
    HostBenchmarkCase,
    patch_wrapper_to_skip_native_compute,
    synchronize,
)

pytest.importorskip("pytest_benchmark")

BUILD_TIME_ROUNDS = 10
ONESHOT_ROUNDS = 20
ONESHOT_ITERATIONS = 100
TWOSHOT_ROUNDS = 20
TWOSHOT_ITERATIONS = 1000


def _case_params() -> list[pytest.ParameterSet]:
    params = []
    for case in CASES:
        marks = []
        if case.skip_reason is not None:
            marks.append(pytest.mark.skip(reason=case.skip_reason))
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


@pytest.mark.benchmark(group="cuda.compute.host.build_time")
@pytest.mark.parametrize("case", _case_params())
def test_build_time(benchmark, case: HostBenchmarkCase):
    state = case.setup()
    synchronize()

    def setup() -> None:
        cc.clear_all_caches()

    def build():
        return case.make_wrapper(state)

    benchmark.pedantic(
        build,
        setup=setup,
        rounds=BUILD_TIME_ROUNDS,
        iterations=1,
        warmup_rounds=0,
    )


@pytest.mark.benchmark(group="cuda.compute.host.oneshot_cached")
@pytest.mark.parametrize("case", _case_params())
def test_oneshot_cached_host_overhead(benchmark, case: HostBenchmarkCase):
    cc.clear_all_caches()
    state = case.setup()
    wrapper = case.make_wrapper(state)
    patch_wrapper_to_skip_native_compute(wrapper, case.noop_return_kind)
    synchronize()

    def call() -> None:
        case.oneshot(state)

    benchmark.pedantic(
        call,
        rounds=ONESHOT_ROUNDS,
        iterations=ONESHOT_ITERATIONS,
        warmup_rounds=0,
    )


@pytest.mark.benchmark(group="cuda.compute.host.twoshot_call")
@pytest.mark.parametrize("case", _case_params())
def test_twoshot_call_host_overhead(benchmark, case: HostBenchmarkCase):
    cc.clear_all_caches()
    state = case.setup()
    wrapper = case.make_wrapper(state)
    patch_wrapper_to_skip_native_compute(wrapper, case.noop_return_kind)
    synchronize()

    def call() -> None:
        case.twoshot(state, wrapper)

    benchmark.pedantic(
        call,
        rounds=TWOSHOT_ROUNDS,
        iterations=TWOSHOT_ITERATIONS,
        warmup_rounds=0,
    )
