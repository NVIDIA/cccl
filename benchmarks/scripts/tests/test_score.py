# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Run with: `uvx --with numpy --with fpzip --with pandas --with pytest pytest -q benchmarks/scripts/tests/`

"""Tests for the workload weighting in `Bench.score`."""

import pytest
from cccl.bench.bench import Bench
from cccl.bench.config import RangePoint, VariantPoint
from cccl.bench.score import compute_axes_ids, compute_weight_matrices

ALGNAME = "cub.bench.reduce.sum"
SUBBENCH = "base"
CT_WORKLOAD = "T{ct}=I32 OffsetT{ct}=I64"
ELEMENTS = "Elements{io}"

# The runtime axis as declared by the benchmark, i.e. what `rt_axes_values()`
# reports. NVBench may skip some of these states at runtime, in which case they
# never reach `speedups`.
DECLARED = {SUBBENCH: {ELEMENTS: ["16", "20", "24", "28"]}}


class StubBench(Bench):
    """A `Bench` whose measurements are supplied instead of benchmarked.

    Everything up to `Bench.speedup()` needs a build directory, a GPU and a
    database; the weighting `Bench.score()` applies on top of the speedups does
    not, so it is tested against canned measurements.
    """

    def __init__(self, measured):
        super().__init__(
            ALGNAME, VariantPoint([RangePoint("TUNE_TPB", "tpb", 256)]), [CT_WORKLOAD]
        )
        self.measured = measured

    def speedup(self, ct_workload_point, rt_values, base_estimator, variant_estimator):
        return {
            SUBBENCH: {
                "{} {}={}".format(CT_WORKLOAD, ELEMENTS, value): s
                for value, s in self.measured.items()
            }
        }


def score_of(measured):
    return StubBench(measured).score([CT_WORKLOAD], DECLARED, None, None)


def declared_weight(value):
    axes_ids = compute_axes_ids(DECLARED)
    matrices = compute_weight_matrices(DECLARED, axes_ids)
    return matrices[SUBBENCH][DECLARED[SUBBENCH][ELEMENTS].index(value)]


def test_score_is_the_weighted_mean_of_the_speedups():
    measured = {"16": 0.5, "20": 1.0, "24": 1.5, "28": 2.0}

    assert score_of(measured) == pytest.approx(
        sum(declared_weight(v) * s for v, s in measured.items())
    )


def test_score_of_a_failed_run_is_minus_infinity():
    class FailedBench(StubBench):
        def speedup(
            self, ct_workload_point, rt_values, base_estimator, variant_estimator
        ):
            return {}

    assert FailedBench({}).score([CT_WORKLOAD], DECLARED, None, None) == float("-inf")


@pytest.mark.parametrize(
    "measured", [["16", "20", "24", "28"], ["24", "28"], ["16", "28"], ["28"]]
)
def test_a_variant_on_par_with_base_scores_one_whatever_was_skipped(measured):
    """Skipped states must not take their weight mass into the score.

    If it stayed in, a variant that is exactly as fast as base would score below
    1.0 and look like a regression.
    """
    assert score_of({value: 1.0 for value in measured}) == pytest.approx(1.0)


def test_relative_importance_does_not_depend_on_what_was_skipped():
    """The weight ratio of two workloads is fixed by the declared axis.

    Importance is assigned by position on the axis (see `score.compute_weights`),
    so deriving the weights from the states that ran would move the smallest
    surviving value onto the least importance anchor -- 2^24 would stop being
    nearly as important as 2^28 just because 2^16 was skipped.
    """
    measured = ["24", "28"]
    total = sum(declared_weight(v) for v in measured)

    for value in measured:
        score = score_of({v: (2.0 if v == value else 1.0) for v in measured})

        # Doubling the speedup of one workload raises the score by that
        # workload's share of the declared weight of the measured set.
        assert score - 1.0 == pytest.approx(declared_weight(value) / total)


def test_a_uniformly_faster_variant_beats_one_that_trades_a_regression():
    """2^24 and 2^28 carry almost the same weight in the declared axis.

    Trading a 10% regression on 2^24 for a 20% gain on 2^28 is therefore not
    worth it. Rebuilding the weights from the measured states would demote 2^24
    to the least importance anchor and pick the regressing variant instead.
    """
    regressing = score_of({"24": 0.90, "28": 1.20})
    uniform = score_of({"24": 1.05, "28": 1.08})

    assert uniform > regressing
