# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Run with: `uvx --with numpy --with fpzip --with matplotlib --with pandas --with scipy --with pytest pytest -q benchmarks/scripts/tests/`

"""Tests for the axis space a score is weighed over, and the weighting itself."""

import cccl.bench.bench as bench_module
import pytest
from cccl.bench.bench import Bench
from cccl.bench.config import RangePoint, VariantPoint
from cccl.bench.score import compute_axes_ids, compute_weight_matrices

ALGNAME = "cub.bench.reduce.sum"
SUBBENCH = "base"
CT_WORKLOAD = "T{ct}=I32 OffsetT{ct}=I64"
ELEMENTS = "Elements{io}"

# The runtime axis as declared by the benchmark, i.e. what
# `declared_rt_axes_values()` reports. A campaign may narrow it with `-a`, and
# NVBench may skip some of the states it does ask for; neither reaches
# `speedups`, and neither changes the weights.
DECLARED = {SUBBENCH: {ELEMENTS: ["16", "20", "24", "28"]}}

# What `--jsonlist-benches` would report for a benchmark declaring `DECLARED`.
BENCHES = {
    "benchmarks": [
        {
            "name": SUBBENCH,
            "axes": [
                {"name": "T{ct}", "flags": "", "values": [{"input_string": "I32"}]},
                {
                    "name": "OffsetT{ct}",
                    "flags": "",
                    "values": [{"input_string": "I64"}],
                },
                {
                    "name": ELEMENTS,
                    "flags": "",
                    "values": [
                        {"input_string": value}
                        for value in DECLARED[SUBBENCH][ELEMENTS]
                    ],
                },
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def declared_benchmark(monkeypatch):
    """Let `Bench` read its axis space the way it does from a real binary.

    Stubbing `declared_rt_axes_values` instead would take the code under test
    out of the test: what `score` weighs over moving off the caller-supplied
    `rt_values` is the whole point.
    """
    monkeypatch.setattr(bench_module, "json_benches", lambda algname: BENCHES)


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


@pytest.mark.parametrize(
    "asked_for", [["16", "20", "24", "28"], ["20", "24", "28"], ["24", "28"], ["28"]]
)
def test_narrowing_the_campaign_does_not_change_the_weights(asked_for):
    """`-a` narrows what runs, not what a speedup is worth.

    Weighing over what was asked for would move the smallest value asked for
    onto the least importance anchor: a campaign narrowed to 2^24 and 2^28
    would split them 38/62 instead of the 50/50 their positions on the declared
    axis carry.
    """
    measured = {"24": 1.0, "28": 2.0}

    narrowed = StubBench(measured).score(
        [CT_WORKLOAD], {SUBBENCH: {ELEMENTS: asked_for}}, None, None
    )

    assert narrowed == pytest.approx(score_of(measured))


def test_a_uniformly_faster_variant_beats_one_that_trades_a_regression():
    """2^24 and 2^28 carry almost the same weight in the declared axis.

    Trading a 10% regression on 2^24 for a 20% gain on 2^28 is therefore not
    worth it. Rebuilding the weights from the measured states would demote 2^24
    to the least importance anchor and pick the regressing variant instead.
    """
    regressing = score_of({"24": 0.90, "28": 1.20})
    uniform = score_of({"24": 1.05, "28": 1.08})

    assert uniform > regressing


def test_a_narrowing_naming_an_undeclared_value_is_rejected():
    """`-a` replaces a value axis in NVBench rather than narrowing it.

    An undeclared value would run, be stored, and only then have no position on
    the axis to be weighed by. Quietly dropping it would hide the typo and
    silently tune something other than what was asked for, so it is an error,
    raised before anything is built.
    """
    with pytest.raises(Exception, match=r"does not declare 22 on axis Elements"):
        StubBench({}).axes_values({ELEMENTS: ["20", "22", "28"]}, False)


def test_a_narrowing_of_declared_values_is_kept_verbatim():
    narrowed = StubBench({}).axes_values({ELEMENTS: ["20", "28"]}, False)

    assert narrowed == {SUBBENCH: {ELEMENTS: ["20", "28"]}}


def test_an_axis_the_narrowing_does_not_name_keeps_its_declared_values():
    narrowed = StubBench({}).axes_values({}, False)

    assert narrowed == DECLARED


def subbench_stub(monkeypatch, subbenches):
    """`--jsonlist-benches` for a benchmark file declaring several subbenches."""
    monkeypatch.setattr(
        bench_module,
        "json_benches",
        lambda algname: {
            "benchmarks": [
                {
                    "name": name,
                    "axes": [
                        {
                            "name": axis,
                            "flags": "",
                            "values": [{"input_string": v} for v in values],
                        }
                        for axis, values in axes.items()
                    ],
                }
                for name, axes in subbenches.items()
            ]
        },
    )


def test_an_axis_split_across_subbenches_can_only_be_narrowed_to_shared_values(
    monkeypatch,
):
    """`cub.bench.segmented_reduce` splits `SegmentSize` three ways.

    NVBench would hand a value to a subbench that does not declare it and run
    it there anyway, so a narrowing only one of them covers has to be rejected,
    naming the subbench that cannot take it.
    """
    subbench_stub(
        monkeypatch,
        {
            "small": {"SegmentSize": ["1", "16"]},
            "large": {"SegmentSize": ["512", "65536"]},
        },
    )

    with pytest.raises(Exception, match=r"\.large does not declare 16 on axis"):
        StubBench({}).axes_values({"SegmentSize": ["16"]}, False)

    assert StubBench({}).axes_values({}, False) == {
        "small": {"SegmentSize": ["1", "16"]},
        "large": {"SegmentSize": ["512", "65536"]},
    }


def test_an_axis_a_subbench_does_not_have_is_left_alone(monkeypatch):
    """`segmented_sort.keys` gives `Entropy` to `power` and to neither sibling.

    One `-a` is meant to be usable across a whole `-R` selection, so an axis a
    benchmark does not have is ignored rather than rejected.
    """
    subbench_stub(
        monkeypatch,
        {
            "power": {ELEMENTS: ["22", "26"], "Entropy": ["1.000", "0.201"]},
            "small": {ELEMENTS: ["22", "26"]},
        },
    )

    narrowed = StubBench({}).axes_values({"Entropy": ["1.000"]}, False)

    assert narrowed == {
        "power": {ELEMENTS: ["22", "26"], "Entropy": ["1.000"]},
        "small": {ELEMENTS: ["22", "26"]},
    }
