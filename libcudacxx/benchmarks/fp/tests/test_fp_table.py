# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Run with: `uvx --with pytest pytest -q libcudacxx/benchmarks/fp/tests`

"""Tests of the table that fp_table.py joins from the fpmp benchmark CSVs: the merge of the
throughput, latency and reference rows of a cell, the gain columns against the reference
precision, the formatting and the column order. The SASS count needs cuobjdump and a built
executable, so it is replaced by a fixed mapping here."""

import csv

import fp_table
import pytest

HEADER = [
    "Benchmark",
    "Device Name",
    "Func{ct}",
    "Type{ct}",
    "Accuracy{ct}",
    "Skipped",
    "u2",
    "Bits",
    "Check",
    "Worst at",
    "Regs",
    "ev/clk/SM",
    "clk/ev",
]

ROWS = [
    # one fp32mp2 cell with its throughput and latency rows
    [
        "throughput",
        "GPU",
        "div",
        "fp32mp2",
        "mid",
        "No",
        "6.0622",
        "45",
        "PASS",
        "0x1p+0",
        "16",
        "6.0",
        "",
    ],
    ["latency", "GPU", "div", "fp32mp2", "mid", "No", "", "", "", "", "", "", "50.0"],
    # an fp64mp2 cell: no u2, no reference precision yet
    [
        "throughput",
        "GPU",
        "div",
        "fp64mp2",
        "mid",
        "No",
        "-",
        "51",
        "PASS",
        "0x1p+0",
        "26",
        "0.15",
        "",
    ],
    ["latency", "GPU", "div", "fp64mp2", "mid", "No", "", "", "", "", "", "", "600.0"],
    # the reference precision of fp32mp2: no accuracy axis, no accuracy columns
    ["ref_throughput", "GPU", "div", "fp64", "", "No", "", "", "", "", "32", "0.1", ""],
    ["ref_latency", "GPU", "div", "fp64", "", "No", "", "", "", "", "", "", "500.0"],
    # a skipped row is left out
    ["throughput", "GPU", "add", "fp32mp2", "low", "Yes", "", "", "", "", "", "", ""],
]

SASS = {
    ("div", "fp32mp2", "mid"): 15,
    ("div", "fp64mp2", "mid"): 30,
    ("div", "fp64", "-"): 21,
}


@pytest.fixture
def rows(tmp_path, monkeypatch):
    path = tmp_path / "arith.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(ROWS)
    monkeypatch.setattr(fp_table, "sass_counts", lambda exe: SASS)
    merged = {}
    fp_table.merge("bin/libcudacxx.bench.fp.fpmp.arith.base", str(path), merged)
    fp_table.add_gains(merged)
    return merged


def test_merge_joins_the_rows_of_a_cell_and_drops_the_ct_suffix(rows):
    cell = rows[("div", "fp32mp2", "mid")]
    assert cell["Family"] == "arith"
    assert cell["u2"] == "6.0622"
    assert cell["Bits"] == "45"
    assert cell["Check"] == "PASS"
    assert cell["Worst at"] == "0x1p+0"
    assert cell["Regs"] == "16"
    assert cell["ev/clk/SM"] == "6.0"
    assert cell["clk/ev"] == "50.0"
    assert cell["SASS"] == 15
    assert ("add", "fp32mp2", "low") not in rows


def test_reference_rows_have_no_accuracy(rows):
    ref = rows[("div", "fp64", "-")]
    assert ref["Accuracy"] == "-"
    assert ref["u2"] == ref["Bits"] == ref["Check"] == ref["Worst at"] == "-"
    assert ref["ev/clk/SM"] == "0.1"
    assert ref["clk/ev"] == "500.0"
    assert ref["SASS"] == 21
    assert ref["Gain tp"] == ref["Gain lat"] == "-"


def test_gains_are_against_the_reference_precision(rows):
    cell = rows[("div", "fp32mp2", "mid")]
    assert cell["Gain tp"] == pytest.approx(6.0 / 0.1)
    assert cell["Gain lat"] == pytest.approx(500.0 / 50.0)
    # fp64mp2 waits for a binary128 reference
    assert rows[("div", "fp64mp2", "mid")]["Gain tp"] == "-"


def test_fmt_keeps_four_digits_for_the_timings_and_three_for_the_gains():
    assert fp_table.fmt("ev/clk/SM", "0.063281") == "0.06328"
    assert fp_table.fmt("clk/ev", "496.2345") == "496.2"
    assert fp_table.fmt("Gain tp", 12.0765) == "12.1"
    assert fp_table.fmt("u2", "-") == "-"
    assert fp_table.fmt("SASS", 15) == "15"
    assert fp_table.fmt("Worst at", None) == ""


def test_count_operation_drops_the_call_bookkeeping():
    # the bare function of fp32mp2 `low` mul as nvdisasm lists it: five FP instructions, the
    # return-address setup, the return
    block = [
        "FMUL R8, R4, R6 ;",
        "MOV R3, 0x0 ;",
        "FFMA R0, R4, R6, -R8 ;",
        "FFMA R0, R5, R7, R0 ;",
        "FFMA R0, R4, R7, R0 ;",
        "FFMA R9, R5, R6, R0 ;",
        "RET.REL.NODEC R2 `(_ZN3fpb11bare_kernel) ;",
    ]
    assert fp_table.count_operation(block) == 5
    # a MOV of zero into any other register is an instruction of the operation
    assert fp_table.count_operation(["MOV R7, 0x0 ;", "RET.REL.NODEC R2 `(k) ;"]) == 1
    # an ABI return without a register does not name a setup instruction
    assert fp_table.count_operation(["MOV R3, 0x0 ;", "RET.ABS.NODEC R20 0x0 ;"]) == 1


def test_column_order_starts_with_the_key_and_the_family():
    assert fp_table.COLS[:4] == ["Func", "Family", "Type", "Accuracy"]
    assert fp_table.MD_COLS == fp_table.COLS[:13]
    assert "Worst at" not in fp_table.MD_COLS
