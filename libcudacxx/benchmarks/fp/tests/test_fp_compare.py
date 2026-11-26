# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Unit tests of fp_compare.py: the cell rules, the device slug, the baseline modes and the verdict line.

pytest -q libcudacxx/benchmarks/fp/tests
"""

import argparse
import csv
import sys

import fp_compare
import pytest

COLS = [
    "Func",
    "Family",
    "Type",
    "Accuracy",
    "u2",
    "Bits",
    "Check",
    "ev/clk/SM",
    "Gain tp",
    "clk/ev",
    "Gain lat",
    "SASS",
    "Regs",
    "Worst at",
    "Device",
]


def tolerances(**kw):
    a = dict(acc_factor=1.5, acc_abs=0.5, perf_fail=0.25, perf_warn=0.10, regs=8)
    a.update(kw)
    return argparse.Namespace(**a)


def cell(**kw):
    c = {
        "u2": "1.0",
        "Bits": "48",
        "Check": "PASS",
        "ev/clk/SM": "0.100",
        "clk/ev": "100.0",
        "SASS": "20",
        "Regs": "30",
    }
    c.update({k: str(v) for k, v in kw.items()})
    return c


def test_accuracy_rules():
    a = tolerances()
    # grew by 1.5x and by 0.5: FAIL
    assert fp_compare.compare_cell(cell(u2=1.0), cell(u2=1.6), a)[0] == "FAIL"
    # grew by 1.5x but not by 0.5 u2: not a FAIL (small absolute change)
    assert fp_compare.compare_cell(cell(u2=0.1), cell(u2=0.4), a)[0] == "OK"
    # a smaller error is a note
    verdict, msgs = fp_compare.compare_cell(cell(u2=2.0), cell(u2=1.0), a)
    assert verdict == "NOTE" and "better" in msgs[0][1]
    # without u2 (fp64mp2), a Bits drop fails and a Bits gain passes
    assert (
        fp_compare.compare_cell(cell(u2="-", Bits=53), cell(u2="-", Bits=52), a)[0]
        == "FAIL"
    )
    assert (
        fp_compare.compare_cell(cell(u2="-", Bits=52), cell(u2="-", Bits=53), a)[0]
        == "OK"
    )


def test_performance_rules():
    a = tolerances()
    # throughput: down 30% FAIL, down 15% WARN, up 30% NOTE, within 10% OK
    assert fp_compare.compare_cell(cell(), cell(**{"ev/clk/SM": 0.07}), a)[0] == "FAIL"
    assert fp_compare.compare_cell(cell(), cell(**{"ev/clk/SM": 0.085}), a)[0] == "WARN"
    assert fp_compare.compare_cell(cell(), cell(**{"ev/clk/SM": 0.13}), a)[0] == "NOTE"
    assert fp_compare.compare_cell(cell(), cell(**{"ev/clk/SM": 0.095}), a)[0] == "OK"
    # latency: the other way round
    assert fp_compare.compare_cell(cell(), cell(**{"clk/ev": 130.0}), a)[0] == "FAIL"
    assert fp_compare.compare_cell(cell(), cell(**{"clk/ev": 115.0}), a)[0] == "WARN"
    assert fp_compare.compare_cell(cell(), cell(**{"clk/ev": 70.0}), a)[0] == "NOTE"
    # a wider tolerance turns the FAIL into nothing
    assert (
        fp_compare.compare_cell(
            cell(), cell(**{"clk/ev": 130.0}), tolerances(perf_fail=100, perf_warn=100)
        )[0]
        == "OK"
    )


def test_code_rules():
    a = tolerances()
    verdict, msgs = fp_compare.compare_cell(cell(SASS=20), cell(SASS=19), a)
    assert verdict == "WARN" and msgs == [("WARN", "SASS 20 -> 19 (the code changed)")]
    assert fp_compare.compare_cell(cell(Regs=30), cell(Regs=38), a)[0] == "OK"
    assert fp_compare.compare_cell(cell(Regs=30), cell(Regs=39), a)[0] == "WARN"
    # the worst level wins and every message is kept
    verdict, msgs = fp_compare.compare_cell(
        cell(), cell(u2=5.0, SASS=25, **{"clk/ev": 60.0}), a
    )
    assert verdict == "FAIL" and [m[0] for m in msgs] == ["FAIL", "WARN", "NOTE"]


def test_device_slug():
    rows = {
        ("a", "b", "c"): {"Device": "NVIDIA RTX A6000"},
        ("d", "b", "c"): {"Device": "NVIDIA RTX A6000"},
    }
    assert fp_compare.device_slug(rows) == "NVIDIA_RTX_A6000"
    rows[("e", "b", "c")] = {"Device": "NVIDIA L40S"}
    with pytest.raises(SystemExit):
        fp_compare.device_slug(rows)


def write_table(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in COLS}, **r})


def row(func, typ="fp32mp2", acc="mid", **kw):
    r = {
        "Func": func,
        "Family": "arith",
        "Type": typ,
        "Accuracy": acc,
        "Device": "NVIDIA L40S",
        **cell(),
    }
    r.update({k: str(v) for k, v in kw.items()})
    return r


def run_main(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["fp_compare.py", *args])
    code = fp_compare.main()
    return code, capsys.readouterr().out


def test_store_and_compare_with_baseline_dir(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.csv"
    write_table(
        base,
        [row("add"), row("div"), row("add", "fp64", "-", u2="-", Bits="-", Check="-")],
    )
    code, out = run_main(
        monkeypatch,
        capsys,
        "--results",
        str(base),
        "--baseline-dir",
        str(tmp_path / "bl"),
        "--store",
        "--note",
        "main abc",
    )
    assert code == 0 and "baseline stored" in out
    assert (tmp_path / "bl" / "NVIDIA_L40S.csv").exists()
    assert "note: main abc" in (tmp_path / "bl" / "NVIDIA_L40S.meta").read_text()

    # the same numbers again: a clean verdict; the reference row is listed as `ref`, not judged
    new = tmp_path / "new.csv"
    write_table(
        new,
        [
            row("add"),
            row("div"),
            row("add", "fp64", "-", u2="-", Bits="-", Check="-", SASS=99),
        ],
    )
    code, out = run_main(
        monkeypatch,
        capsys,
        "--results",
        str(new),
        "--baseline-dir",
        str(tmp_path / "bl"),
        "--out-csv",
        str(tmp_path / "c.csv"),
        "--out-md",
        str(tmp_path / "c.md"),
    )
    assert code == 0
    assert (
        out.strip().splitlines()[-1]
        == "[OK]   compare: 2 cells against NVIDIA_L40S.csv"
    )
    details = {
        (r["Func"], r["Type"], r["Accuracy"]): r
        for r in csv.DictReader(open(tmp_path / "c.csv"))
    }
    assert details[("add", "fp64", "-")]["Verdict"] == "ref"
    assert details[("add", "fp32mp2", "mid")]["Verdict"] == "OK"
    assert (
        (tmp_path / "c.md")
        .read_text()
        .startswith("| Func | Family | Type | Accuracy | Check |")
    )

    # a regression and a missing cell: FAIL with the counts, exit 1
    write_table(new, [row("add", **{"clk/ev": 140.0})])
    code, out = run_main(
        monkeypatch,
        capsys,
        "--results",
        str(new),
        "--baseline-dir",
        str(tmp_path / "bl"),
    )
    assert code == 1
    lines = out.strip().splitlines()
    assert "[FAIL] add fp32mp2 mid: clk/ev 100.0 -> 140.0 (+40.0%)" in lines
    assert "[WARN] div fp32mp2 mid: in the baseline, not in the results" in lines
    assert (
        lines[-1] == "[FAIL] compare: 1 cells against NVIDIA_L40S.csv (1 fail, 1 warn)"
    )


def test_single_baseline_file_and_check_column(tmp_path, monkeypatch, capsys):
    base, new = tmp_path / "base.csv", tmp_path / "new.csv"
    write_table(base, [row("add"), row("sub")])
    # the run's own Check column fails the run before any baseline rule
    write_table(
        new,
        [row("add", Check="FAIL", **{"Worst at": "0x1p+0"}), row("sub", Check="WARN")],
    )
    code, out = run_main(
        monkeypatch, capsys, "--results", str(new), "--baseline", str(base)
    )
    assert code == 1
    assert (
        "[FAIL] add fp32mp2 mid: accuracy check FAIL (u2 1.0, Bits 48, worst at 0x1p+0)"
        in out
    )
    assert "[WARN] sub fp32mp2 mid: accuracy check WARN" in out
    assert (
        out.strip().splitlines()[-1]
        == "[FAIL] compare: 2 cells against base.csv (1 fail, 1 warn)"
    )
    # no baseline at all: only the Check column, WARN
    code, out = run_main(
        monkeypatch,
        capsys,
        "--results",
        str(base),
        "--baseline",
        str(tmp_path / "none.csv"),
    )
    assert code == 0 and out.strip().splitlines()[-1].startswith(
        "[WARN] compare: no baseline"
    )


def test_baseline_options_are_required(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.csv"
    write_table(base, [row("add")])
    with pytest.raises(SystemExit):
        run_main(monkeypatch, capsys, "--results", str(base))
    with pytest.raises(SystemExit):
        run_main(
            monkeypatch,
            capsys,
            "--results",
            str(base),
            "--baseline",
            str(base),
            "--store",
        )
