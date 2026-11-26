#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare a run of the CCCL fpmp benchmarks (the joined table fpmp.csv of fp_table.py) with
the stored baseline of the same GPU, cell by cell, and give a verdict.

    fp_compare.py --results OUT/results/fpmp.csv --baseline-dir baselines [tolerances]
                  [--out-csv compare.csv] [--out-md compare.md]
    fp_compare.py --results OUT/results/fpmp.csv --baseline other-run/fpmp.csv [...]
    fp_compare.py --results OUT/results/fpmp.csv --baseline-dir baselines --store [--note '...']

A cell is one (Func, Type, Accuracy) row. Rules, with the defaults:

  accuracy    FAIL when u2 grew by more than --acc-factor (1.5x) AND by more than --acc-abs
              (0.5 u2); for the cells without u2 (fp64mp2, double reference) FAIL when Bits
              dropped. A smaller error is reported as a note.
  throughput  FAIL when ev/clk/SM dropped by more than --perf-fail (25%), WARN beyond
              --perf-warn (10%); a gain beyond --perf-fail is a note. A single run on an idle
              GPU moves by about 5%; a GPU shared with another process by 20-40%, which is
              why the launcher warns when the device is busy.
  latency     the same rules on clk/ev, the other way round.
  SASS        any change is a WARN (the generated code changed; the number says by how much).
  Regs        WARN when the chain kernel takes more than --regs (8) registers more.

Only the cells present in both tables are compared; cells missing from either side are WARN.
The reference-precision rows (Type fp64, Accuracy '-') time the CUDA math library, not the
component: they appear in the detail table for orientation but take no part in the verdict,
and neither do the derived Gain columns.

Before the comparison, the Check column of every cell (the benchmark's own PASS/WARN/FAIL on
the maximum relative error, with the ts/fpmp thresholds) is reported: a FAIL fails the run
whether or not a baseline exists.

--out-csv and --out-md write the detail table: one row per cell with the baseline and the new
value of every compared column, the relative change of the timings and the cell's verdict (OK,
NOTE, WARN, FAIL; `ref` for a reference row, `new`/`missing` for a cell on one side only), so
that a clean summary line can still be looked into function by function.

The baseline is either one table given with --baseline (a run to compare with, e.g. the same
benchmarks on `main`) or, with --baseline-dir, the file <dir>/<Device slug>.csv chosen by the Device
column of the results, with a .meta sidecar (date, note). The exit code is 1 on any FAIL; the lines start with [OK],
[WARN], [FAIL] or [NOTE] so that a log scan picks them up. The last line is the verdict:
`[OK]   compare: 306 cells against NVIDIA_L40S.csv`, with the non-zero counts in parentheses
(`(2 fail, 5 warn, 1 note)`) and the level raised to WARN or FAIL accordingly. The tolerances
are not repeated there; they are the options above and the README.
"""

import argparse
import csv
import datetime
import os
import re
import sys

KEY = ("Func", "Type", "Accuracy")
TYPE_ORDER = {"fp32mp2": 0, "fp64mp2": 1, "fp64": 2, "fp128": 3}
ACC_ORDER = {"low": 0, "mid": 1, "high": 2, "-": 3}
# the detail table: the compared columns as baseline / new pairs, the timings with their change
DETAIL_COLS = [
    "Func",
    "Family",
    "Type",
    "Accuracy",
    "Check",
    "u2 base",
    "u2",
    "Bits base",
    "Bits",
    "ev/clk/SM base",
    "ev/clk/SM",
    "tp %",
    "clk/ev base",
    "clk/ev",
    "lat %",
    "SASS base",
    "SASS",
    "Regs base",
    "Regs",
    "Verdict",
]
MD_COLS = [
    "Func",
    "Family",
    "Type",
    "Accuracy",
    "Check",
    "u2",
    "Bits",
    "ev/clk/SM",
    "clk/ev",
    "SASS",
    "Regs",
    "Verdict",
]


def read_table(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {tuple(r[k] for k in KEY): r for r in rows}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def device_slug(rows):
    names = {r.get("Device", "") for r in rows.values()} - {""}
    if len(names) != 1:
        sys.exit(f"[FAIL] the results carry {len(names)} device names: {sorted(names)}")
    return re.sub(r"[^A-Za-z0-9]+", "_", names.pop()).strip("_")


def pct(b, n):
    return f"{(n / b - 1.0) * 100:+.1f}%" if b and n else ""


def compare_cell(b, n, a):
    """The rules on one cell: its verdict and the (level, message) pairs behind it."""
    fails, warns, notes = [], [], []
    bu, nu = num(b.get("u2")), num(n.get("u2"))
    if bu is not None and nu is not None:
        if nu > bu * a.acc_factor and nu - bu > a.acc_abs:
            fails.append(f"u2 {bu:.2f} -> {nu:.2f}")
        elif bu > nu * a.acc_factor and bu - nu > a.acc_abs:
            notes.append(f"u2 {bu:.2f} -> {nu:.2f} (better)")
    else:
        bb, nb = num(b.get("Bits")), num(n.get("Bits"))
        if bb is not None and nb is not None and nb < bb:
            fails.append(f"Bits {bb:.0f} -> {nb:.0f}")
    bt, nt = num(b.get("ev/clk/SM")), num(n.get("ev/clk/SM"))
    if bt and nt:
        rel = nt / bt - 1.0
        if rel < -a.perf_fail:
            fails.append(f"ev/clk/SM {bt:.3f} -> {nt:.3f} ({rel:+.1%})")
        elif rel < -a.perf_warn:
            warns.append(f"ev/clk/SM {bt:.3f} -> {nt:.3f} ({rel:+.1%})")
        elif rel > a.perf_fail:
            notes.append(f"ev/clk/SM {bt:.3f} -> {nt:.3f} ({rel:+.1%}, faster)")
    bl, nl = num(b.get("clk/ev")), num(n.get("clk/ev"))
    if bl and nl:
        rel = nl / bl - 1.0
        if rel > a.perf_fail:
            fails.append(f"clk/ev {bl:.1f} -> {nl:.1f} ({rel:+.1%})")
        elif rel > a.perf_warn:
            warns.append(f"clk/ev {bl:.1f} -> {nl:.1f} ({rel:+.1%})")
        elif rel < -a.perf_fail:
            notes.append(f"clk/ev {bl:.1f} -> {nl:.1f} ({rel:+.1%}, faster)")
    bs, ns = num(b.get("SASS")), num(n.get("SASS"))
    if bs is not None and ns is not None and bs != ns:
        warns.append(f"SASS {bs:.0f} -> {ns:.0f} (the code changed)")
    br, nr = num(b.get("Regs")), num(n.get("Regs"))
    if br is not None and nr is not None and nr - br > a.regs:
        warns.append(f"Regs {br:.0f} -> {nr:.0f}")
    verdict = "FAIL" if fails else "WARN" if warns else "NOTE" if notes else "OK"
    return verdict, [("FAIL", m) for m in fails] + [("WARN", m) for m in warns] + [
        ("NOTE", m) for m in notes
    ]


def detail_row(key, b, n, verdict):
    def g(r, c):
        return (r.get(c) or "") if r else ""

    return {
        "Func": key[0],
        "Family": g(n, "Family") or g(b, "Family"),
        "Type": key[1],
        "Accuracy": key[2],
        "Check": g(n, "Check"),
        "u2 base": g(b, "u2"),
        "u2": g(n, "u2"),
        "Bits base": g(b, "Bits"),
        "Bits": g(n, "Bits"),
        "ev/clk/SM base": g(b, "ev/clk/SM"),
        "ev/clk/SM": g(n, "ev/clk/SM"),
        "tp %": pct(num(g(b, "ev/clk/SM")), num(g(n, "ev/clk/SM"))),
        "clk/ev base": g(b, "clk/ev"),
        "clk/ev": g(n, "clk/ev"),
        "lat %": pct(num(g(b, "clk/ev")), num(g(n, "clk/ev"))),
        "SASS base": g(b, "SASS"),
        "SASS": g(n, "SASS"),
        "Regs base": g(b, "Regs"),
        "Regs": g(n, "Regs"),
        "Verdict": verdict,
    }


def md_cell(row, col):
    """`base -> new` where they differ, the timings always with their change."""
    b, n = row.get(col + " base", ""), row.get(col, "")
    change = row.get({"ev/clk/SM": "tp %", "clk/ev": "lat %"}.get(col, ""), "")
    if change:
        return f"{b} → {n} ({change})" if b else n
    return f"{b} → {n}" if b and b != n else n


def write_details(rows, out_csv, out_md):
    rows = sorted(
        rows,
        key=lambda r: (
            r["Family"],
            r["Func"],
            TYPE_ORDER.get(r["Type"], 9),
            ACC_ORDER.get(r["Accuracy"], 9),
        ),
    )
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
            w.writeheader()
            w.writerows(rows)
    if out_md:
        with open(out_md, "w") as f:
            f.write("| " + " | ".join(MD_COLS) + " |\n|" + "---|" * len(MD_COLS) + "\n")
            for r in rows:
                f.write(
                    "| "
                    + " | ".join(
                        md_cell(r, c)
                        if c not in KEY + ("Family", "Check", "Verdict")
                        else r[c]
                        for c in MD_COLS
                    )
                    + " |\n"
                )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", required=True, help="fpmp.csv of the run (fp_table.py output)"
    )
    ap.add_argument(
        "--baseline", help="one table to compare with (fpmp.csv of another run)"
    )
    ap.add_argument(
        "--baseline-dir", help="directory of the per-GPU baselines, <Device slug>.csv"
    )
    ap.add_argument(
        "--store",
        action="store_true",
        help="store the results as the baseline of their GPU",
    )
    ap.add_argument(
        "--note",
        default="",
        help="with --store: a note for the .meta sidecar (CCCL sha, ...)",
    )
    ap.add_argument(
        "--out-csv", help="write the detail table (one row per cell) as CSV here"
    )
    ap.add_argument("--out-md", help="write the detail table as markdown here")
    ap.add_argument("--acc-factor", type=float, default=1.5)
    ap.add_argument("--acc-abs", type=float, default=0.5)
    ap.add_argument("--perf-fail", type=float, default=0.25)
    ap.add_argument("--perf-warn", type=float, default=0.10)
    ap.add_argument("--regs", type=int, default=8)
    a = ap.parse_args()

    new = read_table(a.results)
    if not new:
        sys.exit(f"[FAIL] no rows in {a.results}")
    if a.baseline_dir:
        base_path = os.path.join(a.baseline_dir, device_slug(new) + ".csv")
    elif a.baseline and not a.store:
        base_path = a.baseline
    else:
        sys.exit(
            "[FAIL] give --baseline <table> or --baseline-dir <dir> (--store needs --baseline-dir)"
        )

    if a.store:
        os.makedirs(a.baseline_dir, exist_ok=True)
        with open(a.results) as src, open(base_path, "w") as dst:
            dst.write(src.read())
        with open(base_path[:-4] + ".meta", "w") as m:
            m.write(
                f"date: {datetime.datetime.now():%Y-%m-%d %H:%M}\nhost: {os.uname().nodename}\ncells: {len(new)}\n"
            )
            if a.note:
                m.write(f"note: {a.note}\n")
        print(f"[OK]   baseline stored: {base_path} ({len(new)} cells)")
        return 0

    fails, warns, notes, details, compared = [], [], [], [], 0

    # the absolute verdict of each cell, from the benchmark's Check column
    for key, n in sorted(new.items()):
        if key[2] == "-":
            continue  # a reference-precision row: not the component's code
        chk = n.get("Check", "")
        if chk in ("FAIL", "WARN"):
            line = (
                f"[{chk}] {' '.join(key)}: accuracy check {chk} "
                f"(u2 {n.get('u2') or '-'}, Bits {n.get('Bits') or '-'}, worst at {n.get('Worst at') or '-'})"
            )
            (fails if chk == "FAIL" else warns).append(line)

    base = {}
    if os.path.exists(base_path):
        base = read_table(base_path)
        meta = base_path[:-4] + ".meta"
        if os.path.exists(meta):
            with open(meta) as m:
                print(
                    "baseline: " + "; ".join(line.strip() for line in m if line.strip())
                )
    else:
        warns.append(
            f"[WARN] no baseline for this GPU at {base_path}: only the accuracy check applies "
            f"(store one with --store once the numbers are the ones to keep)"
        )
        details = [detail_row(key, None, n, "-") for key, n in new.items()]

    for key in sorted(set(base) | set(new)) if base else []:
        name = " ".join(key)
        if key[2] == "-":  # a reference-precision row: shown, not judged
            if key in base and key in new:
                details.append(detail_row(key, base[key], new[key], "ref"))
            continue
        if key not in new:
            warns.append(f"[WARN] {name}: in the baseline, not in the results")
            details.append(detail_row(key, base[key], None, "missing"))
            continue
        if key not in base:
            warns.append(f"[WARN] {name}: new cell, not in the baseline")
            details.append(detail_row(key, None, new[key], "new"))
            continue
        compared += 1
        verdict, msgs = compare_cell(base[key], new[key], a)
        for level, m in msgs:
            {"FAIL": fails, "WARN": warns, "NOTE": notes}[level].append(
                f"[{level}] {name}: {m}"
            )
        details.append(detail_row(key, base[key], new[key], verdict))

    if a.out_csv or a.out_md:
        write_details(details, a.out_csv, a.out_md)
        print("details: " + ", ".join(p for p in (a.out_csv, a.out_md) if p))
    for line in fails + warns + notes:
        print(line)
    # the verdict line: the level, what was compared, and only the counts that are not zero
    level = "FAIL" if fails else "WARN" if warns else "OK"
    counts = [
        f"{len(x)} {w}"
        for x, w in (
            (fails, "fail"),
            (warns, "warn"),
            (notes, "note" if len(notes) == 1 else "notes"),
        )
        if x
    ]
    what = (
        f"{compared} cells against {os.path.basename(base_path)}"
        if base
        else "no baseline for this GPU, Check column only"
    )
    print(
        f"[{level}]{'   ' if level == 'OK' else ' '}compare: {what}"
        + (f" ({', '.join(counts)})" if counts else "")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
