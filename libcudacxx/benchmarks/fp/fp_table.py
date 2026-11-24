#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""One row per (Func, Type, Accuracy), with the family second, from the CSV output of the fpmp benchmark executables
(libcudacxx/benchmarks/bench/fp/fpmp/*.cu), with the SASS instruction count of one evaluation
joined in from each executable.

    fp_table.py --exe bin/libcudacxx.bench.fp.fpmp.arith.base --csv out/arith.csv \\
                --exe bin/libcudacxx.bench.fp.fpmp.explog.base --csv out/explog.csv ... \\
                [--out table.csv] [--md table.md]

or, with the executables in one directory and the CSVs named after them in another:

    fp_table.py --bin-dir build/bin --csv-dir out [--out table.csv] [--md table.md]

Each executable compiles a __noinline__ function `fpb::bare<Op, cuda::fpmp2<L, A>>` per cell.
cuobjdump extracts the cubins and nvdisasm lists every kernel's section with the local functions
it calls as labelled blocks (`$<kernel>$<function>:`); the SASS column is the number of
instructions of the bare function's block, label to RET, i.e. the instructions one evaluation of
the operation takes, without its CALL/RET pair, the return-address setup before the RET and the
kernel's loads and stores. Where a cubin holds several architectures the largest count is kept.

Each executable also carries `ref_throughput` and `ref_latency`: the same timing of the reference
precision, the double expression of the declaration on a double (fp32mp2's reference; fp64mp2
waits for a binary128 type on the device). Their rows have Type `fp64` and Accuracy `-`, and every
pair cell gets `Gain tp` (its ev/clk/SM over the reference's) and `Gain lat` (the reference's
clk/ev over its own), both above 1 when the pair is faster than the reference precision.

`Check` is the benchmark's own verdict on the accuracy, PASS, WARN or FAIL against the thresholds
of the cccl-fpdev ts/fpmp suite (see the README); the table carries it over.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ACC = {"1": "low", "2": "mid", "3": "high"}
TYP = {"float": "fp32mp2", "double": "fp64mp2"}
BARE_RE = re.compile(
    r"fpb::bare<fpb::op_(\w+), cuda::(?:__\d+::)?fpmp2<(float|double), \(cuda::(?:__\d+::)?fpmp2_accuracy\)(\d)>"
)
# the bare function of the reference precision: a plain scalar
BARE_REF_RE = re.compile(r"fpb::bare<fpb::op_(\w+), (float|double)>")
REF_TYP = {"float": "fp32", "double": "fp64"}
# the reference precision of each pair type, as the Type of its ref_* rows
REF_OF = {"fp32mp2": "fp64", "fp64mp2": "fp128"}
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
MD_COLS = COLS[:13]
TYPE_ORDER = {"fp32mp2": 0, "fp64mp2": 1, "fp64": 2, "fp128": 3}
ACC_ORDER = {"low": 0, "mid": 1, "high": 2, "-": 3}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sass_counts(exe):
    """{(func, type, accuracy): instruction count} of the bare functions in `exe`.

    cuobjdump extracts the cubins and nvdisasm lists each kernel's section: the kernel's own code
    first, then the local functions it calls as labelled blocks (`$<kernel>$<function>:`), each
    split further by branch labels (`.L_x_N:`). In a bare kernel's section everything after the
    bare function's label is the operation: its own body, its out-of-line paths and any fpmp
    helper the compiler kept as a function. Not counted: NOP padding, RET, the self-loop BRA that
    ends a section, and CUDA's own `__cuda_*` subroutines (the suite's rule as well).
    """
    bindir = os.path.dirname(
        shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    )
    filt = shutil.which("c++filt")

    def demangle(names):
        if not filt or not names:
            return list(names)
        return subprocess.run(
            [filt], input="\n".join(names), capture_output=True, text=True
        ).stdout.splitlines()

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [os.path.join(bindir, "cuobjdump"), "-xelf", "all", os.path.abspath(exe)],
            check=True,
            capture_output=True,
            cwd=tmp,
        )
        for cubin in sorted(os.listdir(tmp)):
            if not cubin.endswith(".cubin"):
                continue
            dis = subprocess.run(
                [os.path.join(bindir, "nvdisasm"), "-c", os.path.join(tmp, cubin)],
                capture_output=True,
                text=True,
            ).stdout
            # pass 1: the sections of the bare kernels and the local labels they contain
            sections = {}  # section symbol -> list of lines
            cur = None
            for line in dis.splitlines():
                m = re.match(r"\t\.section\t\.text\.(\S+?),", line)
                if m:
                    cur = m.group(1) if "bare_kernel" in m.group(1) else None
                    if cur:
                        sections[cur] = []
                    continue
                if cur:
                    sections[cur].append(line)
            if not sections:
                continue
            kernels = list(sections)
            locals_ = sorted(
                {
                    m.group(1)
                    for lines in sections.values()
                    for text in lines
                    for m in [re.match(r"^\$\S+\$(\S+):\s*$", text)]
                    if m
                }
            )
            pretty = dict(zip(kernels + locals_, demangle(kernels + locals_)))
            # pass 2: count
            for sym, lines in sections.items():
                name = pretty[sym].replace("bare_kernel", "bare")
                m = BARE_RE.search(name)
                if m:
                    key = (m.group(1), TYP[m.group(2)], ACC[m.group(3)])
                else:
                    m = BARE_REF_RE.search(name)
                    if not m:
                        continue
                    key = (m.group(1), REF_TYP[m.group(2)], "-")
                kind, label = "kernel", None
                bodies = []  # the instructions of the bare function and its helpers
                for line in lines:
                    lm = re.match(r"^\$\S+\$(\S+):\s*$", line)
                    if lm:
                        name = pretty.get(lm.group(1), lm.group(1))
                        kind = (
                            "cuda"
                            if name.startswith("__cuda")
                            else ("bare" if "fpb::bare<" in name else "helper")
                        )
                        label = lm.group(1)
                        continue
                    lm = re.match(r"^(\.L_x_\d+):\s*$", line)
                    if lm:
                        label = lm.group(1)
                        continue
                    if re.match(r"^\S+:\s*$", line):  # the kernel's own symbol
                        kind = "kernel"
                        continue
                    if kind not in ("bare", "helper") or not re.match(
                        r"\s+/\*[0-9a-f]{4,}\*/", line
                    ):
                        continue
                    body = line.split("*/", 1)[1].strip()
                    if body.startswith("NOP"):
                        continue
                    bm = re.match(r"^BRA\s+`\((\S+)\)", body)
                    if bm and label and bm.group(1) == label:
                        continue  # the self-loop after the last RET
                    bodies.append(body)
                out[key] = max(out.get(key, 0), count_operation(bodies))
    return out


def count_operation(bodies):
    """The instructions of one evaluation: the block without its call bookkeeping. `RET` returns
    to the caller, and the `MOV R<k+1>, 0x0` before a `RET.REL.NODEC R<k>` only completes the
    64-bit return address the caller passed in R<k>; neither belongs to the operation. What is
    left agrees with a standalone compilation of the function up to its own ABI moves."""
    rets = [
        m.group(1)
        for b in bodies
        for m in [re.match(r"^RET\.REL\.NODEC R(\d+)", b)]
        if m
    ]
    skip = {f"MOV R{int(r) + 1}, 0x0" for r in rets}
    return sum(
        1 for b in bodies if not re.match(r"^RET\b", b) and b.rstrip(" ;") not in skip
    )


def family_of(exe):
    m = re.search(r"fp\.fpmp\.(\w+)", os.path.basename(exe))
    return m.group(1) if m else os.path.basename(exe)


def merge(exe, csv_path, rows):
    sass = sass_counts(exe)
    fam = family_of(exe)
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            # the type axes carry the {ct} suffix of the benchmark infrastructure
            # (compile-time axes); the table's columns do not
            r = {k.replace("{ct}", ""): v for k, v in raw.items()}
            if r.get("Skipped", "No") == "Yes":
                continue
            # the reference rows have no Accuracy axis
            key = (r["Func"], r["Type"], r.get("Accuracy") or "-")
            row = rows.setdefault(
                key,
                {
                    "Func": key[0],
                    "Type": key[1],
                    "Accuracy": key[2],
                    "Family": fam,
                    "Device": r.get("Device Name", ""),
                },
            )
            bench = r["Benchmark"]
            if bench in ("throughput", "ref_throughput"):
                row["Regs"] = r.get("Regs", "")
                row["ev/clk/SM"] = r.get("ev/clk/SM", "")
                if bench == "throughput":
                    row["u2"] = r.get("u2", "")
                    row["Bits"] = r.get("Bits", "")
                    row["Check"] = r.get("Check", "")
                    row["Worst at"] = r.get("Worst at", "")
                else:  # the reference has no accuracy to measure
                    row["u2"] = row["Bits"] = row["Check"] = row["Worst at"] = "-"
            elif bench in ("latency", "ref_latency"):
                row["clk/ev"] = r.get("clk/ev", "")
            row["SASS"] = sass.get(key, "")


def add_gains(rows):
    """Gain tp and Gain lat of each pair cell against the reference-precision row of its
    function: throughput as pair over reference, latency as reference over pair, so both read
    above 1 when the pair is faster. '-' without a reference row (fp64mp2 until binary128)."""
    for row in rows.values():
        row["Gain tp"] = row["Gain lat"] = "-"
        ref = rows.get((row["Func"], REF_OF.get(row["Type"], ""), "-"))
        if ref is None:
            continue
        tp, rtp = num(row.get("ev/clk/SM")), num(ref.get("ev/clk/SM"))
        if tp and rtp:
            row["Gain tp"] = tp / rtp
        lat, rlat = num(row.get("clk/ev")), num(ref.get("clk/ev"))
        if lat and rlat:
            row["Gain lat"] = rlat / lat


def fmt(k, v):
    """Four significant digits for the measured quantities: two decimals would quantize the
    small fp64mp2 throughputs (0.08 ev/clk/SM) to 12%, more than the noise a comparison looks for."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v if v is not None else ""
    if k in ("u2", "ev/clk/SM", "clk/ev"):
        return f"{x:.4g}"
    if k in ("Gain tp", "Gain lat"):
        return f"{x:.3g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exe",
        action="append",
        default=[],
        help="a benchmark executable (repeatable, paired with --csv)",
    )
    ap.add_argument(
        "--csv", action="append", default=[], help="its --csv output (repeatable)"
    )
    ap.add_argument(
        "--bin-dir",
        help="directory with the libcudacxx.bench.fp.fpmp.*.base executables",
    )
    ap.add_argument("--csv-dir", help="directory with <family>.csv for each of them")
    ap.add_argument("--out", help="merged CSV (default: stdout)")
    ap.add_argument("--md", help="also write a markdown table here")
    a = ap.parse_args()

    pairs = list(zip(a.exe, a.csv))
    if len(a.exe) != len(a.csv):
        sys.exit("--exe and --csv must be given in pairs")
    if a.bin_dir and a.csv_dir:
        for exe in sorted(
            glob.glob(os.path.join(a.bin_dir, "libcudacxx.bench.fp.fpmp.*.base"))
        ):
            c = os.path.join(a.csv_dir, family_of(exe) + ".csv")
            if os.path.exists(c):
                pairs.append((exe, c))
    if not pairs:
        sys.exit("nothing to do: give --exe/--csv pairs or --bin-dir and --csv-dir")

    rows = {}
    for exe, c in pairs:
        merge(exe, c, rows)
    add_gains(rows)
    table = sorted(
        rows.values(),
        key=lambda r: (
            r["Family"],
            r["Func"],
            TYPE_ORDER.get(r["Type"], 9),
            ACC_ORDER.get(r["Accuracy"], 9),
        ),
    )

    out = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.writer(out)
    w.writerow(COLS)
    for r in table:
        w.writerow([fmt(c, r.get(c, "")) for c in COLS])
    if a.out:
        out.close()
    if a.md:
        md_cols = MD_COLS
        with open(a.md, "w") as f:
            f.write("| " + " | ".join(md_cols) + " |\n")
            f.write("|" + "---|" * len(md_cols) + "\n")
            for r in table:
                f.write(
                    "| " + " | ".join(fmt(c, r.get(c, "")) for c in md_cols) + " |\n"
                )


if __name__ == "__main__":
    main()
