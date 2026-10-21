# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare the SASS of the Python and C++ benchmark kernels.

For every (algorithm, type, unroll factor, unroll mode) this records register,
shared, local and stack usage plus an opcode histogram for the cuda.coop kernel
and the matching ``benchmark_kernel`` instantiation in the CUB benchmark
executables. Disassembly is written next to the JSON summary for diffing.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

from numba_cuda_mlir import cuda

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from device_side_benchmark import (
    NUMPY_TYPES,
    kernel_attributes,
    kernel_cubin,
    make_benchmark_kernel,
    occupancy_grid_size,
)

BLOCK_SIZE = 256
CPP_TYPES = {
    "signed char": "I8",
    "short": "I16",
    "int": "I32",
    "long": "I64",
    "__half": "F16",
    "float": "F32",
    "double": "F64",
}
# Python algorithm -> (C++ executable, C++ functor) it is compared with.
CPP_COUNTERPARTS = {
    "sum": [("cub.bench.reduce.warp_reduce_sum_method.base", "sum_op_t")],
    "reduce_add": [("cub.bench.reduce.warp_reduce_sum.base", "benchmark_op_t")],
}
_INSTRUCTION = re.compile(
    r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)"
)
_KERNEL = re.compile(
    r"benchmark_kernel<\(int\)256, \(int\)(\d+), (\w+), ([\w ]+?), "
    r"\(bool\)([01])>"
)
_RESOURCES = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


def opcode_histogram(sass):
    histogram = collections.Counter()
    for match in _INSTRUCTION.finditer(sass):
        histogram[match.group(1) + match.group(2)] += 1
    return histogram


def summarize(sass):
    histogram = opcode_histogram(sass)
    families = collections.Counter()
    for opcode, count in histogram.items():
        families[opcode.split(".")[0]] += count
    # NOP padding is scheduler-dependent and differs between the toolchains
    # even when the instruction streams are otherwise identical.
    real = {
        opcode: count for opcode, count in histogram.items() if opcode != "NOP"
    }
    return {
        "instructions": sum(histogram.values()),
        "non_nop_instructions": sum(real.values()),
        "shfl": families["SHFL"],
        "redux": families["REDUX"],
        "branches": families["BRA"],
        "families": dict(sorted(families.items())),
        "opcodes": dict(sorted(histogram.items())),
    }


def disassemble_cubin(cubin):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        return subprocess.run(
            ["cuobjdump", "-sass", f.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def demangle(names):
    result = subprocess.run(
        ["cu++filt"],
        input="\n".join(names),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def split_functions(sass):
    """Map mangled function name to its SASS text."""
    functions = {}
    current = None
    for line in sass.splitlines():
        header = re.match(r"\s*Function : (\S+)", line)
        if header:
            current = header.group(1)
            functions[current] = []
        elif current is not None:
            functions[current].append(line)
    return {name: "\n".join(lines) for name, lines in functions.items()}


def cpp_kernels(executable):
    """Index benchmark kernels by (functor, T, unroll factor, unroll)."""
    sass = subprocess.run(
        ["cuobjdump", "-sass", executable],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    resources = subprocess.run(
        ["cuobjdump", "-res-usage", executable],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    usage = {}
    lines = resources.splitlines()
    for i, line in enumerate(lines):
        name = re.match(r"\s*Function (\S+):", line)
        if name and i + 1 < len(lines):
            numbers = _RESOURCES.search(lines[i + 1])
            if numbers:
                usage[name.group(1)] = numbers
    functions = {
        k: v
        for k, v in split_functions(sass).items()
        if "benchmark_kernel" in k
    }
    kernels = {}
    for mangled, readable in zip(functions, demangle(list(functions))):
        match = _KERNEL.search(readable)
        if not match:
            continue
        factor, functor, cpp_type, unroll = match.groups()
        type_name = CPP_TYPES.get(cpp_type.strip())
        if type_name is None:
            continue
        numbers = usage[mangled]
        kernels[(functor, type_name, int(factor), unroll == "1")] = {
            "name": readable,
            "sass": functions[mangled],
            "registers": int(numbers.group(1)),
            "stack_bytes": int(numbers.group(2)),
            "static_shared_bytes": int(numbers.group(3)),
            "local_bytes": int(numbers.group(4)),
        }
    return kernels


def _tag(record):
    return "-".join(
        [
            record["algorithm"],
            record["type"],
            str(record["unroll_factor"]),
            record["unroll"],
        ]
    )


def _write(out, name, text):
    with open(os.path.join(out, name), "w") as f:
        f.write(text)


def _real_opcodes(summary):
    return {k: v for k, v in summary["opcodes"].items() if k != "NOP"}


def python_record(algorithm, type_name, factor, unroll, out):
    """Compile one cuda.coop kernel and summarize its SASS."""
    kernel = make_benchmark_kernel(
        algorithm, type_name, unroll_factor=factor, unroll=unroll
    )
    sink_buffer = cuda.device_array(16, dtype=NUMPY_TYPES[type_name])
    grid = occupancy_grid_size(kernel, BLOCK_SIZE, sink_buffer)
    sass = disassemble_cubin(kernel_cubin(kernel))
    record = {
        "algorithm": algorithm,
        "type": type_name,
        "unroll_factor": factor,
        "unroll": "full" if unroll else "loop",
    }
    record["python"] = {
        **kernel_attributes(kernel),
        "grid": grid,
        **summarize(sass),
    }
    _write(out, f"python-{_tag(record)}.sass", sass)
    return record


def cpp_record(record, functor, cpp, out):
    """Summarize a C++ kernel and diff its opcodes against the Python one."""
    _write(out, f"cpp-{functor}-{_tag(record)}.sass", cpp["sass"])
    summary = summarize(cpp["sass"])
    python_opcodes = _real_opcodes(record["python"])
    cpp_opcodes = _real_opcodes(summary)
    differences = {
        k: python_opcodes.get(k, 0) - cpp_opcodes.get(k, 0)
        for k in sorted(set(python_opcodes) | set(cpp_opcodes))
        if python_opcodes.get(k, 0) != cpp_opcodes.get(k, 0)
    }
    return {
        "kernel": cpp["name"],
        "registers": cpp["registers"],
        "stack_bytes": cpp["stack_bytes"],
        "static_shared_bytes": cpp["static_shared_bytes"],
        "local_bytes": cpp["local_bytes"],
        **summary,
        "non_nop_opcodes_equal_python": python_opcodes == cpp_opcodes,
        "opcode_differences": differences,
    }


def _format_summary(label, summary):
    return (
        f"{label} regs={summary['registers']:3d} "
        f"insts={summary['non_nop_instructions']:5d} "
        f"shfl={summary['shfl']:4d} redux={summary['redux']:4d} "
        f"bra={summary['branches']:3d}"
    )


def format_record(record):
    parts = [f"{_tag(record):24s} " + _format_summary("py", record["python"])]
    for functor, c in record.get("cpp", {}).items():
        parts.append(
            _format_summary(functor, c)
            + f" same={c['non_nop_opcodes_equal_python']}"
            + f" diff={c['opcode_differences']}"
        )
    return " | ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cpp-bin",
        required=True,
        help="directory of the CUB benchmark executables",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--types", default="I8,I16,I32,I64,F32,F64")
    parser.add_argument("--unroll-factors", default="128")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    types = args.types.split(",")
    factors = [int(f) for f in args.unroll_factors.split(",")]
    cpp_index = {}
    for counterparts in CPP_COUNTERPARTS.values():
        for executable, _ in counterparts:
            if executable not in cpp_index:
                cpp_index[executable] = cpp_kernels(
                    os.path.join(args.cpp_bin, executable)
                )

    records = []
    for algorithm, counterparts in CPP_COUNTERPARTS.items():
        for type_name in types:
            for factor in factors:
                for unroll in (False, True):
                    record = python_record(
                        algorithm, type_name, factor, unroll, args.out
                    )
                    for executable, functor in counterparts:
                        cpp = cpp_index[executable].get(
                            (functor, type_name, factor, unroll)
                        )
                        if cpp is not None:
                            record.setdefault("cpp", {})[functor] = cpp_record(
                                record, functor, cpp, args.out
                            )
                    records.append(record)
                    print(format_record(record))
    with open(os.path.join(args.out, "audit.json"), "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
