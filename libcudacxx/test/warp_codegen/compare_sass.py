#!/usr/bin/env python3

# ===----------------------------------------------------------------------===#
#
# Part of libcu++ in the CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# ===----------------------------------------------------------------------===#

import argparse
import re
import subprocess
import sys
from collections import Counter

FUNCTION_PATTERN = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSTRUCTION_PATTERN = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b"
)
RESOURCE_FUNCTION_PATTERN = re.compile(r"^\s*Function\s+(\S+):")
REGISTER_COUNT_PATTERN = re.compile(r"^\s*REG:(\d+)")


def canonicalize_opcode(opcode):
    if opcode == "IMAD.MOV.U32":
        return "MOV"
    return opcode


def extract_functions(sass):
    functions = {}
    function_name = None

    for line in sass.splitlines():
        function_match = FUNCTION_PATTERN.match(line)
        if function_match:
            function_name = function_match.group(1)
            functions[function_name] = []
            continue

        if function_name is None:
            continue

        instruction_match = INSTRUCTION_PATTERN.match(line)
        if not instruction_match:
            continue

        opcode = canonicalize_opcode(instruction_match.group(1))
        functions[function_name].append(opcode)
        if opcode.startswith("RET"):
            function_name = None

    return functions


def extract_register_counts(resource_usage):
    register_counts = {}
    function_name = None

    for line in resource_usage.splitlines():
        function_match = RESOURCE_FUNCTION_PATTERN.match(line)
        if function_match:
            function_name = function_match.group(1)
            continue

        if function_name is None:
            continue

        register_match = REGISTER_COUNT_PATTERN.match(line)
        if register_match:
            register_counts[function_name] = int(register_match.group(1))
            function_name = None

    return register_counts


def check_pair(functions, mode, suffix):
    wrapper_name = f"wrapper_{mode}_{suffix}"
    native_name = f"native_{mode}_{suffix}"

    try:
        wrapper = functions[wrapper_name]
        native = functions[native_name]
    except KeyError as error:
        return [f"missing function in SASS: {error.args[0]}"]

    wrapper_shuffles = [opcode for opcode in wrapper if opcode.startswith("SHFL.")]
    native_shuffles = [opcode for opcode in native if opcode.startswith("SHFL.")]
    errors = []

    if wrapper_shuffles != native_shuffles:
        errors.append(
            f"{wrapper_name} shuffle sequence differs from {native_name}:\n"
            f"  wrapper: {wrapper_shuffles}\n"
            f"  native:  {native_shuffles}"
        )

    wrapper_opcodes = Counter(wrapper)
    native_opcodes = Counter(native)
    has_no_extra_instructions = (
        len(wrapper) <= len(native) and wrapper_opcodes <= native_opcodes
    )
    if not has_no_extra_instructions:
        errors.append(
            f"{wrapper_name} is not instruction-equivalent to or shorter than {native_name}:\n"
            f"  wrapper ({len(wrapper)}): {' '.join(wrapper)}\n"
            f"  native  ({len(native)}): {' '.join(native)}"
        )

    return errors


def check_stress(functions, register_counts):
    errors = []
    missing_functions = {"wrapper_stress", "native_stress"} - functions.keys()
    if missing_functions:
        return [f"missing function in SASS: {name}" for name in sorted(missing_functions)]

    missing_register_counts = {"wrapper_stress", "native_stress"} - register_counts.keys()
    if missing_register_counts:
        return [
            f"missing register count in resource usage: {name}"
            for name in sorted(missing_register_counts)
        ]

    wrapper = functions["wrapper_stress"]
    native = functions["native_stress"]
    wrapper_shuffles = [opcode for opcode in wrapper if opcode.startswith("SHFL.")]
    native_shuffles = [opcode for opcode in native if opcode.startswith("SHFL.")]

    if wrapper_shuffles != native_shuffles:
        errors.append(
            "wrapper_stress shuffle sequence differs from native_stress:\n"
            f"  wrapper: {wrapper_shuffles}\n"
            f"  native:  {native_shuffles}"
        )

    wrapper_registers = register_counts["wrapper_stress"]
    native_registers = register_counts["native_stress"]
    if wrapper_registers > native_registers:
        errors.append(
            "wrapper_stress uses more registers than native_stress:\n"
            f"  wrapper: {wrapper_registers}\n"
            f"  native:  {native_registers}"
        )

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuobjdump", required=True)
    parser.add_argument("--binary", required=True)
    arguments = parser.parse_args()

    sass = subprocess.run(
        [arguments.cuobjdump, "--dump-sass", arguments.binary],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    functions = extract_functions(sass)
    resource_usage = subprocess.run(
        [arguments.cuobjdump, "--dump-resource-usage", arguments.binary],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    register_counts = extract_register_counts(resource_usage)

    errors = []
    if "wrapper_idx_u8" in functions:
        for suffix in ("u8", "u16", "u32", "u64"):
            for mode in ("idx", "up", "down", "xor"):
                errors.extend(check_pair(functions, mode, suffix))
    elif "wrapper_stress" in functions:
        errors.extend(check_stress(functions, register_counts))
    else:
        errors.append("no known warp codegen functions found")

    if errors:
        print("\n\n".join(errors), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
