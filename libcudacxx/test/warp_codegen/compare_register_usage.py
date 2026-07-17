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

FUNCTION_PATTERN = re.compile(r"^\s*Function\s+(\S+):")
REGISTER_COUNT_PATTERN = re.compile(r"^\s*REG:(\d+)")


def extract_register_counts(resource_usage):
    register_counts = {}
    function_name = None

    for line in resource_usage.splitlines():
        function_match = FUNCTION_PATTERN.match(line)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuobjdump", required=True)
    parser.add_argument("--binary", required=True)
    arguments = parser.parse_args()

    resource_usage = subprocess.run(
        [arguments.cuobjdump, "--dump-resource-usage", arguments.binary],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    register_counts = extract_register_counts(resource_usage)

    required_functions = {"wrapper_stress", "native_stress"}
    missing_functions = required_functions - register_counts.keys()
    if missing_functions:
        for name in sorted(missing_functions):
            print(f"missing register count in resource usage: {name}", file=sys.stderr)
        return 1

    wrapper_registers = register_counts["wrapper_stress"]
    native_registers = register_counts["native_stress"]
    if wrapper_registers > native_registers:
        print(
            "wrapper_stress uses more registers than native_stress:\n"
            f"  wrapper: {wrapper_registers}\n"
            f"  native:  {native_registers}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
