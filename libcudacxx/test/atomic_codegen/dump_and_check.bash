#!/usr/bin/env bash
set -euo pipefail

## Usage: dump_and_check test.a test.cu PREFIXES [cuobjdump-mode]
input_archive="${1}"
input_testfile="${2}"
input_prefix="${3}"
dump_mode="${4:---dump-ptx}"
filecheck="${FILECHECK:-FileCheck}"

cuobjdump "${dump_mode}" "${input_archive}" | "${filecheck}" --match-full-lines --check-prefixes="${input_prefix}" "${input_testfile}"
