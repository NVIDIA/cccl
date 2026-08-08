#!/usr/bin/env bash
set -euo pipefail

## Usage: dump_and_check test.a test.cu PREFIXES cuobjdump-mode [FileCheck-options...]
input_archive="${1}"
input_testfile="${2}"
input_prefix="${3}"
shift 3
dump_mode="${1:---dump-ptx}"
if (( $# > 0 )); then
  shift
fi
filecheck="${FILECHECK:-FileCheck}"
cuobjdump="${CUOBJDUMP:-cuobjdump}"

if [[ "${dump_mode}" == "--dump-sass" ]]; then
  # cuobjdump prints each instruction's control word on a separate line. Remove
  # those lines so FileCheck -NEXT describes adjacent SASS instructions.
  "${cuobjdump}" "${dump_mode}" "${input_archive}" \
    | sed -E '\|^[[:space:]]*/\* 0x[[:xdigit:]]+ \*/[[:space:]]*$|d' \
    | "${filecheck}" --match-full-lines --check-prefixes="${input_prefix}" "$@" "${input_testfile}"
else
  "${cuobjdump}" "${dump_mode}" "${input_archive}" \
    | "${filecheck}" --match-full-lines --check-prefixes="${input_prefix}" "$@" "${input_testfile}"
fi
