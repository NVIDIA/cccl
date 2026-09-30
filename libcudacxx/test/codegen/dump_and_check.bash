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
dump_functions="${CUOBJDUMP_FUNCTIONS:-}"
filecheck="${FILECHECK:-FileCheck}"
cuobjdump="${CUOBJDUMP:-cuobjdump}"

cuobjdump_options=("${dump_mode}")
if [[ -n "${dump_functions}" ]]; then
  cuobjdump_options+=(--function "${dump_functions}")
fi

# cuobjdump prints each SASS instruction's control word on a separate line.
# Remove those lines so FileCheck -NEXT describes adjacent SASS instructions.
"${cuobjdump}" "${cuobjdump_options[@]}" "${input_archive}" \
  | sed -E '\|^[[:space:]]*/\* 0x[[:xdigit:]]+ \*/[[:space:]]*$|d' \
  | "${filecheck}" --match-full-lines --check-prefixes="${input_prefix}" "$@" "${input_testfile}"
