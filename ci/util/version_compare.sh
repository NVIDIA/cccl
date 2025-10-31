#!/bin/bash

set -euo pipefail

readonly usage=$(cat <<EOF
Usage: $0 X.Y[.Z[.W[...]]] <compare_op> A.B[.C[.D[...]]]

Compares two version strings with the specified operator.

compare_ops:
  lt - less than
  le - less than or equal to
  eq - equal to
  ne - not equal to
  ge - greater than or equal to
  gt - greater than
EOF
)

if [ "$#" -ne 3 ]; then
  echo "Error: Invalid arguments: $*" >&2
  echo "$usage" >&2
  exit 1
fi

version_a="$1"
operator="$2"
version_b="$3"

# Validate operator
if [[ ! "$operator" =~ ^(lt|le|eq|ne|ge|gt)$ ]]; then
  echo "Error: Invalid operator '$operator'. Must be one of: lt, le, eq, ne, ge, gt." >&2
  echo "$usage" >&2
  exit 1
fi

# Validate versions:
version_regex='^[0-9]+(\.[0-9]+)*$'
if [[ ! "$version_a" =~ $version_regex ]]; then
  echo "Error: Invalid version string '$version_a'." >&2
  echo "$usage" >&2
  exit 1
fi
if [[ ! "$version_b" =~ $version_regex ]]; then
  echo "Error: Invalid version string '$version_b'." >&2
  echo "$usage" >&2
  exit 1
fi

# Split versions into arrays
IFS='.' read -r -a ver_a_parts <<< "$version_a"
IFS='.' read -r -a ver_b_parts <<< "$version_b"
max_length=${#ver_a_parts[@]}
if [ "${#ver_b_parts[@]}" -gt "$max_length" ]; then
  max_length=${#ver_b_parts[@]}
fi

# Compare each part
for ((i=0; i<max_length; i++)); do
  part_a=${ver_a_parts[i]:-0}
  part_b=${ver_b_parts[i]:-0}
  if ((part_a < part_b)); then
    result="lt"
    break
  elif ((part_a > part_b)); then
    result="gt"
    break
  else
    result="eq"
  fi
done

# Evaluate the comparison based on the operator
case "$operator" in
  lt) [[ "$result" == "lt" ]] ;;
  le) [[ "$result" == "lt" || "$result" == "eq" ]] ;;
  eq) [[ "$result" == "eq" ]] ;;
  ne) [[ "$result" != "eq" ]] ;;
  ge) [[ "$result" == "gt" || "$result" == "eq" ]] ;;
  gt) [[ "$result" == "gt" ]] ;;
esac

exit $?
