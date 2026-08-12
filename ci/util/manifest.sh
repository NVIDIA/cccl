#!/usr/bin/env bash

# Do not enable `eo pipefail`
# read relies on a quirk of unterminated/unquoted heredoc to capture input.
# TODO: rewrite using printf if we can keep nice formatting.
set -u

path="$(realpath "$1")"
outfile="$2"
version="$3"
platform="$4"

read -r -d '' manifest << EOF
{
"schema-version": 1,
"componentName": "cccl",
"componentVersion": "$version",
"platform": "$platform"
}
EOF

read -r -d '' prog << 'EOF'
split(.,"\n")
  | select(length > 2)
  | [{(.[]) : {"type": "header"}}]
  | add
  | $manifest + { "files" : . }
EOF

find "$path" -wholename '*include/*' -type f -printf '%P\n' \
  | jq -s --raw-input --argjson manifest "$manifest" "$prog" > "${outfile}"
