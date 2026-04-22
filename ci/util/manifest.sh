#!/usr/bin/env bash
set -u

path="$(realpath $1)"
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

find "$path" -wholename "*include/*" -type f -printf "%P\n" \
  | jq -s --raw-input --argjson manifest "$manifest" "$prog" > $2