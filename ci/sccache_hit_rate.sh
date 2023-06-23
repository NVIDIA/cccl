#!/bin/bash

set -euo pipefail

# Ensure two arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <before-file> <after-file>" >&2
  exit 1
fi

# Print the contents of the before file
echo "=== Contents of $1 ===" >&2
cat $1 >&2
echo "=== End of $1 ===" >&2

# Print the contents of the after file
echo "=== Contents of $2 ==="  >&2
cat $2 >&2
echo "=== End of $2 ===" >&2

# Extract compile requests and cache hits from the before and after files
requests_before=$(awk '/^[ \t]*Compile requests[ \t]+[0-9]+/ {print $3}' "$1")
hits_before=$(awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}' "$1")
requests_after=$(awk '/^[ \t]*Compile requests[ \t]+[0-9]+/ {print $3}' "$2")
hits_after=$(awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}' "$2")

# Calculate the differences to find out how many new requests and hits
requests_diff=$((requests_after - requests_before))
hits_diff=$((hits_after - hits_before))

echo "New Compile Requests: $requests_diff" >&2
echo "New Hits: $hits_diff" >&2

# Calculate and print the hit rate
if [ $requests_diff -eq 0 ]; then
    echo "No new compile requests, hit rate is not applicable"
else
    hit_rate=$(awk -v hits=$hits_diff -v requests=$requests_diff 'BEGIN {printf "%.2f", hits/requests * 100}')
    echo "sccache hit rate: $hit_rate%" >&2
    echo "$hit_rate" 
fi
