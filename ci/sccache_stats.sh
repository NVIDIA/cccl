#!/bin/bash

# This script prints the sccache hit rate between two calls to sccache --show-stats.
# It must be sourced. Exits with an error if executed directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: ci/sccache_stats.sh must be sourced, not executed."
  echo "Usage: source ci/sccache_stats.sh {start|end}"
  exit 2
fi

mode=$1

if [[ "$mode" != "start" && "$mode" != "end" ]]; then
    echo "Invalid mode: $mode"
    echo "Usage: source ${BASH_SOURCE[0]} {start|end}"
    return 1
fi

# Check if sccache is available
if ! command -v sccache &> /dev/null; then
    echo "Notice: sccache is not available. Skipping..."
    return 0
fi

case $mode in
  start)
    export SCCACHE_START_HITS=$(sccache --show-stats | awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}')
    export SCCACHE_START_MISSES=$(sccache --show-stats | awk '/^[ \t]*Cache misses[ \t]+[0-9]+/ {print $3}')
    ;;
  end)
    if [[ -z ${SCCACHE_START_HITS+x} || -z ${SCCACHE_START_MISSES+x} ]]; then
        echo "Error: start stats not collected. Did you call this script with 'start' before your operations?"
        return 1
    fi

    final_hits=$(sccache --show-stats | awk '/^[ \t]*Cache hits[ \t]+[0-9]+/ {print $3}')
    final_misses=$(sccache --show-stats | awk '/^[ \t]*Cache misses[ \t]+[0-9]+/ {print $3}')
    hits=$((final_hits - SCCACHE_START_HITS))
    misses=$((final_misses - SCCACHE_START_MISSES))
    total=$((hits + misses))

    if (( total > 0 )); then
      hit_rate=$(awk -v hits="$hits" -v total="$total" 'BEGIN { printf "%.2f", (hits / total) * 100 }')
      echo "sccache hits: $hits | misses: $misses | hit rate: $hit_rate%"
    else
      echo "sccache stats: N/A No new compilation requests"
    fi
    unset SCCACHE_START_HITS
    unset SCCACHE_START_MISSES
    ;;
esac
