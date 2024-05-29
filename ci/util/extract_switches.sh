#!/bin/bash

# Similar to getopt, but only extracts recognized switches and leaves all other arguments in place.
#
# Example Usage:
#   new_args=$(extract_switches.sh -cpu-only -gpu-only -- "$@")
#   eval set -- ${new_args}
#   while true; do
#     case "$1" in
#     -cpu-only) CPU_ONLY=true; shift;;
#     -gpu-only) GPU_ONLY=true; shift;;
#     --) shift; break;;
#     *) echo "Unknown argument: $1"; exit 1;;
#     esac
#   done
#
# This leaves all unrecognized arguments in $@ for later parsing.

# Parse switches
switches=()
for arg in "$@"; do
  case "$arg" in
  --help | -h)
    cat <<-EOF | cut -c 5-
    Usage: extract_switches.sh <switch> [<switch> ...] -- <argv>

    Sorts any recognized switches in argv to the front and returns the result.
    Unrecognized switches are left in place.

    Example Usage:
      new_args=\$(extract_switches.sh -cpu-only -gpu-only
      eval set -- \${new_args}
      while true; do
        case "\$1" in
        -cpu-only) CPU_ONLY=true; shift;;
        -gpu-only) GPU_ONLY=true; shift;;
        --) shift; break;;
        *) echo "Unknown argument: \$1"; exit 1;;
        esac
      done
EOF
    exit
    ;;
  --)
    shift
    break
    ;;
  *)
    switches+=("$arg")
    shift
    ;;
  esac
done

found_switches=()
other_args=()
for arg in "$@"; do
  for switch in "${switches[@]}"; do
    if [ "$arg" = "$switch" ]; then
      found_switches+=("$arg")
      continue 2
    fi
  done
  other_args+=("$arg")
done

echo "${found_switches[@]} -- ${other_args[@]}"
