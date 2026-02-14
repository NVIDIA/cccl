#!/bin/bash

set -euo pipefail

# Time limit for installation steps (seconds)
readonly install_time_limit=10

ci_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$ci_dir/pretty_printing.sh"

cd "$ci_dir/.."

# Prefixes for installations
prefix_default=$(mktemp -d /tmp/cccl-default-XXXX)
prefix_preset=$(mktemp -d /tmp/cccl-preset-XXXX)
prefix_script=$(mktemp -d /tmp/cccl-script-XXXX)
prefix_preset_unstable=$(mktemp -d /tmp/cccl-preset-unstable-XXXX)
prefix_preset_unstable_only=$(mktemp -d /tmp/cccl-preset-unstable-only-XXXX)

# Default configure + install
default_start_time=$SECONDS
run_command "Configure default" \
  cmake -S . -B build/default -DCMAKE_INSTALL_PREFIX="$prefix_default"
run_command "Install default" \
  cmake --build build/default --target install
default_time=$((SECONDS - default_start_time))

# Preset configure + install
preset_start_time=$SECONDS
CCCL_BUILD_INFIX=preset \
run_command "Configure preset" \
  cmake --preset install -DCMAKE_INSTALL_PREFIX="$prefix_preset"
CCCL_BUILD_INFIX=preset \
run_command "Install preset" \
  cmake --build --preset install --target install
preset_time=$((SECONDS - preset_start_time))

# Preset configure + install-unstable
preset_unstable_start_time=$SECONDS
CCCL_BUILD_INFIX=preset_unstable \
run_command "Configure preset-unstable" \
  cmake --preset install-unstable -DCMAKE_INSTALL_PREFIX="$prefix_preset_unstable"
CCCL_BUILD_INFIX=preset_unstable \
run_command "Install preset-unstable" \
  cmake --build build/preset_unstable/install-unstable --target install
preset_unstable_time=$((SECONDS - preset_unstable_start_time))

# Preset configure + install-unstable-only
preset_unstable_only_start_time=$SECONDS
CCCL_BUILD_INFIX=preset_unstable_only \
run_command "Configure preset-unstable-only" \
  cmake --preset install-unstable-only -DCMAKE_INSTALL_PREFIX="$prefix_preset_unstable_only"
CCCL_BUILD_INFIX=preset_unstable_only \
run_command "Install preset-unstable-only" \
  cmake --build build/preset_unstable_only/install-unstable-only \
        --target install
preset_unstable_only_time=$((SECONDS - preset_unstable_only_start_time))

# Script install
script_start_time=$SECONDS
CCCL_BUILD_INFIX=script \
run_command "Install script" \
  ci/install_cccl.sh "$prefix_script"
script_time=$((SECONDS - script_start_time))

# Compare directories
if ! diff -ruN "$prefix_default" "$prefix_preset" > /tmp/diff_default_preset.log; then
  cat /tmp/diff_default_preset.log
  echo -e "\e[0;31mDefault and preset installations differ\e[0m"
  exit 1
fi
if ! diff -ruN "$prefix_default" "$prefix_script" > /tmp/diff_default_script.log; then
  cat /tmp/diff_default_script.log
  echo -e "\e[0;31mDefault and script installations differ\e[0m"
  exit 1
fi

# Verify unstable-only contains only cudax headers and cmake config
if find "$prefix_preset_unstable_only" -type f | \
    sed "s|$prefix_preset_unstable_only/||" | \
    grep -vE '^include/cuda/experimental/|^lib/cmake/+cudax/|^lib/cmake/+cccl/' | \
    grep -q .; then
  echo -e "\e[0;31minstall-unstable-only contained unexpected files\e[0m" >&2
  exit 1
fi

# Verify install-unstable is union of install and install-unstable-only
prefix_union=$(mktemp -d /tmp/cccl-union-XXXX)
cp -a "$prefix_default/." "$prefix_union/"
cp -a "$prefix_preset_unstable_only/." "$prefix_union/"
if ! diff -ruN "$prefix_union" "$prefix_preset_unstable" > /tmp/diff_union_unstable.log; then
  echo -e "\e[0;31mUnion of install and install-unstable-only did not match install-unstable\e[0m" >&2
  cat /tmp/diff_union_unstable.log
  exit 1
fi

print_time_summary

# Verify times
violations=()
(( default_time > install_time_limit )) && violations+=("default_time=${default_time}s")
(( preset_time > install_time_limit )) && violations+=("preset_time=${preset_time}s")
(( preset_unstable_time > install_time_limit )) && violations+=("preset_unstable_time=${preset_unstable_time}s")
(( preset_unstable_only_time > install_time_limit )) && violations+=("preset_unstable_only_time=${preset_unstable_only_time}s")
(( script_time > install_time_limit )) && violations+=("script_time=${script_time}s")

if (( ${#violations[@]} > 0 )); then
  echo -e "\e[0;31mInstallation time limit (${install_time_limit}s) exceeded by: ${violations[*]}\e[0m" >&2
  exit 1
fi

# Clean up
rm -rf "$prefix_default" "$prefix_preset" "$prefix_script" \
       "$prefix_preset_unstable" "$prefix_preset_unstable_only" \
       "$prefix_union"

echo -e "\e[0;32mAll installation tests passed successfully.\e[0m"
