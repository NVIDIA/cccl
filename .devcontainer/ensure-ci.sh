#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

readonly source_root="/home/coder/cccl"
readonly ci_root="/home/coder/cccl-ci"
readonly source_file="${source_root}/.devcontainer/.ci-source"
readonly repository="https://github.com/NVIDIA/cccl.git"

# CI jobs bind-mount the called workflow's implementation here. That checkout
# intentionally takes precedence over the revision recorded by the last
# devcontainer promotion.
if [[ -x "${ci_root}/ci/util/build_and_test_targets.sh" ]]; then
    exit 0
fi

if [[ -e "${ci_root}" ]]; then
    echo "error: ${ci_root} exists but is not a CCCL CI checkout" >&2
    exit 1
fi

if [[ ! -f "${source_file}" || -L "${source_file}" ]]; then
    echo "error: missing CCCL CI revision file: ${source_file}" >&2
    exit 1
fi

mapfile -t ci_source < "${source_file}"
if (( ${#ci_source[@]} != 1 )) || [[ ! "${ci_source[0]}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: invalid CCCL CI revision in ${source_file}" >&2
    exit 1
fi
readonly ci_revision="${ci_source[0]}"

ci_tmp="$(mktemp -d "${ci_root}.tmp.XXXXXX")"
cleanup() {
    if [[ -n "${ci_tmp:-}" ]]; then
        rm -rf -- "${ci_tmp}"
    fi
}
trap cleanup EXIT

echo "Cloning CCCL CI revision ${ci_revision} into ${ci_root}..."
git -C "${ci_tmp}" init --quiet
git -C "${ci_tmp}" remote add origin "${repository}"
git -C "${ci_tmp}" fetch --quiet --depth 1 origin "${ci_revision}"
git -C "${ci_tmp}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
mv "${ci_tmp}" "${ci_root}"
ci_tmp=""
