#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build one CCCL revision and preserve the unmodified compiler traces. Analysis
# is a separate command, so CI can archive the dataset before processing it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project=""
output=""
declare -a targets=()
while (($#)); do
  case "$1" in
    -project) project="$2"; shift 2 ;;
    -target) targets+=("$2"); shift 2 ;;
    -output) output="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 -project matx|pytorch|rapids [-target RAPIDS_LIB] [-output DIRECTORY]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${project}" in
  matx|pytorch|rapids) ;;
  *) echo "Expected -project matx, pytorch, or rapids" >&2; exit 2 ;;
esac
if [[ "${project}" != rapids && ${#targets[@]} -ne 0 ]]; then
  echo "-target is only supported for RAPIDS" >&2
  exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CCCL_BUILD_INFIX="${CCCL_BUILD_INFIX:-compile-time-third-party}"
workdir="${repo_root}/build/${CCCL_BUILD_INFIX}/${project}"
output="${output:-${workdir}/compile_time}"
mkdir -p "${output}"
output="$(cd "${output}" && pwd)"
# A run must not silently reuse a previous run's dataset.
if [[ -e "${output}/raw_traces" || -e "${output}/dataset.json" ]]; then
  echo "Dataset already exists: ${output}; choose a fresh output directory" >&2
  exit 2
fi
export CCCL_COMPILE_TIME_BENCH=1
export SCCACHE_DISABLE=1
unset CCCL_TAG CMAKE_CUDA_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER CMAKE_C_COMPILER_LAUNCHER
available_processing_units="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
export PARALLEL_LEVEL="${PARALLEL_LEVEL:-${available_processing_units}}"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
trace_repo_root="${workdir}"
phase=""
declare -a inputs=()
declare -a sources=()

rapids_source_root() {
  case "$1" in
    cudf_kafka) printf '%s\n' "${HOME}/cudf/cpp/libcudf_kafka" ;;
    wholegraph) printf '%s\n' "${HOME}/cugraph-gnn" ;;
    *) printf '%s\n' "${HOME}/$1" ;;
  esac
}

collect_traces() {
  declare -a collect_args=()
  local input
  for input in "${inputs[@]}"; do
    # A failed configure may not have created every selected project yet.
    [[ -d "${input#*=}" ]] && collect_args+=(--input "${input}")
  done
  if ((${#collect_args[@]} == 0)); then
    echo "No build directories available for trace collection" >&2
    return 1
  fi
  if [[ "${project}" == rapids ]]; then
    collect_args+=(--state "${output}/collection-state.json")
  fi
  python3 ci/compile_time/collect_traces.py \
    "${collect_args[@]}" --output "${output}/raw_traces${phase:+/${phase}}"
}

finish() {
  local build_rc="$?"
  trap - EXIT
  set +e
  collect_traces
  local collect_rc="$?"
  declare -a source_args=()
  local source
  for source in "${sources[@]}"; do
    source_args+=(--source "${source}")
  done
  for source in "${inputs[@]}"; do
    source_args+=(--build-tree "${source}")
  done
  python3 -m ci.compile_time.write_trace_metadata \
    --output "${output}/dataset.json" --trace-dir "${output}/raw_traces" \
    --project "${project}" --cccl-root "${repo_root}" \
    --repo-root "${trace_repo_root}" --started-at "${started_at}" \
    --build-exit-code "${build_rc}" --collection-exit-code "${collect_rc}" \
    "${source_args[@]}"
  local metadata_rc="$?"
  ((build_rc != 0)) && exit "${build_rc}"
  ((collect_rc != 0)) && exit "${collect_rc}"
  exit "${metadata_rc}"
}

if [[ "${project}" == rapids ]]; then
  trace_repo_root="${HOME}"
  if ((${#targets[@]} == 0)); then
    mapfile -t targets < <(yq -r '.repos[].cpp[].name' "${PROJECT_MANIFEST_YML:-/opt/rapids-build-utils/manifest.yaml}")
  fi
  ((${#targets[@]} > 0)) || { echo "No RAPIDS targets selected" >&2; exit 2; }
  export RAPIDS_LIBS="${targets[*]}"
  for target in "${targets[@]}"; do
    inputs+=("${target}=$(rapids_source_root "${target}")")
    sources+=("${target}=$(rapids_source_root "${target}")")
  done
  phase=no-tests
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  # shellcheck source=ci/rapids/post-create-command.sh
  source ci/rapids/post-create-command.sh
  uninstall-all -j -qqq
  clean-all -j
  # Snapshot the first build before reconfiguration can overwrite its traces.
  for tests in OFF ON; do
    export RAPIDS_ENABLE_TESTS="${tests}"
    _apply_manifest_modifications
    for target in "${targets[@]}"; do
      configure-"${target}"-cpp
      build-"${target}"-cpp -j"${PARALLEL_LEVEL}"
    done
    if [[ "${tests}" == OFF ]]; then
      collect_traces
      phase=with-tests
    fi
  done
else
  inputs=("${project}=${workdir}/build")
  if [[ "${project}" == matx ]]; then
    sources=("matx=${workdir}/MatX")
  else
    sources=("pytorch=${workdir}/pytorch")
  fi
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "ci/${project}/build_${project}.sh"
fi
