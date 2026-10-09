#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${ci_dir}/.." && pwd)"
tool_dir="${repo_root}/ci/compile_time"

default_preset="all-dev"
preset="${default_preset}"
skip_configure=0
skip_build=0
prepare_perfetto=1
run_ctadvisor=0
write_tu_csv=1
explicit_tu_csv=0
tu_csv=""
perfetto_output_dir=""
baseline_ref=""
baseline_worktree=""
current_worktree=""
comparison_worktrees=""
comparison_build_started=0
build_order="auto"
max_detail_len=180
cloc_processes=0

declare -a build_targets=()
declare -a event_args=()
declare -a common_args=()
declare -a default_build_targets=(
  "cub.headers.base"
  "thrust.cpp.cuda.headers.base"
  "libcudacxx.test.public_headers"
)
declare -a bench_cmake_args=(
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DCMAKE_CUDA_COMPILER_LAUNCHER="
  "-DCMAKE_CXX_COMPILER_LAUNCHER="
  "-DCMAKE_C_COMPILER_LAUNCHER="
  "-DCCCL_ENABLE_TESTING=OFF"
  "-DCCCL_ENABLE_EXAMPLES=OFF"
  "-DCCCL_ENABLE_BENCHMARKS=OFF"
  "-DCCCL_ENABLE_C_PARALLEL=OFF"
  "-DCCCL_ENABLE_C_EXPERIMENTAL_STF=OFF"
  "-DCUB_ENABLE_TESTING=OFF"
  "-DCUB_ENABLE_EXAMPLES=OFF"
  "-DTHRUST_ENABLE_TESTING=OFF"
  "-DTHRUST_ENABLE_EXAMPLES=OFF"
  "-DTHRUST_MULTICONFIG_WORKLOAD=SMALL"
  "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_OMP=OFF"
  "-DTHRUST_MULTICONFIG_ENABLE_SYSTEM_TBB=OFF"
  "-Dcudax_ENABLE_TESTING=OFF"
  "-Dcudax_ENABLE_EXAMPLES=OFF"
  "-Dcudax_ENABLE_CUDASTF=OFF"
  "-Dcudax_ENABLE_CUFILE=OFF"
  "-DCCCL_COMPILE_TIME_SAVE_PREPROCESSED_TUS=ON"
  "-DCCCL_COMPILE_TIME_GENERATE_DEVICE_TIME_TRACES=ON"
)

usage() {
  cat <<'EOF'
Usage: ci/build_compile_time_bench.sh [options] [-- <event summary args>]

Build options:
  -preset <name>              CMake configure preset (default: all-dev)
  -cmake-options <args>       Extra CMake configure options handled by ci/build_common.sh
  -target <name>              Build target; repeatable
                              (default: public include-check target set)
  -baseline-ref <commit-ish>  Build this commit-ish as a comparison baseline
  -build-order <order>        auto, current-first, or baseline-first (default: auto)
                              auto alternates CI run attempts; baseline-first locally
  -skip-configure             Do not run cmake configure
  -skip-build                 Do not run cmake --build
  -cuda, -cxx, -std, -arch    Common compiler/standard/arch options from ci/build_common.sh

Summary options:
  -tu-csv <path>              Generated-TU summary CSV
                              (default: <preset-build-dir>/compile_time/tu_summary.csv)
  -no-tu-csv                  Do not write the generated-TU summary CSV
  -prepare-perfetto           Prepare Perfetto-friendly trace copies (default)
  -no-prepare-perfetto        Skip Perfetto trace preparation
  -perfetto-output <path>     Perfetto trace output directory
                              (default: <preset-build-dir>/compile_time/perfetto_traces)
  -max-detail-len <n>         Max promoted detail length for Perfetto traces (default: 180)
  -cloc-processes <n>         cloc process count for generated-TU summary CSV
  -ctadvisor                  Print ctadvisor report for raw traces

Event summary args:
  Arguments after '--' are forwarded to ci/compile_time/summarize_events.py
  after the raw trace directory. If omitted, the default summary is:
    -f file-processing -e -n 15
  Pass --slices <json-file> after '--' to emit multiple event report slices
  and an event_reports/summary.json manifest.
  With -baseline-ref, comparison-only options such as --threshold <seconds>
  may also be passed after '--'. Baseline raw traces are preserved under
  <preset-build-dir>/compile_time/baseline_raw_traces.

Examples:
  ci/build_compile_time_bench.sh
  ci/build_compile_time_bench.sh -target cudax.headers.basic.no_stf -- -f scanning-function-body -i -n 20
  ci/build_compile_time_bench.sh -preset cub -target cub.headers.base -- -f template-instantiation -e -n 15
  ci/build_compile_time_bench.sh -baseline-ref origin/main -- -f file-processing -e --threshold 0.001
EOF
}

status() { echo "[compile-time-bench] $*" >&2; }

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null \
    || { echo "error: ${command_name} not found" >&2; exit 1; }
}

cleanup_baseline_worktree() {
  # Keep measured partial traces on failure, while preventing an older
  # successful summary from being uploaded for this invocation.
  if (( comparison_build_started )); then
    if [[ ! -d "${trace_dir}" && -d "${current_measurement_trace_dir}" ]]; then
      mkdir -p "${report_root}"
      cp -a "${current_measurement_trace_dir}" "${trace_dir}" || :
    fi
    if [[ ! -d "${baseline_artifact_trace_dir}" && -d "${baseline_trace_dir}" ]]; then
      mkdir -p "${report_root}"
      cp -a "${baseline_trace_dir}" "${baseline_artifact_trace_dir}" || :
    fi
  fi
  if [[ -n "${baseline_worktree}" && -d "${baseline_worktree}" ]]; then
    git -C "${repo_root}" worktree remove --force "${baseline_worktree}" >/dev/null 2>&1 \
      || rm -rf "${baseline_worktree}"
  fi
  if [[ -n "${current_worktree}" && -d "${current_worktree}" ]]; then
    git -C "${repo_root}" worktree remove --force "${current_worktree}" >/dev/null 2>&1 \
      || rm -rf "${current_worktree}"
  fi
  if [[ -n "${comparison_worktrees}" && -d "${comparison_worktrees}" ]]; then
    rmdir "${comparison_worktrees}" 2>/dev/null || :
  fi
}

cleanup_baseline_worktree_and_exit() {
  local exit_code="$1"
  trap - EXIT HUP INT TERM
  cleanup_baseline_worktree
  exit "${exit_code}"
}

install_baseline_cleanup_traps() {
  trap cleanup_baseline_worktree EXIT
  trap 'cleanup_baseline_worktree_and_exit 129' HUP
  trap 'cleanup_baseline_worktree_and_exit 130' INT
  trap 'cleanup_baseline_worktree_and_exit 143' TERM
}

overlay_current_bench_file() {
  local rel_path="$1"
  local worktree="$2"
  mkdir -p "$(dirname "${worktree}/${rel_path}")"
  rm -rf "${worktree:?}/${rel_path}"
  ln -s "${repo_root}/${rel_path}" "${worktree}/${rel_path}"
}

overlay_current_bench_logic() {
  overlay_current_bench_file "ci" "$1"
  overlay_current_bench_file "CMakePresets.json" "$1"
  overlay_current_bench_file "cmake/CCCLGenerateHeaderTests.cmake" "$1"
}

for arg in "$@"; do
  case "$arg" in
    -h|-help|--help) usage; exit 0 ;;
    *) ;;
  esac
done

new_args="$("${ci_dir}/util/extract_switches.sh" \
  -skip-configure \
  -skip-build \
  -no-tu-csv \
  -prepare-perfetto \
  -no-prepare-perfetto \
  -ctadvisor \
  -- "$@")"

declare -a new_args="(${new_args})"
set -- "${new_args[@]}"
while true; do
  case "$1" in
    -skip-configure) skip_configure=1; shift ;;
    -skip-build) skip_build=1; shift ;;
    -no-tu-csv) write_tu_csv=0; shift ;;
    -prepare-perfetto) prepare_perfetto=1; shift ;;
    -no-prepare-perfetto) prepare_perfetto=0; shift ;;
    -ctadvisor) run_ctadvisor=1; shift ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

while (($#)); do
  case "$1" in
    -preset) preset="$2"; shift 2 ;;
    -target) build_targets+=("$2"); shift 2 ;;
    -baseline-ref) baseline_ref="$2"; shift 2 ;;
    -build-order) build_order="$2"; shift 2 ;;
    -tu-csv) explicit_tu_csv=1; write_tu_csv=1; tu_csv="$2"; shift 2 ;;
    -perfetto-output) perfetto_output_dir="$2"; shift 2 ;;
    -max-detail-len) max_detail_len="$2"; shift 2 ;;
    -cloc-processes) cloc_processes="$2"; shift 2 ;;
    --) shift; event_args+=("$@"); break ;;
    *) common_args+=("$1"); shift ;;
  esac
done

set -- "${common_args[@]}"
# shellcheck source=ci/build_common.sh
source "${ci_dir}/build_common.sh"

current_build_root="${BUILD_ROOT}"
current_build_dir="${BUILD_DIR}"

build_root_for_source() {
  local source_root="$1"
  mkdir -p "${source_root}/build"
  (cd "${source_root}/build" && pwd)
}

build_dir_for_source() {
  local source_root="$1"
  local build_root="$2"
  local build_dir="${build_root}/${CCCL_BUILD_INFIX}"
  mkdir -p "${build_dir}"
  readlink -f "${build_dir}"
}

run_bench_build() {
  local source_root="$1"
  local build_name="$2"
  local build_root="$3"
  local build_dir="$4"
  local phase="${5:-both}"

  (
    # build_common.sh keeps the active build tree in these globals; override
    # them only inside this subshell so current-tree state is restored after
    # each build.
    # shellcheck disable=SC2030
    BUILD_ROOT="${build_root}"
    # shellcheck disable=SC2030
    BUILD_DIR="${build_dir}"
    # CMake's compiler checks can inherit these even when cache entries are
    # empty. A cache hit must never substitute for a measured compilation.
    export CMAKE_C_COMPILER_LAUNCHER=""
    export CMAKE_CXX_COMPILER_LAUNCHER=""
    export CMAKE_CUDA_COMPILER_LAUNCHER=""
    cd "${source_root}/ci"

    if [[ "${phase}" != build ]] && (( ! skip_configure )); then
      status "Configuring preset '${preset}' with compile-time bench instrumentation (${build_name})..."
      configure_preset "${build_name}" "${preset}" "${bench_cmake_args[@]}"
    fi

    if [[ "${phase}" != configure ]] && (( ! skip_build )); then
      status "Building target(s) (${build_name}): ${build_targets[*]}"
      build_preset "${build_name}" "${preset}" --target "${build_targets[@]}"
    fi
  )
}

prepare_perfetto_traces() {
  local input_dir="$1"
  local output_dir="$2"
  local trace_repo_root="$3"
  local label="$4"

  status "Preparing Perfetto trace copies (${label})..."
  rm -rf "${output_dir}"
  "${tool_dir}/prepare_traces.py" \
    --input "${input_dir}" \
    --output "${output_dir}" \
    --repo-root "${trace_repo_root}" \
    --max-detail-len "${max_detail_len}"
  status "Perfetto traces (${label}): ${output_dir}"
}

if ((${#build_targets[@]} == 0)); then
  build_targets=("${default_build_targets[@]}")
fi

preset_build_dir="${current_build_dir}/${preset}"
baseline_trace_dir=""
current_trace_repo_root="${repo_root}"
generated_tu_build_dir="${preset_build_dir}"

case "${build_order}" in
  auto|current-first|baseline-first) ;;
  *) echo "error: -build-order must be auto, current-first, or baseline-first" >&2; exit 1 ;;
esac

if [[ -n "${baseline_ref}" ]]; then
  if (( skip_configure || skip_build )); then
    echo "error: -baseline-ref requires fresh configure/build; replay existing traces with summarize_events.py" >&2
    exit 1
  fi
  baseline_commit="$(git -C "${repo_root}" rev-parse --verify "${baseline_ref}^{commit}")"
  current_commit="$(git -C "${repo_root}" rev-parse HEAD)"
  comparison_worktrees="$(mktemp -d "${current_build_dir}/compile-time-pair.XXXXXX")"
  baseline_worktree="${comparison_worktrees}/source-a"
  current_worktree="${comparison_worktrees}/source-b"
  install_baseline_cleanup_traps
  status "Creating baseline worktree for ${baseline_ref} (${baseline_commit})..."
  git -C "${repo_root}" worktree add --detach "${baseline_worktree}" "${baseline_commit}" >/dev/null
  git -C "${repo_root}" worktree add --detach "${current_worktree}" "${current_commit}" >/dev/null
  # Preserve tracked staged/unstaged current changes without touching the
  # user's source tree. New files are copied as well, including local headers.
  git -C "${repo_root}" diff --no-ext-diff --no-textconv --binary HEAD \
    | git -C "${current_worktree}" apply --allow-empty
  while IFS= read -r -d '' rel_path; do
    # Never copy the build tree containing these temporary worktrees into
    # itself, even if a local .gitignore exposes build outputs.
    case "${rel_path}" in
      build/*) continue ;;
      *) ;;
    esac
    mkdir -p "$(dirname "${current_worktree}/${rel_path}")"
    cp -a "${repo_root}/${rel_path}" "${current_worktree}/${rel_path}"
  done < <(git -C "${repo_root}" ls-files --others --exclude-standard -z)
  overlay_current_bench_logic "${baseline_worktree}"
  overlay_current_bench_logic "${current_worktree}"

  baseline_build_root="$(build_root_for_source "${baseline_worktree}")"
  baseline_build_dir="$(build_dir_for_source "${baseline_worktree}" "${baseline_build_root}")"
  baseline_trace_dir="${baseline_build_dir}/${preset}/compile_time/raw_traces"
  current_measurement_build_root="$(build_root_for_source "${current_worktree}")"
  current_measurement_build_dir="$(build_dir_for_source "${current_worktree}" "${current_measurement_build_root}")"
  current_measurement_trace_dir="${current_measurement_build_dir}/${preset}/compile_time/raw_traces"
  current_trace_repo_root="${current_worktree}"
  generated_tu_build_dir="${current_measurement_build_dir}/${preset}"
  if [[ "${build_order}" == auto ]]; then
    if [[ -z "${GITHUB_RUN_NUMBER:-}" ]]; then
      build_order="baseline-first"
    elif (( (GITHUB_RUN_NUMBER + ${GITHUB_RUN_ATTEMPT:-1}) % 2 )); then
      build_order="current-first"
    else
      build_order="baseline-first"
    fi
  fi
fi

report_root="${preset_build_dir}/compile_time"
trace_dir="${report_root}/raw_traces"
baseline_artifact_trace_dir="${report_root}/baseline_raw_traces"
event_output_dir="${report_root}/event_reports"
tu_csv="${tu_csv:-${report_root}/tu_summary.csv}"
perfetto_output_dir="${perfetto_output_dir:-${report_root}/perfetto_traces}"
if [[ -n "${baseline_ref}" && "${explicit_tu_csv}" -eq 0 ]]; then
  write_tu_csv=0
fi

require_command python3
if (( write_tu_csv )); then
  require_command cloc
fi
if (( run_ctadvisor )); then
  require_command ctadvisor
fi

if [[ -n "${baseline_ref}" ]]; then
  mkdir -p "${report_root}"
  rm -rf "${event_output_dir}" "${trace_dir}" "${baseline_artifact_trace_dir}" "${perfetto_output_dir}"
  python3 - "${report_root}/build_pair.json" "${baseline_commit}" "${current_commit}" "${build_order}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "baseline_commit": sys.argv[2], "current_commit": sys.argv[3],
    "build_order": sys.argv[4], "warmup_tus_per_side": 1,
    "current_includes_working_tree_changes": True,
}, indent=2) + "\n")
PY
  status "Comparison build order: ${build_order}"
  declare -a sides=(current baseline)
  if [[ "${build_order}" == baseline-first ]]; then
    sides=(baseline current)
  fi
  for phase in configure warmup build; do
    if [[ "${phase}" == build ]]; then
      comparison_build_started=1
    fi
    for side in "${sides[@]}"; do
      if [[ "${side}" == current ]]; then
        source_root="${current_worktree}"
        side_build_root="${current_measurement_build_root}"
        side_build_dir="${current_measurement_build_dir}"
        peer_build_dir="${baseline_build_dir}"
      else
        source_root="${baseline_worktree}"
        side_build_root="${baseline_build_root}"
        side_build_dir="${baseline_build_dir}"
        peer_build_dir="${current_measurement_build_dir}"
      fi
      if [[ "${phase}" == warmup ]]; then
        python3 "${tool_dir}/warmup.py" "${side_build_dir}/${preset}" --peer-build-dir "${peer_build_dir}/${preset}"
      else
        run_bench_build "${source_root}" "Compile-time Bench (${side})" "${side_build_root}" "${side_build_dir}" "${phase}"
      fi
    done
    if $CONFIGURE_ONLY; then
      exit 0
    fi
  done
  cp -a "${current_measurement_trace_dir}" "${trace_dir}"
else
  run_bench_build "${repo_root}" "Compile-time Bench (current)" "${current_build_root}" "${current_build_dir}"
  if $CONFIGURE_ONLY; then
    exit 0
  fi
fi

shopt -s nullglob globstar
trace_paths=("${trace_dir}"/**/*.json)
(( ${#trace_paths[@]} > 0 )) \
  || { echo "error: no device-time-trace JSON files found under ${trace_dir}" >&2; exit 1; }
if [[ -n "${baseline_ref}" ]]; then
  baseline_trace_paths=("${baseline_trace_dir}"/**/*.json)
  (( ${#baseline_trace_paths[@]} > 0 )) \
    || { echo "error: no device-time-trace JSON files found under ${baseline_trace_dir}" >&2; exit 1; }

  status "Copying baseline raw traces to ${baseline_artifact_trace_dir}..."
  rm -rf "${baseline_artifact_trace_dir}"
  mkdir -p "${baseline_artifact_trace_dir}"
  cp -a "${baseline_trace_dir}/." "${baseline_artifact_trace_dir}/"
fi

if (( prepare_perfetto )); then
  if [[ -n "${baseline_ref}" ]]; then
    rm -rf "${perfetto_output_dir}"
    prepare_perfetto_traces \
      "${trace_dir}" \
      "${perfetto_output_dir}/current" \
      "${current_trace_repo_root}" \
      "current"
    prepare_perfetto_traces \
      "${baseline_trace_dir}" \
      "${perfetto_output_dir}/baseline" \
      "${baseline_worktree}" \
      "baseline"
  else
    prepare_perfetto_traces \
      "${trace_dir}" \
      "${perfetto_output_dir}" \
      "${repo_root}" \
      "current"
  fi
fi

if (( write_tu_csv )); then
  status "Writing generated-TU summary CSV..."
  "${tool_dir}/summarize_tus.py" \
    --build-dir "${generated_tu_build_dir}" \
    --output-csv "${tu_csv}" \
    --cloc-processes "${cloc_processes}"
  status "Generated-TU summary CSV: ${tu_csv}"
fi

declare -a summary_args=("${event_args[@]}")
if ((${#summary_args[@]} == 0)); then
  summary_args=(-f file-processing -e -n 15)
fi
summary_args+=(-o "${event_output_dir}")
if [[ -n "${baseline_ref}" ]]; then
  summary_args+=(
    --baseline-dir "${baseline_trace_dir}"
    --baseline-repo-root "${baseline_worktree}"
    --repo-root "${current_trace_repo_root}"
    --stability-filter
    --build-metadata "${report_root}/build_pair.json"
  )
fi
status "Writing event summary..."
"${tool_dir}/summarize_events.py" "${trace_dir}" "${summary_args[@]}"

if (( run_ctadvisor )); then
  status "Running ctadvisor over ${#trace_paths[@]} trace(s)..."
  ctadvisor \
    --trace-file-path "${trace_dir}" \
    --header-advisor-entries 20 \
    --thread-number "$(nproc --all --ignore=2)"
fi
