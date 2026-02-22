#!/bin/bash

set -euo pipefail

die() {
  local message="$1"
  local code="${2:-2}"
  echo "${message}" >&2
  exit "${code}"
}

usage() {
  cat <<EOF
Usage: $0 <base-path> <test-path> [filter1 [filter2 ...]] \
  [--arch "<arch>"] \
  [--nvbench-args "<args>"] \
  [--nvbench-compare-args "<args>"]

Compare CUB benchmark performance between two checked-out CCCL trees.

Arguments:
  <base-path>  Path to baseline CCCL source tree.
  <test-path>  Path to comparison CCCL source tree.
  [filterN]    Optional regex filters matched against benchmark target names.

Environment:
  CCCL_BENCH_ARTIFACT_ROOT   Root directory for outputs.
                             Default: "\$(pwd)/bench-artifacts"
  CCCL_BENCH_ARTIFACT_TAG    Optional explicit artifact directory name.
  CCCL_BENCH_BASE_LABEL      Optional label used in auto-generated artifact names.
  CCCL_BENCH_TEST_LABEL      Optional label used in auto-generated artifact names.
  CCCL_BENCH_BASE_BUILD_DIR  Optional preconfigured build tree for base path.
  CCCL_BENCH_TEST_BUILD_DIR  Optional preconfigured build tree for test path.
                             If either is set, both must be set.
  CCCL_BENCH_BUILD_ROOT      Root directory for generated build trees.
                             Default: "/tmp/cccl-bench-builds"
EOF
}

sanitize_label() {
  local label="$1"
  label="${label//[^a-zA-Z0-9._-]/_}"
  label="${label#_}"
  label="${label%_}"
  label="${label:-unknown}"
  printf "%s" "${label}"
}

resolve_repo_label() {
  local repo_path="$1"
  local branch=""
  branch="$(git -C "${repo_path}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -n "${branch}" && "${branch}" != "HEAD" ]]; then
    printf "%s" "${branch}"
    return 0
  fi
  local short_sha=""
  short_sha="$(git -C "${repo_path}" rev-parse --short=12 HEAD 2>/dev/null || true)"
  if [[ -n "${short_sha}" ]]; then
    printf "%s" "${short_sha}"
    return 0
  fi
  basename "${repo_path}"
}

validate_repo_path() {
  local repo_path="$1"
  if [[ ! -d "${repo_path}" ]]; then
    die "Path does not exist: ${repo_path}"
  fi
  if [[ ! -f "${repo_path}/cccl-version.json" ]]; then
    die "Path is not a CCCL source tree: ${repo_path}"
  fi
}

validate_filters() {
  local filter=""
  for filter in "${FILTERS[@]}"; do
    grep -Eq -- "${filter}" <<< "" >/dev/null 2>&1 || {
      [[ "$?" -eq 1 ]] || die "Invalid regex filter: ${filter}"
    }
  done
}

print_shell_command() {
  local arg=""
  # Print a shell-escaped command line so it can be copied and re-run directly.
  printf '$'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

print_shell_command_with_env() {
  local env_assignment="$1"
  shift
  local arg=""
  # Include env assignment for reproducibility of compare invocations.
  printf '$ %q' "${env_assignment}"
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

configure_build_tree() {
  local src_path="$1"
  local build_path="$2"
  local side="$3"
  local log_path="$4"
  local target_arch="$5"
  local -a cmake_cmd
  cmake_cmd=(cmake --preset "cub-benchmark" -S "${src_path}" -B "${build_path}")
  if [[ -n "${target_arch}" ]]; then
    cmake_cmd+=("-DCMAKE_CUDA_ARCHITECTURES=${target_arch}")
  fi
  run_grouped_logged_command "[configure:${side}]" "${log_path}" "${cmake_cmd[@]}"
}

validate_build_dir() {
  local build_path="$1"
  local label="$2"
  if [[ ! -d "${build_path}" ]]; then
    die "Configured ${label} build tree does not exist: ${build_path}"
  fi
  if [[ ! -f "${build_path}/build.ninja" ]]; then
    die "Configured ${label} build tree is missing build.ninja: ${build_path}"
  fi
}

list_all_benchmark_targets() {
  local build_path="$1"
  ninja -C "${build_path}" -t targets all \
    | awk -F':' '/^.*\.bench\./ { print $1 }' \
    | sort -u
}

target_matches_filters() {
  local target="$1"
  local filter=""
  if [[ "${#FILTERS[@]}" -eq 0 ]]; then
    return 0
  fi
  for filter in "${FILTERS[@]}"; do
    if grep -Eq -- "${filter}" <<< "${target}"; then
      return 0
    fi
  done
  return 1
}

resolve_compare_script() {
  local build_path="$1"
  local candidate="${build_path}/_deps/nvbench-src/scripts/nvbench_compare.py"
  if [[ -f "${candidate}" ]]; then
    printf "%s" "${candidate}"
    return 0
  fi
  return 1
}

run_grouped_logged_command() {
  local label="$1"
  local log_path="$2"
  shift 2
  local started_at=0
  local elapsed_s=0
  local rc=0
  local command_rc=0
  local tee_rc=0

  echo "::group::${label}"
  print_shell_command "$@"
  started_at="${SECONDS}"
  if "$@" 2>&1 | tee "${log_path}"; then
    rc=0
  else
    command_rc="${PIPESTATUS[0]}"
    tee_rc="${PIPESTATUS[1]}"
    if [[ "${command_rc}" -ne 0 ]]; then
      rc="${command_rc}"
    else
      rc="${tee_rc}"
    fi
  fi
  elapsed_s=$((SECONDS - started_at))
  echo "::endgroup::"
  if [[ "${rc}" -eq 0 ]]; then
    echo "${label} completed in ${elapsed_s}s"
  else
    echo "${label} failed in ${elapsed_s}s (rc=${rc})"
  fi
  return "${rc}"
}

run_compare_target() {
  local target="$1"
  local compare_script="$2"
  local compare_script_dir="$3"
  local base_json="$4"
  local test_json="$5"
  local compare_out="$6"
  local compare_log="$7"

  local label="[compare] ${target}"
  local started_at=0
  local elapsed_s=0
  local rc=0
  local compare_pythonpath="${compare_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  local -a compare_cmd
  compare_cmd=(python3 "${compare_script}" "${NVBENCH_COMPARE_ARGS[@]}" "${base_json}" "${test_json}")

  : > "${compare_log}"
  echo "::group::${label}"
  print_shell_command_with_env "PYTHONPATH=${compare_pythonpath}" "${compare_cmd[@]}"
  started_at="${SECONDS}"
  if PYTHONPATH="${compare_pythonpath}" \
    "${compare_cmd[@]}" \
    > >(tee "${compare_out}" | tee -a "${compare_log}") \
    2> >(tee -a "${compare_log}" >&2); then
    rc=0
  else
    rc=$?
  fi
  elapsed_s=$((SECONDS - started_at))
  echo "::endgroup::"
  if [[ "${rc}" -eq 0 ]]; then
    echo "${label} completed in ${elapsed_s}s"
  else
    echo "${label} failed in ${elapsed_s}s (rc=${rc})"
  fi
  return "${rc}"
}

parse_quoted_args_to_nul_file() {
  local quoted_args="$1"
  local output_file="$2"
  local option_name="$3"

  QUOTED_ARGS="${quoted_args}" OPTION_NAME="${option_name}" python3 - "${output_file}" <<'PY'
import os
import shlex
import sys

quoted = os.environ.get("QUOTED_ARGS", "")
option_name = os.environ.get("OPTION_NAME", "--args")
try:
    parsed = shlex.split(quoted)
except ValueError as exc:
    print(f"Invalid {option_name}: {exc}", file=sys.stderr)
    raise SystemExit(2)

with open(sys.argv[1], "wb") as out_file:
    for arg in parsed:
        out_file.write(arg.encode("utf-8"))
        out_file.write(b"\0")
PY
}

parse_nvbench_run_args() {
  local quoted_args="$1"
  local parsed_nvbench_args_file=""

  NVBENCH_RUN_ARGS=()
  if [[ -z "${quoted_args}" ]]; then
    return 0
  fi

  parsed_nvbench_args_file="$(mktemp "/tmp/cccl-nvbench-args-XXXXXX")"
  if ! parse_quoted_args_to_nul_file "${quoted_args}" "${parsed_nvbench_args_file}" "--nvbench-args"; then
    rm -f "${parsed_nvbench_args_file}"
    return 2
  fi
  mapfile -d '' -t NVBENCH_RUN_ARGS < "${parsed_nvbench_args_file}"
  rm -f "${parsed_nvbench_args_file}"
}

parse_nvbench_compare_args() {
  local quoted_args="$1"
  local parsed_nvbench_compare_args_file=""

  NVBENCH_COMPARE_ARGS=()
  if [[ -z "${quoted_args}" ]]; then
    return 0
  fi

  parsed_nvbench_compare_args_file="$(mktemp "/tmp/cccl-nvbench-compare-args-XXXXXX")"
  if ! parse_quoted_args_to_nul_file "${quoted_args}" "${parsed_nvbench_compare_args_file}" "--nvbench-compare-args"; then
    rm -f "${parsed_nvbench_compare_args_file}"
    return 2
  fi
  mapfile -d '' -t NVBENCH_COMPARE_ARGS < "${parsed_nvbench_compare_args_file}"
  rm -f "${parsed_nvbench_compare_args_file}"
}

run_target_for_side() {
  local side="$1"
  local build_path="$2"
  local target="$3"
  local json_path="$4"
  local md_path="$5"
  local run_log="$6"
  local binary_path="${build_path}/bin/${target}"
  local -a bench_cmd

  if [[ ! -x "${binary_path}" ]]; then
    echo "Benchmark binary missing: ${binary_path}" >&2
    return 127
  fi

  bench_cmd=(
    "${binary_path}"
    -d 0
    "${NVBENCH_RUN_ARGS[@]}"
    --json "${json_path}"
    --md "${md_path}"
  )

  if run_grouped_logged_command \
    "[run:${side}] ${target}" \
    "${run_log}" \
    "${bench_cmd[@]}"; then
    return 0
  else
    return $?
  fi
}

select_targets() {
  local base_build_path="$1"
  local test_build_path="$2"
  local -n selected_targets_ref="$3"
  local -a base_targets
  local -a test_targets
  local -a common_targets
  local target=""

  mapfile -t base_targets < <(list_all_benchmark_targets "${base_build_path}")
  mapfile -t test_targets < <(list_all_benchmark_targets "${test_build_path}")

  if [[ "${#base_targets[@]}" -eq 0 ]]; then
    die "No CUB benchmark targets were found in base build tree." 1
  fi
  if [[ "${#test_targets[@]}" -eq 0 ]]; then
    die "No CUB benchmark targets were found in test build tree." 1
  fi

  mapfile -t common_targets < <(
    comm -12 \
      <(printf "%s\n" "${base_targets[@]}" | sort -u) \
      <(printf "%s\n" "${test_targets[@]}" | sort -u)
  )

  selected_targets_ref=()
  for target in "${common_targets[@]}"; do
    [[ -n "${target}" ]] || continue
    if target_matches_filters "${target}"; then
      selected_targets_ref+=("${target}")
    fi
  done

  if [[ "${#selected_targets_ref[@]}" -eq 0 ]]; then
    die "No benchmark targets matched the supplied filters." 1
  fi
}

write_summary() {
  local summary_file="$1"
  local target=""
  local compare_report_file=""
  local reports_emitted=0

  {
    echo "# CUB Benchmark Comparison Summary"
    echo
    echo "- Timestamp (UTC): ${timestamp}"
    echo "- Base label: ${base_label_raw}"
    echo "- Test label: ${test_label_raw}"
    echo "- Base source path: \`${BASE_PATH}\`"
    echo "- Test source path: \`${TEST_PATH}\`"
    echo "- Base build dir: \`${base_build_dir}\`"
    echo "- Test build dir: \`${test_build_dir}\`"
    echo "- Selected targets: ${#selected_targets[@]}"
    echo "- Comparisons attempted: ${compares_attempted}"
    echo "- Comparisons succeeded (nvbench_compare exit 0): ${compares_succeeded}"
    echo "- Target arch: ${TARGET_ARCH:-preset-default}"
    echo "- Artifact directory: \`${artifact_dir}\`"
    echo
    echo "## Filters"
    if [[ "${#FILTERS[@]}" -gt 0 ]]; then
      for filter in "${FILTERS[@]}"; do
        echo "- \`${filter}\`"
      done
    else
      echo "- (none)"
    fi
    echo
    echo "## Compare Reports"
    for target in "${selected_targets[@]}"; do
      compare_report_file="${artifact_dir}/compare/${target}.md"
      if [[ ! -f "${compare_report_file}" ]]; then
        continue
      fi
      reports_emitted=$((reports_emitted + 1))
      echo
      echo "### \`${target}\`"
      echo
      echo "<details><summary>Expand full compare output for \`${target}\`</summary>"
      echo
      cat "${compare_report_file}"
      echo
      echo "</details>"
    done
    if [[ "${reports_emitted}" -eq 0 ]]; then
      echo
      echo "_No per-target compare reports were produced._"
    fi
  } > "${summary_file}"
}

parse_cli_args() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  if [[ "$#" -lt 2 ]]; then
    usage
    exit 2
  fi

  BASE_PATH="$(realpath "$1")"
  TEST_PATH="$(realpath "$2")"
  shift 2

  NVBENCH_ARGS_STRING=""
  NVBENCH_COMPARE_ARGS_STRING=""
  TARGET_ARCH=""
  FILTERS=()
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --arch)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --arch"
        fi
        TARGET_ARCH="$2"
        shift 2
        ;;
      --nvbench-args)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --nvbench-args"
        fi
        NVBENCH_ARGS_STRING="$2"
        shift 2
        ;;
      --nvbench-compare-args)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --nvbench-compare-args"
        fi
        NVBENCH_COMPARE_ARGS_STRING="$2"
        shift 2
        ;;
      --)
        shift
        FILTERS+=("$@")
        break
        ;;
      *)
        FILTERS+=("$1")
        shift
        ;;
    esac
  done
}

parse_cli_args "$@"

declare -a NVBENCH_RUN_ARGS
declare -a NVBENCH_COMPARE_ARGS
NVBENCH_RUN_ARGS=()
NVBENCH_COMPARE_ARGS=()
if ! parse_nvbench_run_args "${NVBENCH_ARGS_STRING}"; then
  die "Failed to parse --nvbench-args."
fi
if ! parse_nvbench_compare_args "${NVBENCH_COMPARE_ARGS_STRING}"; then
  die "Failed to parse --nvbench-compare-args."
fi

validate_repo_path "${BASE_PATH}"
validate_repo_path "${TEST_PATH}"
validate_filters

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
base_label_raw="${CCCL_BENCH_BASE_LABEL:-$(resolve_repo_label "${BASE_PATH}")}"
test_label_raw="${CCCL_BENCH_TEST_LABEL:-$(resolve_repo_label "${TEST_PATH}")}"
base_label="$(sanitize_label "${base_label_raw}")"
test_label="$(sanitize_label "${test_label_raw}")"

artifact_root="${CCCL_BENCH_ARTIFACT_ROOT:-$(pwd)/bench-artifacts}"
artifact_tag="${CCCL_BENCH_ARTIFACT_TAG:-bench-${test_label}-${timestamp}-${base_label}}"
artifact_tag="$(sanitize_label "${artifact_tag}")"
artifact_dir="${artifact_root}/${artifact_tag}"

build_root="${CCCL_BENCH_BUILD_ROOT:-/tmp/cccl-bench-builds}"
build_token="$(sanitize_label "${test_label}-${timestamp}-${base_label}")"
base_build_dir="${build_root}/base-${build_token}"
test_build_dir="${build_root}/test-${build_token}"

external_base_build_dir="${CCCL_BENCH_BASE_BUILD_DIR:-}"
external_test_build_dir="${CCCL_BENCH_TEST_BUILD_DIR:-}"
if [[ -n "${external_base_build_dir}" || -n "${external_test_build_dir}" ]]; then
  if [[ -z "${external_base_build_dir}" || -z "${external_test_build_dir}" ]]; then
    die "Both CCCL_BENCH_BASE_BUILD_DIR and CCCL_BENCH_TEST_BUILD_DIR must be set together."
  fi
  base_build_dir="$(realpath "${external_base_build_dir}")"
  test_build_dir="$(realpath "${external_test_build_dir}")"
  validate_build_dir "${base_build_dir}" "base"
  validate_build_dir "${test_build_dir}" "test"
  if [[ -n "${TARGET_ARCH}" ]]; then
    echo "Warning: --arch is ignored when using preconfigured build directories." >&2
  fi
fi

for subdir in base compare logs meta test; do
  mkdir -p "${artifact_dir}/${subdir}"
done
mkdir -p "${build_root}"

echo "Artifact directory: ${artifact_dir}"
echo "Base source: ${BASE_PATH}"
echo "Test source: ${TEST_PATH}"
if [[ "${#FILTERS[@]}" -gt 0 ]]; then
  echo "Filters:"
  for filter in "${FILTERS[@]}"; do
    echo "  - ${filter}"
  done
else
  echo "Filters: (none, all benchmark targets)"
fi
if [[ -n "${TARGET_ARCH}" ]]; then
  echo "Target arch: ${TARGET_ARCH}"
fi
if [[ "${#NVBENCH_RUN_ARGS[@]}" -gt 0 ]]; then
  echo "Extra run args:"
  for arg in "${NVBENCH_RUN_ARGS[@]}"; do
    echo "  - ${arg}"
  done
fi
if [[ "${#NVBENCH_COMPARE_ARGS[@]}" -gt 0 ]]; then
  echo "Extra compare args:"
  for arg in "${NVBENCH_COMPARE_ARGS[@]}"; do
    echo "  - ${arg}"
  done
fi

if [[ -n "${external_base_build_dir}" ]]; then
  echo "[configure:base] skipped (using existing build tree)"
  echo "[configure:test] skipped (using existing build tree)"
else
  configure_build_tree "${BASE_PATH}" "${base_build_dir}" "base" "${artifact_dir}/logs/configure.base.log" "${TARGET_ARCH}"
  configure_build_tree "${TEST_PATH}" "${test_build_dir}" "test" "${artifact_dir}/logs/configure.test.log" "${TARGET_ARCH}"
fi

declare -a selected_targets
select_targets "${base_build_dir}" "${test_build_dir}" selected_targets

printf "%s\n" "${selected_targets[@]}" > "${artifact_dir}/meta/selected_targets.txt"

compare_script="$(resolve_compare_script "${test_build_dir}" || true)"
if [[ -z "${compare_script}" ]]; then
  compare_script="$(resolve_compare_script "${base_build_dir}" || true)"
fi
if [[ -z "${compare_script}" ]]; then
  die "Unable to locate nvbench_compare.py in build dependencies." 1
fi
compare_script_dir="$(dirname "${compare_script}")"

any_failures=0
compares_attempted=0
compares_succeeded=0
base_build_all_rc=0
test_build_all_rc=0

if run_grouped_logged_command \
  "[build:base]" \
  "${artifact_dir}/logs/build.base.log" \
  ninja -C "${base_build_dir}" "${selected_targets[@]}"; then
  base_build_all_rc=0
else
  base_build_all_rc=$?
  any_failures=1
fi

if run_grouped_logged_command \
  "[build:test]" \
  "${artifact_dir}/logs/build.test.log" \
  ninja -C "${test_build_dir}" "${selected_targets[@]}"; then
  test_build_all_rc=0
else
  test_build_all_rc=$?
  any_failures=1
fi

for target in "${selected_targets[@]}"; do
  base_target_run_rc=125
  test_target_run_rc=125
  base_run_log="${artifact_dir}/logs/run.base.${target}.log"
  test_run_log="${artifact_dir}/logs/run.test.${target}.log"
  compare_report_md="${artifact_dir}/compare/${target}.md"
  compare_report_log="${artifact_dir}/logs/compare.${target}.log"

  base_json="${artifact_dir}/base/${target}.json"
  base_md="${artifact_dir}/base/${target}.md"
  test_json="${artifact_dir}/test/${target}.json"
  test_md="${artifact_dir}/test/${target}.md"

  if [[ "${base_build_all_rc}" -eq 0 ]]; then
    if run_target_for_side \
      "base" \
      "${base_build_dir}" \
      "${target}" \
      "${base_json}" \
      "${base_md}" \
      "${base_run_log}"; then
      base_target_run_rc=0
    else
      base_target_run_rc=$?
      any_failures=1
    fi
  fi

  if [[ "${test_build_all_rc}" -eq 0 ]]; then
    if run_target_for_side \
      "test" \
      "${test_build_dir}" \
      "${target}" \
      "${test_json}" \
      "${test_md}" \
      "${test_run_log}"; then
      test_target_run_rc=0
    else
      test_target_run_rc=$?
      any_failures=1
    fi
  fi

  if [[ "${base_target_run_rc}" -eq 0 && "${test_target_run_rc}" -eq 0 ]]; then
    compares_attempted=$((compares_attempted + 1))
    if run_compare_target \
      "${target}" \
      "${compare_script}" \
      "${compare_script_dir}" \
      "${base_json}" \
      "${test_json}" \
      "${compare_report_md}" \
      "${compare_report_log}"; then
      compares_succeeded=$((compares_succeeded + 1))
    else
      any_failures=1
    fi
  fi
done

summary_file="${artifact_dir}/summary.md"
write_summary "${summary_file}"

echo "Wrote summary: ${summary_file}"
echo "Benchmark artifacts: ${artifact_dir}"
echo
echo "Main summary:"
cat "${summary_file}"
echo

if [[ "${any_failures}" -ne 0 ]]; then
  exit 1
fi

exit 0
