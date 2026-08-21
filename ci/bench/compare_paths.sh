#!/usr/bin/env bash

set -euo pipefail

die() {
  local message="$1"
  local code="${2:-2}"
  echo "${message}" >&2
  exit "${code}"
}

usage() {
  cat <<EOF
Usage: $0 <base-path> <test-path> \
  [--cub-filter "<regex>"] \
  [--python-filter "<regex>"] \
  [--arch "<arch>"] \
  [--nvbench-args "<args>"] \
  [--nvbench-compare-args "<args>"] \
  [--nvbench-compare-legacy-args "<args>"]

Compare benchmark performance between two checked-out CCCL trees.

At least one --cub-filter or --python-filter must be provided.
CUB filters are regex patterns matched against ninja target names.
Python filters are regex patterns matched against benchmark script paths
under python/cuda_cccl/benchmarks/ (e.g. compute/reduce/sum.py).

Arguments:
  <base-path>  Path to baseline CCCL source tree.
  <test-path>  Path to comparison CCCL source tree.

Options:
  --cub-filter <regex>      CUB benchmark regex filter (repeatable).
  --python-filter <regex>   Python benchmark regex filter (repeatable).
  --arch <arch>             CMAKE_CUDA_ARCHITECTURES for CUB builds.
  --nvbench-args <args>     Extra args passed to benchmark binaries/scripts.
  --nvbench-compare-args <args>  Extra args passed to nvbench-compare-robust.
  --nvbench-compare-legacy-args <args>
                             Extra args passed to nvbench-compare-legacy.

Environment:
  CCCL_BENCH_ARTIFACT_ROOT   Root directory for outputs.
                             Default: "\$(pwd)/bench-artifacts"
  CCCL_BENCH_ARTIFACT_TAG    Optional explicit artifact directory name.
  CCCL_BENCH_BASE_LABEL      Optional label used in auto-generated artifact names.
  CCCL_BENCH_TEST_LABEL      Optional label used in auto-generated artifact names.
  CCCL_BENCH_BASE_BUILD_DIR  Optional preconfigured build tree for base path.
  CCCL_BENCH_TEST_BUILD_DIR  Optional preconfigured build tree for test path.
                             If either is set, both must be set.
  CCCL_BENCH_GPU_NAME        Optional GPU name included in artifact directory names.
  CCCL_BENCH_NVBENCH_SHA     NVBench SHA/tag used for generated CUB build trees.
                             Default: "main"
  CCCL_BENCH_COMPARE_PYTHON  Optional Python executable used for CUB benchmark
                             comparisons.
  CCCL_BENCH_COMPARE_BIN     Optional nvbench-compare-robust executable used for
                             Python benchmark comparisons.
  CCCL_BENCH_LEGACY_COMPARE_BIN
                             Optional nvbench-compare-legacy executable used for
                             Python benchmark comparisons.
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

validate_filter_array() {
  local -n _validate_filters_ref="$1"
  local label="$2"
  local filter=""
  for filter in "${_validate_filters_ref[@]}"; do
    grep -Eq -- "${filter}" <<< "" >/dev/null 2>&1 || {
      [[ "$?" -eq 1 ]] || die "Invalid ${label} regex filter: ${filter}"
    }
  done
}

print_shell_command() {
  # Print a shell-escaped command line so it can be copied and re-run directly.
  # Usage: print_shell_command [--env "VAR=value" ...] cmd [args...]
  local -a env_prefixes=()
  while [[ "$#" -gt 0 && "$1" == --env ]]; do
    shift
    env_prefixes+=("$1")
    shift
  done
  printf '$'
  local item=""
  for item in "${env_prefixes[@]}"; do
    printf ' %q' "${item}"
  done
  for item in "$@"; do
    printf ' %q' "${item}"
  done
  printf '\n'
}

# ============================================================================
# CUB helpers
# ============================================================================

configure_build_tree() {
  local src_path="$1"
  local build_path="$2"
  local side="$3"
  local log_path="$4"
  local target_arch="$5"
  local -a cmake_cmd
  cmake_cmd=(
    cmake
    --preset "cub-benchmark"
    -S "${src_path}"
    -B "${build_path}"
    "-DCCCL_NVBENCH_SHA=${nvbench_sha}"
  )
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
  local nvbench_src="${build_path}/_deps/nvbench-src"
  local candidate=""
  # Different versions have the script at different locations:
  for candidate in \
    "${nvbench_src}/python/scripts/nvbench_compare_robust.py" \
    "${nvbench_src}/scripts/nvbench_compare_robust.py"; do
    if [[ -f "${candidate}" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_legacy_compare_script() {
  local build_path="$1"
  local nvbench_src="${build_path}/_deps/nvbench-src"
  local candidate=""
  # Different versions have the script at different locations:
  for candidate in \
    "${nvbench_src}/python/scripts/nvbench_compare.py" \
    "${nvbench_src}/scripts/nvbench_compare.py"; do
    if [[ -f "${candidate}" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  done
  return 1
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
    --jsonbin "${json_path}"
    --md "${md_path}"
  )

  run_grouped_logged_command \
    "[run:${side}] ${target}" \
    "${run_log}" \
    "${bench_cmd[@]}"
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
    die "No CUB benchmark targets matched the supplied filters." 1
  fi
}

# ============================================================================
# Python helpers
# ============================================================================

detect_cuda_major_version() {
  local cuda_major=""
  if command -v nvcc >/dev/null 2>&1; then
    cuda_major="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  fi
  if [[ -z "${cuda_major}" ]]; then
    cuda_major="12"
  fi
  printf "%s" "${cuda_major}"
}

python_path_to_target_name() {
  local py_path="$1"
  # compute/reduce/sum.py -> py.compute.reduce.sum
  local name="${py_path%.py}"
  name="${name//\//.}"
  printf "py.%s" "${name}"
}

list_all_python_benchmarks() {
  local benchmarks_path="$1"
  if [[ ! -d "${benchmarks_path}" ]]; then
    return 0
  fi
  find "${benchmarks_path}" -name '*.py' -type f \
    ! -name 'utils.py' \
    ! -name 'run_benchmarks.py' \
    ! -name 'device_side_benchmark.py' \
    ! -name '__init__.py' \
    ! -path '*/__pycache__/*' \
    -printf '%P\n' \
    | sort -u
}

python_target_matches_filters() {
  local target="$1"
  local filter=""
  for filter in "${PYTHON_FILTERS[@]}"; do
    if grep -Eq -- "${filter}" <<< "${target}"; then
      return 0
    fi
  done
  return 1
}

select_python_targets() {
  local base_bench_path="$1"
  local test_bench_path="$2"
  local -n selected_py_targets_ref="$3"
  local -a base_py_targets
  local -a test_py_targets
  local -a common_py_targets
  local target=""

  mapfile -t base_py_targets < <(list_all_python_benchmarks "${base_bench_path}")
  mapfile -t test_py_targets < <(list_all_python_benchmarks "${test_bench_path}")

  if [[ "${#base_py_targets[@]}" -eq 0 ]]; then
    die "No Python benchmark scripts were found in base tree: ${base_bench_path}" 1
  fi
  if [[ "${#test_py_targets[@]}" -eq 0 ]]; then
    die "No Python benchmark scripts were found in test tree: ${test_bench_path}" 1
  fi

  mapfile -t common_py_targets < <(
    comm -12 \
      <(printf "%s\n" "${base_py_targets[@]}" | sort -u) \
      <(printf "%s\n" "${test_py_targets[@]}" | sort -u)
  )

  selected_py_targets_ref=()
  for target in "${common_py_targets[@]}"; do
    [[ -n "${target}" ]] || continue
    if python_target_matches_filters "${target}"; then
      selected_py_targets_ref+=("${target}")
    fi
  done

  if [[ "${#selected_py_targets_ref[@]}" -eq 0 ]]; then
    die "No Python benchmark scripts matched the supplied --python-filter patterns." 1
  fi
}

setup_python_venv() {
  local venv_path="$1"
  local src_path="$2"
  local side="$3"
  local log_path="$4"
  local cuda_major="$5"
  local cuda_cccl_dir="${src_path}/python/cuda_cccl"

  if [[ ! -d "${cuda_cccl_dir}" ]]; then
    die "cuda_cccl source directory not found: ${cuda_cccl_dir}"
  fi

  local -a setup_cmds
  setup_cmds=(
    bash -c "
      set -euo pipefail
      python3 -m venv '${venv_path}'
      '${venv_path}/bin/pip' install --upgrade pip
      '${venv_path}/bin/pip' install -e '${cuda_cccl_dir}[bench-cu${cuda_major}]'
      '${venv_path}/bin/pip' install 'cuda-bench[compare]>=0.3.0'
    "
  )

  run_grouped_logged_command \
    "[py-venv:${side}]" \
    "${log_path}" \
    "${setup_cmds[@]}"
}

run_python_target_for_side() {
  local side="$1"
  local venv_path="$2"
  local script_path="$3"
  local json_path="$4"
  local md_path="$5"
  local run_log="$6"
  local -a bench_cmd

  if [[ ! -f "${script_path}" ]]; then
    echo "Python benchmark script missing: ${script_path}" >&2
    return 127
  fi

  bench_cmd=(
    "${venv_path}/bin/python"
    "${script_path}"
    -d 0
    "${NVBENCH_RUN_ARGS[@]}"
    --jsonbin "${json_path}"
    --md "${md_path}"
  )

  run_grouped_logged_command \
    "[py-run:${side}] ${script_path##*/benchmarks/}" \
    "${run_log}" \
    "${bench_cmd[@]}"
}

resolve_legacy_compare_bin() {
  local venv_path="$1"
  local compare_bin="$2"
  local candidate=""

  if [[ -n "${CCCL_BENCH_LEGACY_COMPARE_BIN:-}" ]]; then
    if [[ -x "${CCCL_BENCH_LEGACY_COMPARE_BIN}" ]]; then
      printf "%s" "${CCCL_BENCH_LEGACY_COMPARE_BIN}"
      return 0
    fi
    echo "CCCL_BENCH_LEGACY_COMPARE_BIN is set but not executable: ${CCCL_BENCH_LEGACY_COMPARE_BIN}" >&2
    return 1
  fi

  if [[ "${compare_bin}" == */* ]]; then
    candidate="$(dirname "${compare_bin}")/nvbench-compare-legacy"
    if [[ -x "${candidate}" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  fi

  candidate="${venv_path}/bin/nvbench-compare-legacy"
  if [[ -x "${candidate}" ]]; then
    printf "%s" "${candidate}"
    return 0
  fi

  command -v nvbench-compare-legacy 2>/dev/null || return 1
}

run_python_compare_target() {
  local target_name="$1"
  local venv_path="$2"
  local base_json="$3"
  local test_json="$4"

  local compare_bin="${CCCL_BENCH_COMPARE_BIN:-${venv_path}/bin/nvbench-compare-robust}"
  local legacy_compare_bin=""
  local display=""
  local compare_out=""
  local compare_log=""
  local robust_rc=0
  local -a compare_cmd

  for display in "${COMPARE_DISPLAYS[@]}"; do
    compare_out="$(compare_report_path_for_display "${target_name}" "${display}")"
    compare_log="$(compare_log_path_for_display "${target_name}" "${display}")"
    compare_cmd=(
      "${compare_bin}"
      --no-color
      "${NVBENCH_COMPARE_ARGS[@]}"
      --display
      "${display}"
      "${base_json}"
      "${test_json}"
    )

    if ! run_compare_command \
      "[py-compare:${display}] ${target_name}" \
      "${compare_out}" \
      "${compare_log}" \
      "${compare_cmd[@]}"; then
      robust_rc=1
    fi
    if [[ "${display}" == "explain" ]] && ! normalize_explain_markdown_report "${compare_out}"; then
      robust_rc=1
    fi
  done

  legacy_compare_bin="$(resolve_legacy_compare_bin "${venv_path}" "${compare_bin}" || true)"
  if [[ -n "${legacy_compare_bin}" ]]; then
    compare_cmd=(
      "${legacy_compare_bin}"
      --no-color
      "${NVBENCH_COMPARE_LEGACY_ARGS[@]}"
      "${base_json}"
      "${test_json}"
    )
    if ! run_compare_command \
      "[py-compare:legacy] ${target_name}" \
      "${artifact_dir}/compare/${target_name}.legacy.md" \
      "${artifact_dir}/logs/compare.${target_name}.legacy.log" \
      "${compare_cmd[@]}"; then
      echo "Legacy compare output is diagnostic; continuing after legacy compare failure." >&2
    fi
  fi

  return "${robust_rc}"
}

# ============================================================================
# Common helpers
# ============================================================================

run_grouped_logged_command() {
  local label="$1"
  local log_path="$2"
  shift 2
  local started_at=0
  local elapsed_s=0
  local rc=0
  local -a pipe_statuses

  echo "::group::${label}"
  print_shell_command "$@"
  started_at="${SECONDS}"
  set +o pipefail
  "$@" 2>&1 | tee "${log_path}"
  pipe_statuses=("${PIPESTATUS[@]}")
  set -o pipefail
  if [[ "${pipe_statuses[0]}" -ne 0 ]]; then
    rc="${pipe_statuses[0]}"
  elif [[ "${pipe_statuses[1]}" -ne 0 ]]; then
    rc="${pipe_statuses[1]}"
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

compare_report_path_for_display() {
  local target_name="$1"
  local display="$2"

  if [[ "${display}" == "intervals" ]]; then
    printf "%s" "${artifact_dir}/compare/${target_name}.md"
  else
    printf "%s" "${artifact_dir}/compare/${target_name}.${display}.md"
  fi
}

compare_log_path_for_display() {
  local target_name="$1"
  local display="$2"

  if [[ "${display}" == "intervals" ]]; then
    printf "%s" "${artifact_dir}/logs/compare.${target_name}.log"
  else
    printf "%s" "${artifact_dir}/logs/compare.${target_name}.${display}.log"
  fi
}

normalize_explain_markdown_report() {
  local report_path="$1"
  local tmp_path=""

  [[ -s "${report_path}" ]] || return 0

  tmp_path="$(mktemp "${report_path}.XXXXXX")"
  if ! awk '
    function clear_array(values, key) {
      for (key in values) {
        delete values[key]
      }
    }

    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }

    function split_markdown_row(line, cells,    i, c, depth, cell, count) {
      clear_array(cells)
      if (substr(line, 1, 1) != "|") {
        return 0
      }

      depth = 0
      cell = ""
      count = 0
      for (i = 2; i <= length(line); i++) {
        c = substr(line, i, 1)
        if (c == "[") {
          depth++
        } else if (c == "]") {
          if (depth > 0) {
            depth--
          }
        }

        if (c == "|" && depth == 0) {
          count++
          cells[count] = cell
          cell = ""
        } else {
          cell = cell c
        }
      }

      if (cell != "") {
        count++
        cells[count] = cell
      }
      return count
    }

    function replace_interval_pipes(value,    i, c, depth, out) {
      depth = 0
      out = ""
      for (i = 1; i <= length(value); i++) {
        c = substr(value, i, 1)
        if (c == "[") {
          depth++
          out = out c
        } else if (c == "]") {
          if (depth > 0) {
            depth--
          }
          out = out c
        } else if (c == "|" && depth > 0) {
          out = out "/"
        } else {
          out = out c
        }
      }
      return out
    }

    function print_markdown_row(cells, count,    i, out) {
      out = "|"
      for (i = 1; i <= count; i++) {
        out = out cells[i] "|"
      }
      print out
    }

    {
      if (substr($0, 1, 1) != "|") {
        clear_array(interval_columns)
        interval_column_count = 0
        print
        next
      }

      cell_count = split_markdown_row($0, cells)
      if (cell_count == 0) {
        print
        next
      }

      ref_interval_column = cell_count - 6
      cmp_interval_column = cell_count - 5
      found_interval_columns = \
        ref_interval_column > 0 && \
        trim(cells[ref_interval_column]) == "Ref [Lo | Ce | Hi]" && \
        trim(cells[cmp_interval_column]) == "Cmp [Lo | Ce | Hi]" && \
        trim(cells[cell_count - 4]) == "Ref Noise" && \
        trim(cells[cell_count - 3]) == "Cmp Noise" && \
        trim(cells[cell_count - 2]) == "Reason" && \
        trim(cells[cell_count - 1]) == "Change" && \
        trim(cells[cell_count]) == "Status"

      if (found_interval_columns) {
        clear_array(interval_columns)
        interval_columns[ref_interval_column] = 1
        interval_columns[cmp_interval_column] = 1
        interval_column_count = 2
        cells[ref_interval_column] = replace_interval_pipes(cells[ref_interval_column])
        cells[cmp_interval_column] = replace_interval_pipes(cells[cmp_interval_column])
        print_markdown_row(cells, cell_count)
        next
      }

      if (interval_column_count > 0) {
        for (i in interval_columns) {
          if (i <= cell_count) {
            cells[i] = replace_interval_pipes(cells[i])
          }
        }
        print_markdown_row(cells, cell_count)
        next
      }

      print
    }
  ' "${report_path}" > "${tmp_path}"; then
    rm -f "${tmp_path}"
    return 1
  fi

  mv "${tmp_path}" "${report_path}"
}

run_compare_command_impl() {
  local label="$1"
  local compare_out="$2"
  local compare_log="$3"
  shift 3

  local -a env_prefixes=()
  while [[ "$#" -gt 0 && "$1" == --env ]]; do
    shift
    env_prefixes+=("$1")
    shift
  done

  local started_at=0
  local elapsed_s=0
  local rc=0
  local stderr_tmp=""
  stderr_tmp="$(mktemp "${compare_log}.stderr.XXXXXX")"

  : > "${compare_out}"
  : > "${compare_log}"
  echo "::group::${label}"

  local -a print_env_args=()
  local env_prefix=""
  for env_prefix in "${env_prefixes[@]}"; do
    print_env_args+=(--env "${env_prefix}")
  done
  print_shell_command "${print_env_args[@]}" "$@"

  started_at="${SECONDS}"
  if [[ "${#env_prefixes[@]}" -gt 0 ]]; then
    if env "${env_prefixes[@]}" "$@" > "${compare_out}" 2> "${stderr_tmp}"; then
      rc=0
    else
      rc=$?
    fi
  elif "$@" > "${compare_out}" 2> "${stderr_tmp}"; then
    rc=0
  else
    rc=$?
  fi
  elapsed_s=$((SECONDS - started_at))
  if [[ -s "${compare_out}" ]]; then
    cat "${compare_out}"
    cat "${compare_out}" >> "${compare_log}"
  fi
  if [[ -s "${stderr_tmp}" ]]; then
    cat "${stderr_tmp}" >&2
    cat "${stderr_tmp}" >> "${compare_log}"
  fi
  rm -f "${stderr_tmp}"
  if [[ "${rc}" -ne 0 && ! -s "${compare_out}" ]]; then
    printf "_Compare command failed with rc=%s; see \`%s\` for stderr/log output._\n" \
      "${rc}" \
      "${compare_log#"${artifact_dir}"/}" \
      > "${compare_out}"
  fi
  echo "::endgroup::"
  if [[ "${rc}" -eq 0 ]]; then
    echo "${label} completed in ${elapsed_s}s"
  else
    echo "${label} failed in ${elapsed_s}s (rc=${rc})"
  fi
  return "${rc}"
}

run_compare_command() {
  run_compare_command_impl "$@"
}

run_compare_command_with_pythonpath() {
  local label="$1"
  local compare_pythonpath="$2"
  local compare_out="$3"
  local compare_log="$4"
  shift 4

  run_compare_command_impl \
    "${label}" \
    "${compare_out}" \
    "${compare_log}" \
    --env "PYTHONPATH=${compare_pythonpath}" \
    "$@"
}

run_compare_target() {
  local target="$1"
  local compare_script="$2"
  local compare_script_dir="$3"
  local base_json="$4"
  local test_json="$5"
  local legacy_compare_script="$6"
  local legacy_compare_script_dir="$7"

  local compare_pythonpath="${compare_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  local compare_python="${CCCL_BENCH_COMPARE_PYTHON:-python3}"
  local display=""
  local compare_out=""
  local compare_log=""
  local robust_rc=0
  local -a compare_cmd

  for display in "${COMPARE_DISPLAYS[@]}"; do
    compare_out="$(compare_report_path_for_display "${target}" "${display}")"
    compare_log="$(compare_log_path_for_display "${target}" "${display}")"
    compare_cmd=(
      "${compare_python}"
      "${compare_script}"
      --no-color
      "${NVBENCH_COMPARE_ARGS[@]}"
      --display
      "${display}"
      "${base_json}"
      "${test_json}"
    )

    if ! run_compare_command_with_pythonpath \
      "[compare:${display}] ${target}" \
      "${compare_pythonpath}" \
      "${compare_out}" \
      "${compare_log}" \
      "${compare_cmd[@]}"; then
      robust_rc=1
    fi
    if [[ "${display}" == "explain" ]] && ! normalize_explain_markdown_report "${compare_out}"; then
      robust_rc=1
    fi
  done

  if [[ -n "${legacy_compare_script}" ]]; then
    compare_pythonpath="${legacy_compare_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    compare_cmd=(
      "${compare_python}"
      "${legacy_compare_script}"
      --no-color
      "${NVBENCH_COMPARE_LEGACY_ARGS[@]}"
      "${base_json}"
      "${test_json}"
    )
    if ! run_compare_command_with_pythonpath \
      "[compare:legacy] ${target}" \
      "${compare_pythonpath}" \
      "${artifact_dir}/compare/${target}.legacy.md" \
      "${artifact_dir}/logs/compare.${target}.legacy.log" \
      "${compare_cmd[@]}"; then
      echo "Legacy compare output is diagnostic; continuing after legacy compare failure." >&2
    fi
  fi

  return "${robust_rc}"
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

parse_quoted_args_to_array() {
  local -n _target_array_ref="$1"
  local quoted_args="$2"
  local option_label="$3"
  local parsed_args_file=""

  _target_array_ref=()
  [[ -n "${quoted_args}" ]] || return 0

  parsed_args_file="$(mktemp "/tmp/cccl-parsed-args-XXXXXX")"
  if ! parse_quoted_args_to_nul_file "${quoted_args}" "${parsed_args_file}" "${option_label}"; then
    rm -f "${parsed_args_file}"
    return 2
  fi
  mapfile -d '' -t _target_array_ref < "${parsed_args_file}"
  rm -f "${parsed_args_file}"
}

# ============================================================================
# Summary
# ============================================================================

write_compare_report_details() {
  local summary="$1"
  local compare_report_file="$2"

  if [[ ! -s "${compare_report_file}" ]]; then
    return 1
  fi

  echo
  echo "<details><summary>${summary}</summary>"
  echo
  cat "${compare_report_file}"
  echo
  echo "</details>"
  return 0
}

write_robust_summary_excerpt() {
  local compare_report_file="$1"

  if [[ ! -s "${compare_report_file}" ]]; then
    return 1
  fi

  awk '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }

    {
      if (!printing && trim($0) == "# Summary") {
        print "**Robust comparison summary**"
        print ""
        printing = 1
        skip_leading_blanks = 1
        emitted = 1
        next
      }
      if (printing) {
        if (skip_leading_blanks && trim($0) == "") {
          next
        }
        skip_leading_blanks = 0
        print
      }
    }

    END { exit emitted ? 0 : 1 }
  ' "${compare_report_file}"
}

write_summary() {
  local summary_file="$1"
  local target=""
  local compare_report_file=""
  local display=""
  local reports_emitted=0

  {
    echo "# Benchmark Comparison Summary"
    echo
    echo "- Timestamp (UTC): ${timestamp}"
    echo "- GPU name: ${CCCL_BENCH_GPU_NAME:-not specified}"
    echo "- Base label: ${base_label_raw}"
    echo "- Test label: ${test_label_raw}"
    echo "- Base source path: \`${BASE_PATH}\`"
    echo "- Test source path: \`${TEST_PATH}\`"
    if [[ "${#FILTERS[@]}" -gt 0 ]]; then
      echo "- Base build dir: \`${base_build_dir}\`"
      echo "- Test build dir: \`${test_build_dir}\`"
      echo "- NVBench SHA/tag: \`${nvbench_sha}\`"
    fi
    echo "- CUB targets selected: ${#selected_targets[@]}"
    echo "- CUB comparisons attempted: ${compares_attempted}"
    echo "- CUB comparisons succeeded: ${compares_succeeded}"
    echo "- Python targets selected: ${#selected_py_targets[@]}"
    echo "- Python comparisons attempted: ${py_compares_attempted}"
    echo "- Python comparisons succeeded: ${py_compares_succeeded}"
    echo "- Target arch: ${TARGET_ARCH:-preset-default}"
    echo "- Artifact directory: \`${artifact_dir}\`"
    echo

    if [[ "${#FILTERS[@]}" -gt 0 ]]; then
      echo "## CUB Filters"
      for filter in "${FILTERS[@]}"; do
        echo "- \`${filter}\`"
      done
      echo
    fi

    echo "## Benchmark Arguments"
    if [[ -n "${NVBENCH_ARGS_STRING}" ]]; then
      echo "- NVBench args: \`${NVBENCH_ARGS_STRING}\`"
    else
      echo "- NVBench args: not specified"
    fi
    if [[ -n "${NVBENCH_COMPARE_ARGS_STRING}" ]]; then
      echo "- NVBench robust compare args: \`${NVBENCH_COMPARE_ARGS_STRING}\`"
    else
      echo "- NVBench robust compare args: not specified"
    fi
    if [[ -n "${NVBENCH_COMPARE_LEGACY_ARGS_STRING}" ]]; then
      echo "- NVBench legacy compare args: \`${NVBENCH_COMPARE_LEGACY_ARGS_STRING}\`"
    else
      echo "- NVBench legacy compare args: not specified"
    fi
    echo

    if [[ "${#PYTHON_FILTERS[@]}" -gt 0 ]]; then
      echo "## Python Filters"
      for filter in "${PYTHON_FILTERS[@]}"; do
        echo "- \`${filter}\`"
      done
      echo
    fi

    if [[ "${#selected_targets[@]}" -gt 0 ]]; then
      echo "## CUB Compare Reports"
      for target in "${selected_targets[@]}"; do
        echo
        echo "### \`${target}\`"
        compare_report_file="$(compare_report_path_for_display "${target}" "intervals")"
        write_robust_summary_excerpt "${compare_report_file}" || true
        for display in "${COMPARE_DISPLAYS[@]}"; do
          compare_report_file="$(compare_report_path_for_display "${target}" "${display}")"
          if write_compare_report_details "Robust ${display} output" "${compare_report_file}"; then
            reports_emitted=$((reports_emitted + 1))
          fi
        done
        compare_report_file="${artifact_dir}/compare/${target}.legacy.md"
        if write_compare_report_details "Legacy output" "${compare_report_file}"; then
          reports_emitted=$((reports_emitted + 1))
        fi
      done
    fi

    if [[ "${#selected_py_targets[@]}" -gt 0 ]]; then
      echo
      echo "## Python Compare Reports"
      local py_target_path=""
      local py_target_name=""
      for py_target_path in "${selected_py_targets[@]}"; do
        py_target_name="$(python_path_to_target_name "${py_target_path}")"
        echo
        echo "### \`${py_target_name}\` (\`${py_target_path}\`)"
        compare_report_file="$(compare_report_path_for_display "${py_target_name}" "intervals")"
        write_robust_summary_excerpt "${compare_report_file}" || true
        for display in "${COMPARE_DISPLAYS[@]}"; do
          compare_report_file="$(compare_report_path_for_display "${py_target_name}" "${display}")"
          if write_compare_report_details "Robust ${display} output" "${compare_report_file}"; then
            reports_emitted=$((reports_emitted + 1))
          fi
        done
        compare_report_file="${artifact_dir}/compare/${py_target_name}.legacy.md"
        if write_compare_report_details "Legacy output" "${compare_report_file}"; then
          reports_emitted=$((reports_emitted + 1))
        fi
      done
    fi

    if [[ "${reports_emitted}" -eq 0 ]]; then
      echo
      echo "_No per-target compare reports were produced._"
    fi
  } > "${summary_file}"
}

# ============================================================================
# CLI parsing
# ============================================================================

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
  NVBENCH_COMPARE_LEGACY_ARGS_STRING=""
  TARGET_ARCH=""
  FILTERS=()
  PYTHON_FILTERS=()
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
      --nvbench-compare-legacy-args)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --nvbench-compare-legacy-args"
        fi
        NVBENCH_COMPARE_LEGACY_ARGS_STRING="$2"
        shift 2
        ;;
      --cub-filter)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --cub-filter"
        fi
        FILTERS+=("$2")
        shift 2
        ;;
      --python-filter)
        if [[ "$#" -lt 2 ]]; then
          die "Missing value for --python-filter"
        fi
        PYTHON_FILTERS+=("$2")
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

parse_cli_args "$@"

declare -a NVBENCH_RUN_ARGS
declare -a NVBENCH_COMPARE_ARGS
declare -a NVBENCH_COMPARE_LEGACY_ARGS
readonly -a COMPARE_DISPLAYS=(intervals simple explain)
parse_quoted_args_to_array NVBENCH_RUN_ARGS "${NVBENCH_ARGS_STRING}" "--nvbench-args" \
  || die "Failed to parse --nvbench-args."
parse_quoted_args_to_array NVBENCH_COMPARE_ARGS "${NVBENCH_COMPARE_ARGS_STRING}" "--nvbench-compare-args" \
  || die "Failed to parse --nvbench-compare-args."
parse_quoted_args_to_array \
  NVBENCH_COMPARE_LEGACY_ARGS \
  "${NVBENCH_COMPARE_LEGACY_ARGS_STRING}" \
  "--nvbench-compare-legacy-args" \
  || die "Failed to parse --nvbench-compare-legacy-args."

validate_repo_path "${BASE_PATH}"
validate_repo_path "${TEST_PATH}"
validate_filter_array FILTERS "CUB"
validate_filter_array PYTHON_FILTERS "Python"

# ============================================================================
# Common setup
# ============================================================================

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
base_label_raw="${CCCL_BENCH_BASE_LABEL:-$(resolve_repo_label "${BASE_PATH}")}"
test_label_raw="${CCCL_BENCH_TEST_LABEL:-$(resolve_repo_label "${TEST_PATH}")}"
base_label="$(sanitize_label "${base_label_raw}")"
test_label="$(sanitize_label "${test_label_raw}")"

artifact_root="${CCCL_BENCH_ARTIFACT_ROOT:-$(pwd)/bench-artifacts}"
gpu_tag="${CCCL_BENCH_GPU_NAME:+$(sanitize_label "${CCCL_BENCH_GPU_NAME}")-}"
artifact_tag="${CCCL_BENCH_ARTIFACT_TAG:-bench-${gpu_tag}${test_label}-${timestamp}-${base_label}}"
artifact_tag="$(sanitize_label "${artifact_tag}")"
artifact_dir="${artifact_root}/${artifact_tag}"

build_root="${CCCL_BENCH_BUILD_ROOT:-/tmp/cccl-bench-builds}"
nvbench_sha="${CCCL_BENCH_NVBENCH_SHA:-main}"
build_token="$(sanitize_label "${test_label}-${timestamp}-${base_label}")"
base_build_dir="${build_root}/base-${build_token}"
test_build_dir="${build_root}/test-${build_token}"

for subdir in base compare logs meta test; do
  mkdir -p "${artifact_dir}/${subdir}"
done
mkdir -p "${build_root}"

echo "Artifact directory: ${artifact_dir}"
if [[ -n "${CCCL_BENCH_GPU_NAME:-}" ]]; then
  echo "GPU name: ${CCCL_BENCH_GPU_NAME}"
fi
echo "Base source: ${BASE_PATH}"
echo "Test source: ${TEST_PATH}"
if [[ "${#FILTERS[@]}" -gt 0 ]]; then
  echo "NVBench SHA/tag: ${nvbench_sha}"
  echo "CUB filters:"
  for filter in "${FILTERS[@]}"; do
    echo "  - ${filter}"
  done
else
  echo "CUB filters: (none)"
fi
if [[ "${#PYTHON_FILTERS[@]}" -gt 0 ]]; then
  echo "Python filters:"
  for filter in "${PYTHON_FILTERS[@]}"; do
    echo "  - ${filter}"
  done
else
  echo "Python filters: (none)"
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
  echo "Extra robust compare args:"
  for arg in "${NVBENCH_COMPARE_ARGS[@]}"; do
    echo "  - ${arg}"
  done
fi
if [[ "${#NVBENCH_COMPARE_LEGACY_ARGS[@]}" -gt 0 ]]; then
  echo "Extra legacy compare args:"
  for arg in "${NVBENCH_COMPARE_LEGACY_ARGS[@]}"; do
    echo "  - ${arg}"
  done
fi

any_failures=0
compares_attempted=0
compares_succeeded=0
declare -a selected_targets=()
py_compares_attempted=0
py_compares_succeeded=0
declare -a selected_py_targets=()

# ============================================================================
# CUB benchmark pipeline
# ============================================================================

if [[ "${#FILTERS[@]}" -gt 0 ]]; then
  echo
  echo "=== CUB Benchmark Pipeline ==="
  echo

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

  if [[ -z "${external_base_build_dir:-}" ]]; then
    configure_build_tree "${BASE_PATH}" "${base_build_dir}" "base" "${artifact_dir}/logs/configure.base.log" "${TARGET_ARCH}"
    configure_build_tree "${TEST_PATH}" "${test_build_dir}" "test" "${artifact_dir}/logs/configure.test.log" "${TARGET_ARCH}"
  else
    echo "[configure:base] skipped (using existing build tree)"
    echo "[configure:test] skipped (using existing build tree)"
  fi

  select_targets "${base_build_dir}" "${test_build_dir}" selected_targets

  printf "%s\n" "${selected_targets[@]}" > "${artifact_dir}/meta/selected_targets.txt"

  compare_script="$(resolve_compare_script "${test_build_dir}" || true)"
  if [[ -z "${compare_script}" ]]; then
    compare_script="$(resolve_compare_script "${base_build_dir}" || true)"
  fi
  if [[ -z "${compare_script}" ]]; then
    die "Unable to locate nvbench_compare_robust.py in build dependencies." 1
  fi
  compare_script_dir="$(dirname "${compare_script}")"

  legacy_compare_script="$(resolve_legacy_compare_script "${test_build_dir}" || true)"
  if [[ -z "${legacy_compare_script}" ]]; then
    legacy_compare_script="$(resolve_legacy_compare_script "${base_build_dir}" || true)"
  fi
  legacy_compare_script_dir=""
  if [[ -n "${legacy_compare_script}" ]]; then
    legacy_compare_script_dir="$(dirname "${legacy_compare_script}")"
  fi

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
        "${legacy_compare_script}" \
        "${legacy_compare_script_dir}"; then
        compares_succeeded=$((compares_succeeded + 1))
      else
        any_failures=1
      fi
    fi
  done
fi

# ============================================================================
# Python benchmark pipeline
# ============================================================================

if [[ "${#PYTHON_FILTERS[@]}" -gt 0 ]]; then
  echo
  echo "=== Python Benchmark Pipeline ==="
  echo

  py_benchmarks_subdir="python/cuda_cccl/benchmarks"
  base_py_bench_dir="${BASE_PATH}/${py_benchmarks_subdir}"
  test_py_bench_dir="${TEST_PATH}/${py_benchmarks_subdir}"

  if [[ ! -d "${base_py_bench_dir}" ]]; then
    die "Python benchmarks directory not found in base tree: ${base_py_bench_dir}"
  fi
  if [[ ! -d "${test_py_bench_dir}" ]]; then
    die "Python benchmarks directory not found in test tree: ${test_py_bench_dir}"
  fi

  cuda_major="$(detect_cuda_major_version)"
  echo "Detected CUDA major version: ${cuda_major}"

  base_py_venv="${build_root}/py-base-${build_token}"
  test_py_venv="${build_root}/py-test-${build_token}"

  setup_python_venv "${base_py_venv}" "${BASE_PATH}" "base" "${artifact_dir}/logs/py.venv.base.log" "${cuda_major}"
  setup_python_venv "${test_py_venv}" "${TEST_PATH}" "test" "${artifact_dir}/logs/py.venv.test.log" "${cuda_major}"

  select_python_targets "${base_py_bench_dir}" "${test_py_bench_dir}" selected_py_targets

  # Append Python targets to the selected targets metadata file.
  for py_target_path in "${selected_py_targets[@]}"; do
    python_path_to_target_name "${py_target_path}" >> "${artifact_dir}/meta/selected_targets.txt"
  done

  for py_target_path in "${selected_py_targets[@]}"; do
    py_target_name="$(python_path_to_target_name "${py_target_path}")"
    base_py_target_run_rc=125
    test_py_target_run_rc=125

    base_py_json="${artifact_dir}/base/${py_target_name}.json"
    base_py_md="${artifact_dir}/base/${py_target_name}.md"
    test_py_json="${artifact_dir}/test/${py_target_name}.json"
    test_py_md="${artifact_dir}/test/${py_target_name}.md"
    base_py_run_log="${artifact_dir}/logs/run.base.${py_target_name}.log"
    test_py_run_log="${artifact_dir}/logs/run.test.${py_target_name}.log"

    if run_python_target_for_side \
      "base" \
      "${base_py_venv}" \
      "${base_py_bench_dir}/${py_target_path}" \
      "${base_py_json}" \
      "${base_py_md}" \
      "${base_py_run_log}"; then
      base_py_target_run_rc=0
    else
      base_py_target_run_rc=$?
      any_failures=1
    fi

    if run_python_target_for_side \
      "test" \
      "${test_py_venv}" \
      "${test_py_bench_dir}/${py_target_path}" \
      "${test_py_json}" \
      "${test_py_md}" \
      "${test_py_run_log}"; then
      test_py_target_run_rc=0
    else
      test_py_target_run_rc=$?
      any_failures=1
    fi

    if [[ "${base_py_target_run_rc}" -eq 0 && "${test_py_target_run_rc}" -eq 0 ]]; then
      py_compares_attempted=$((py_compares_attempted + 1))
      if run_python_compare_target \
        "${py_target_name}" \
        "${test_py_venv}" \
        "${base_py_json}" \
        "${test_py_json}"; then
        py_compares_succeeded=$((py_compares_succeeded + 1))
      else
        any_failures=1
      fi
    fi
  done
fi

# ============================================================================
# Summary and exit
# ============================================================================

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
