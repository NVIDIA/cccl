#!/usr/bin/env bash
set -euo pipefail

# This must be run from the cccl repo root, but the script may be relocated by git_bisect.sh.
# Check that the current directory looks like the repo root:
if [[ ! -f "./cccl-version.json" ]]; then
  echo "This script must be run from the cccl repo root."
  exit 1
fi

usage() {
  cat <<USAGE
Usage: $0 [--preset NAME | --configure-override CMD] [options]

Options:
  -h, --help                Show this help and exit
  --preset NAME             CMake preset
  --cmake-options STR       Extra options passed to CMake preset configure (optional)
  --configure-override CMD  Command to run for configuration instead of cmake preset
                            If set, --preset and --cmake-options will be ignored
  --build-targets STR       Space separated ninja build targets (optional)
                            If omitted, no targets will be built -- explicitly specify 'all' if needed.
  --ctest-targets STR       Space separated CTest -R regex patterns (optional)
                            If omitted, no tests will be run -- explicitly specify '.' to run all.
  --lit-precompile-tests STR  Space-separated libcudacxx lit test paths to precompile without execution (optional)
                              e.g. 'cuda/utility/basic_any.pass.cpp'
  --lit-tests STR            Space-separated libcudacxx lit test paths to execute (optional)
                              e.g. 'cuda/utility/basic_any.pass.cpp'
  --custom-test-cmd CMD     Custom command run after build and tests (optional)
USAGE
}

start_timestamp=${SECONDS}

function elapsed_time {
  local duration=$(( SECONDS - start_timestamp ))
  local minutes=$(( duration / 60 ))
  local seconds=$(( duration % 60 ))
  printf "%dm%02ds" "$minutes" "$seconds"
}

PRESET=""
BUILD_TARGETS=""
CTEST_TARGETS=""
LIT_PRECOMPILE_TESTS=""
LIT_TESTS=""
CMAKE_OPTIONS=""
CONFIGURE_OVERRIDE=""
CUSTOM_TEST_CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --preset)        PRESET="${2:-}"; shift 2 ;;
    --build-targets) BUILD_TARGETS="${2:-}"; shift 2 ;;
    --ctest-targets) CTEST_TARGETS="${2:-}"; shift 2 ;;
    --lit-precompile-tests) LIT_PRECOMPILE_TESTS="${2:-}"; shift 2 ;;
    --lit-tests)     LIT_TESTS="${2:-}"; shift 2 ;;
    --cmake-options) CMAKE_OPTIONS="${2:-}"; shift 2 ;;
    --configure-override) CONFIGURE_OVERRIDE="${2:-}"; shift 2 ;;
    --custom-test-cmd) CUSTOM_TEST_CMD="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PRESET}" && -z "${CONFIGURE_OVERRIDE}" ]]; then
  echo "::error:: --preset or --configure-override is required" >&2
  usage
  exit 2
fi

if [[ -n "${CONFIGURE_OVERRIDE}" ]]; then
  if [[ -n "${PRESET}" ]]; then
    echo "::warning:: --preset ignored due to --configure-override" >&2
  fi
  if [[ -n "${CMAKE_OPTIONS}" ]]; then
    echo "::warning:: --cmake-options ignored due to --configure-override" >&2
  fi
fi

echo "::group::⚙️ Testing $(git log --oneline | head -n1)"

# Configure and parse the build directory from CMake output
BUILD_DIR=""
cmlog_file="$(mktemp /tmp/cmake-config-XXXXXX.log)"
if [[ -n "${CONFIGURE_OVERRIDE}" ]]; then
  if ! (set -x; eval "${CONFIGURE_OVERRIDE}") 2>&1 | tee "${cmlog_file}"; then
    echo "::endgroup::"
    echo "🔴📝 Configuration override failed ($(elapsed_time)):\n\t${CONFIGURE_OVERRIDE}"
    exit 1
  fi
else
  read -r -a _cmake_opts <<< "${CMAKE_OPTIONS}"
  if ! (set -x; cmake --preset "${PRESET}" "${_cmake_opts[@]}") 2>&1 | tee "${cmlog_file}"; then
    echo "::endgroup::"
    echo "🔴📝 CMake configure failed for preset ${PRESET} ($(elapsed_time))"
    exit 1
  fi
fi
BUILD_DIR=$(awk -F': ' '/-- Build files have been written to:/ {print $2}' "${cmlog_file}" | tail -n1)
if [[ -z "${BUILD_DIR}" ]]; then
  echo "::endgroup::"
  echo "🔴‼️ Unable to determine build directory ($(elapsed_time))"
  exit 1
fi

if [[ -n "${BUILD_TARGETS}" ]]; then
  if ! (set -x; ninja -C "${BUILD_DIR}" ${BUILD_TARGETS}); then
    echo "::endgroup::"
    echo "🔴🛠️ Ninja build failed for targets ($(elapsed_time)): ${BUILD_TARGETS}"
    exit 1
  fi
fi

if [[ -n "${CTEST_TARGETS}" ]]; then
  for t in ${CTEST_TARGETS}; do
    if ! (set -x; ctest --test-dir "${BUILD_DIR}" -R "$t" -V --output-on-failure); then
      echo "::endgroup::"
      echo "🔴🔎 CTest failed for target $t ($(elapsed_time))"
      exit 1
    fi
  done
fi

if [[ -n "${LIT_PRECOMPILE_TESTS}" || -n "${LIT_TESTS}" ]]; then
  lit_site_cfg="${BUILD_DIR}/libcudacxx/test/libcudacxx/lit.site.cfg"
  if [[ ! -f "${lit_site_cfg}" ]]; then
    echo "::endgroup::"
    echo "🔴🧪 LIT site config not found ($(elapsed_time)): ${lit_site_cfg}"
    exit 1
  fi
fi

if [[ -n "${LIT_PRECOMPILE_TESTS}" ]]; then
  for t in ${LIT_PRECOMPILE_TESTS}; do
    t_path="libcudacxx/test/libcudacxx/${t}"
    if ! (set -x; LIBCUDACXX_SITE_CONFIG="${lit_site_cfg}" lit -v "-Dexecutor=NoopExecutor()" "${t_path}"); then
      echo "::endgroup::"
      echo "🔴🧪 LIT precompile failed ($(elapsed_time)): ${t}"
      exit 1
    fi
  done
fi

if [[ -n "${LIT_TESTS}" ]]; then
  for t in ${LIT_TESTS}; do
    t_path="libcudacxx/test/libcudacxx/${t}"
    if ! (set -x; LIBCUDACXX_SITE_CONFIG="${lit_site_cfg}" lit -v "${t_path}"); then
      echo "::endgroup::"
      echo "🔴🧪 LIT test failed ($(elapsed_time)): ${t}"
      exit 1
    fi
  done
fi

if [[ -n "${CUSTOM_TEST_CMD}" ]]; then
  if ! (set -x; eval "${CUSTOM_TEST_CMD}"); then
    echo "::endgroup::"
    echo "🔴🧪 Custom test command failed ($(elapsed_time)): ${CUSTOM_TEST_CMD}"
    exit 1
  fi
fi

echo "::endgroup::"
echo "🟢✅ Passed ($(elapsed_time))"
exit 0
