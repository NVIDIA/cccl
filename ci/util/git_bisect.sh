#!/usr/bin/env bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ci_dir/.."

usage() {
  cat <<USAGE
Usage: $0 [--preset NAME | --configure-override CMD] [options]

Generic Options:

  -h, --help             Show this help and exit

Bisection Options:

  --good-ref STR         Good ref/sha/tag/branch. Defaults to latest release tag.
                         Accepts '-Nd' (e.g., '-14d') to mean 'origin/main as of N days ago'.
  --bad-ref STR          Bad ref/sha/tag/branch. Defaults to origin/main.
                         Accepts '-Nd' (e.g., '-14d') to mean 'origin/main as of N days ago'.
  --summary-file PATH    Markdown summary output path (optional)
                         No summary file will be generated if this is omitted.
Build / Test Options:

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
  --repeat N               Re-run the build/test for passing commits N times (default: 1)
USAGE
}

start_timestamp=${SECONDS}

function elapsed_time {
  local duration=$(( SECONDS - start_timestamp ))
  local minutes=$(( duration / 60 ))
  local seconds=$(( duration % 60 ))
  printf "%dm%02ds" "$minutes" "$seconds"
}

GOOD_REF=""
BAD_REF=""
PRESET=""
BUILD_TARGETS=""
CTEST_TARGETS=""
LIT_PRECOMPILE_TESTS=""
LIT_TESTS=""
SUMMARY_FILE=""
CMAKE_OPTIONS=""
CONFIGURE_OVERRIDE=""
CUSTOM_TEST_CMD=""
REPEAT=1

# Basic arg parser (keep simple, no extras)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --preset)        PRESET="${2:-}"; shift 2 ;;
    --good-ref)      GOOD_REF="${2:-}"; shift 2 ;;
    --bad-ref)       BAD_REF="${2:-}"; shift 2 ;;
    --build-targets) BUILD_TARGETS="${2:-}"; shift 2 ;;
    --ctest-targets) CTEST_TARGETS="${2:-}"; shift 2 ;;
    --lit-precompile-tests) LIT_PRECOMPILE_TESTS="${2:-}"; shift 2 ;;
    --lit-tests)     LIT_TESTS="${2:-}"; shift 2 ;;
    --cmake-options) CMAKE_OPTIONS="${2:-}"; shift 2 ;;
    --configure-override) CONFIGURE_OVERRIDE="${2:-}"; shift 2 ;;
    --summary-file)  SUMMARY_FILE="${2:-}"; shift 2 ;;
    --custom-test-cmd) CUSTOM_TEST_CMD="${2:-}"; shift 2 ;;
    --repeat)        REPEAT="${2:-}"; shift 2 ;;
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

# Ensure the checkout has complete history and tags:
git fetch --unshallow > /dev/null 2>&1 || :
git fetch --tags > /dev/null 2>&1 || :

# Resolve good and bad refs
good_ref="${GOOD_REF}"
bad_ref="${BAD_REF}"

# Helper to resolve '-Nd' (N days ago on origin/main) to a SHA
_resolve_days_ago() {
  local spec="$1"
  local base_branch="origin/main"
  local n="${spec#-}"
  n="${n%d}"
  if [[ -z "$n" || ! "$n" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  local when
  when=$(date -u -d "$n days ago" '+%Y-%m-%d %H:%M:%S %z')
  git rev-list -n 1 --before="$when" "$base_branch"
}

# Resolve good_ref
if [[ -z "$good_ref" ]]; then
  good_ref=$(git tag --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1 || :)
  echo "Good ref defaulted to last release: $good_ref"
fi
if [[ "$good_ref" =~ ^-[0-9]+d$ ]]; then
  good_sha=$(_resolve_days_ago "$good_ref")
  if [[ -z "$good_sha" ]]; then
    echo "::error::Unable to resolve good_ref '$good_ref' to a commit on origin/main" >&2
    exit 1
  fi
  echo "Resolved good_ref '$good_ref' to origin/main @ $good_sha"
else
  if [[ -z "$good_ref" ]]; then
    echo "::error::Unable to determine good ref" >&2
    exit 1
  fi
  good_sha=$(git rev-parse "$good_ref")
fi

# Resolve bad_ref
if [[ -z "$bad_ref" ]]; then
  bad_ref="origin/main"
  echo "Bad ref defaulted to origin/main: $bad_ref"
fi
if [[ "$bad_ref" =~ ^-[0-9]+d$ ]]; then
  bad_sha=$(_resolve_days_ago "$bad_ref")
  if [[ -z "$bad_sha" ]]; then
    echo "::error::Unable to resolve bad_ref '$bad_ref' to a commit on origin/main" >&2
    exit 1
  fi
  echo "Resolved bad_ref '$bad_ref' to origin/main @ $bad_sha"
else
  bad_sha=$(git rev-parse "$bad_ref")
fi

# Copy the build-and-test runner to a temp file so it remains available as HEAD changes:
tmp_runner="$(mktemp /tmp/build-and-test-XXXXXX.sh)"
cp "${ci_dir}/util/build_and_test_targets.sh" "${tmp_runner}"
chmod +x "${tmp_runner}"

# If --repeat > 1, wrap the runner to repeat successful runs to detect flakiness.
bisect_runner="${tmp_runner}"
if [[ "${REPEAT}" =~ ^[0-9]+$ ]] && [[ "${REPEAT}" -gt 1 ]]; then
  tmp_repeat="$(mktemp /tmp/build-and-test-repeat-XXXXXX.sh)"
  cat > "${tmp_repeat}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

attempts=${REPEAT}
for ((i=1; i<=attempts; ++i)); do
  echo "::group::✅ build_and_test attempt \${i}/${REPEAT}"
  "${tmp_runner}" "\$@" || {
    echo "Attempt \${i} failed; marking commit as bad."
    exit 1
  }
  echo "::endgroup::"
done
exit 0
EOF
  chmod +x "${tmp_repeat}"
  bisect_runner="${tmp_repeat}"
  echo "Repeating successful runs ${REPEAT} times to check for flakiness."
fi

# Writer that always prints to stdout and tees to file if provided
write_summary() {
  if [[ -n "${SUMMARY_FILE}" ]]; then
    tee -a "${SUMMARY_FILE}"
  else
    cat
  fi
}

echo "Starting bisect with:"
echo "  BAD_SHA:  $bad_sha"
echo "  GOOD_SHA: $good_sha"

echo "::group::⚙️ Starting git bisect"
(set -x; git bisect start "$bad_sha" "$good_sha")
echo "::endgroup::"

bisect_log="$(mktemp /tmp/git-bisect-log-XXXXXX.log)"
bisect_output="$(mktemp /tmp/git-bisect-output-XXXXXX.log)"
(
  set -x
  git bisect run "${bisect_runner}" \
    --preset "${PRESET}" \
    --build-targets "${BUILD_TARGETS}" \
    --ctest-targets "${CTEST_TARGETS}" \
    --lit-precompile-tests "${LIT_PRECOMPILE_TESTS}" \
    --lit-tests "${LIT_TESTS}" \
    --cmake-options "${CMAKE_OPTIONS}" \
    --configure-override "${CONFIGURE_OVERRIDE}" \
    --custom-test-cmd "${CUSTOM_TEST_CMD}" \
    | tee "${bisect_output}" || :
  git bisect log | tee "${bisect_log}" || :
  git bisect reset || :
)

if grep -q " is the first bad commit" "${bisect_output}"; then
  bad_commit=$(awk '/ is the first bad commit/ {print $1}' "${bisect_output}")
  echo -e "\e[1;32mFound bad commit in $(elapsed_time): $bad_commit\e[0m"
  found=true
else
  echo -e "\e[1;31mNo bad commit found ($(elapsed_time)).\e[0m"
  found=false
fi

function print_repro {
  echo "### ♻️ Reproduction Steps"
  echo
  echo '```bash'
  if [[ -n "${LAUNCH_ARGS:-}" ]]; then
    echo "  .devcontainer/launch.sh \\"
    echo "    ${LAUNCH_ARGS} \\"
    echo "    -- \\"
  fi
  echo "    ./ci/util/build_and_test_targets.sh \\"

  declare -a build_test_vars=(
    PRESET
    CMAKE_OPTIONS
    CONFIGURE_OVERRIDE
    BUILD_TARGETS
    CTEST_TARGETS
    LIT_PRECOMPILE_TESTS
    LIT_TESTS
    CUSTOM_TEST_CMD
  )

  # only print the above vars if they're non-empty.
  first=true
  for var in "${build_test_vars[@]}"; do
    if [[ -n "${!var:-}" ]]; then
      flag="--${var,,}"
      flag=${flag//_/-}  # replace _ with -
      if ! $first; then
        echo " \\" # Trailing "\" to escape newlines
      fi
      echo -n "       ${flag} \"${!var}\""
      first=false
    fi
  done
  echo # Final newline for last argument
  echo '```'
}

if [[ "${found}" == "true" ]]; then
  commit_info=$(git log "$bad_commit" -1 --pretty=format:'%h %s')
  pr_ref=$(echo "$commit_info" | grep -oE '#[0-9]+' | head -n1 || :)
  (
    echo "## 🔎 Bisect Result"
    echo
    echo "- Culprit Commit: $commit_info"
    if [[ -n "$pr_ref" ]]; then
      pr_num=${pr_ref#\#}
      echo "- Culprit PR: https://github.com/NVIDIA/cccl/pull/$pr_num"
    fi
    echo "- Commit SHA: $bad_commit"
    echo "- Commit URL: https://github.com/NVIDIA/cccl/commit/${bad_commit}"
    if [[ -n "${GHA_LOG_URL:-}" ]]; then
      echo "- Bisection Logs: [GHA Job](${GHA_LOG_URL})"
      echo "- Bisection Summary: [GHA Report](${STEP_SUMMARY_URL})"
    fi
    echo
    print_repro
    echo
    echo "### ℹ️ Commit Details"
    echo
    echo '```'
    git show "$bad_commit" --stat
    echo '```'
    echo
    echo "### 🪵 Bisect Log"
    echo
    echo '```'
    cat "${bisect_log}"
    echo '```'
  ) | write_summary
else
  (
    echo "## ‼️ Bisect Failed"
    echo
    echo "git bisect did not resolve to a single commit."
    echo
    if [[ -n "${GHA_LOG_URL:-}" ]]; then
      echo "- Bisection Logs: [GHA Job](${GHA_LOG_URL})"
      echo "- Bisection Summary: [GHA Report](${STEP_SUMMARY_URL})"
    fi
    echo
    print_repro
    echo
    echo "### 🪵 Bisect Log"
    echo
    echo '```'
    cat "${bisect_log}"
    echo '```'
  ) | write_summary
  exit 1
fi
