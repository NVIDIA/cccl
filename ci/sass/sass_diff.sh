#!/usr/bin/env bash

# Compare CUB benchmark SASS between two git refs.
#
# The script creates a worktree for each ref and builds the selected benchmark objects.
# It dumps their SASS with `cuobjdump -sass` and compares the results.
#
# Exit status reports execution failures, not SASS changes. A successful comparison
# returns 0 even when the SASS changes. Build failures can return 1, so that status
# cannot identify a SASS change. Read `any(.targets[].changed)` from result/report.json
# for the comparison result.
set -euo pipefail

usage()
{
  cat <<EOF
Usage: $0 <base-ref> <test-ref> [options]

Compare CUB benchmark SASS between two git refs, for example:

  $0 origin/main HEAD -arch all-major-cccl

Options:
  -preset <name>            CMake preset. Default: "cub-benchmark".
  -target-filter <regex>    Regex matched against the build target names
                            (repeatable). Default: "^cub\\\\.bench\\\\.".
  -target-filters-json <j>  The same filters as a JSON array. Used by CI, which
                            reads them from ci/matrix.yaml.
  -output-dir <path>        Artifact directory. Default: "<cwd>/sass-artifacts".
  -render                   Also write result/summary.md and print it. CI does
                            not use this: it renders the summary itself, because
                            the link in it needs the artifact URL, and that URL
                            exists only after CI uploaded the artifact.

Every other option goes to ci/build_common.sh; run it with -h for the list.
-configure is not among them: both sides must be built to be compared. Note that
the cub-benchmark preset sets the architecture to "native", so -arch must be
given for a multi-architecture comparison.

The artifact directory holds the raw dumps under base/ and test/. Under result/
it holds the normalized text the comparison acted on (base/ and test/), the
unified diff of every changed architecture (diff/), report.json and meta.json.
With -render it also holds summary.md.
EOF
}

if [[ "$#" -lt 2 ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
# Keep ci_dir writable because pretty_printing.sh also assigns it.
ci_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${ci_dir}/.." && pwd)"
readonly repo_root

BASE_REF="$1"
TEST_REF="$2"
shift 2

PRESET="cub-benchmark"
OUTPUT_DIR="${PWD}/sass-artifacts"
RENDER=0
TARGET_FILTERS=()
declare -a common_args=()

# build_common.sh parses the remaining options through its positional arguments.
while (($#)); do
  case "$1" in
    -h|-help|--help)           usage; exit 0 ;;
    -preset)                   PRESET="$2"; shift 2 ;;
    -output-dir)               OUTPUT_DIR="$2"; shift 2 ;;
    -render)                   RENDER=1; shift ;;
    -target-filter)            TARGET_FILTERS+=("$2"); shift 2 ;;
    -configure)
      echo "-configure cannot be used: both sides must be built to compare." >&2
      exit 1
      ;;
    -target-filters-json)
      # CI passes the JSON array from ci/matrix.yaml without converting it to flags.
      # Keep an empty array distinct from an empty filter, which matches every target.
      # mapfile does not propagate jq failures through process substitution.
      # The length check rejects empty output from jq, including parse failures.
      mapfile -t json_filters < <(jq -er '.[]' <<< "$2")
      if [[ "${#json_filters[@]}" -eq 0 ]]; then
        echo "-target-filters-json needs a non-empty JSON array, got: $2" >&2
        exit 1
      fi
      TARGET_FILTERS+=("${json_filters[@]}")
      shift 2
      ;;
    *) common_args+=("$1"); shift ;;
  esac
done

if [[ "${#TARGET_FILTERS[@]}" -eq 0 ]]; then
  TARGET_FILTERS=('^cub\.bench\.')
fi
# Combine filters so each target needs one grep invocation.
filter_regex="$(IFS='|'; echo "${TARGET_FILTERS[*]}")"
readonly filter_regex

# build_common.sh declares readonly globals and cannot be sourced twice in one process.
# Each run_side subshell sources its own copy. The parent only needs logging helpers.
# shellcheck source=ci/pretty_printing.sh
source "${ci_dir}/pretty_printing.sh"

# ============================================================================
# Test the comparison scripts
# ============================================================================

if [[ "${CI:-false}" != 'false' ]]; then
  run_command "🗜️  Install pytest for CI" python3 -m pip install -U pytest
fi

# Catch script failures before starting the benchmark builds.
run_command "🧪 Test SASS scripts" python3 -m pytest "${script_dir}"

# ============================================================================
# Set up both worktrees
# ============================================================================

# The base ref can name a remote branch that is not available locally.
# Print both refs because rev-parse does not identify invalid refs in its error message.
git -C "${repo_root}" fetch --no-tags origin "${BASE_REF}" >/dev/null 2>&1 || true
echo "Resolving ${BASE_REF} and ${TEST_REF}..."
base_commit="$(git -C "${repo_root}" rev-parse --verify "${BASE_REF}^{commit}")"
test_commit="$(git -C "${repo_root}" rev-parse --verify "${TEST_REF}^{commit}")"

mkdir -p "${OUTPUT_DIR}"/{base,test,result}
artifact_dir="$(cd "${OUTPUT_DIR}" && pwd)"
readonly artifact_dir

# Source paths appear in preprocessed output and affect sccache keys.
# Fixed worktree paths allow cache reuse across runs; random paths from mktemp prevent it.
readonly worktree_root="${repo_root}/build/sass-worktrees"
readonly base_path="${worktree_root}/base"
readonly test_path="${worktree_root}/test"

echo "Base ref:  ${BASE_REF} (${base_commit})"
echo "Test ref:  ${TEST_REF} (${test_commit})"
echo "Preset:    ${PRESET}"
echo "Artifacts: ${artifact_dir}"

# The EXIT trap removes worktrees on normal exit. Signal traps also handle CI cancellation,
# which otherwise leaves the worktrees registered.
# shellcheck disable=SC2329  # Invoked indirectly by the traps below.
cleanup()
{
  git -C "${repo_root}" worktree remove --force "$1" >/dev/null 2>&1 || true;
}

# shellcheck disable=SC2329  # Invoked indirectly by the traps below.
cleanup_all()
{
  cleanup "${base_path}"
  cleanup "${test_path}"
  rm -rf "${worktree_root}"
  # The EXIT trap supplies no argument and preserves the script's exit status.
  # Signal handlers supply an explicit status. Clear traps before exiting to avoid calling cleanup twice.
  if [[ "$#" -gt 0 ]]; then
    trap - EXIT HUP INT TERM
    exit "$1"
  fi
}
trap cleanup_all EXIT
trap 'cleanup_all 129' HUP
trap 'cleanup_all 130' INT
trap 'cleanup_all 143' TERM

declare -A side_path=([base]="${base_path}" [test]="${test_path}")
declare -A side_commit=([base]="${base_commit}" [test]="${test_commit}")

for side in base test; do
  # SIGKILL can leave worktree files and registrations behind.
  # Prune removes stale registrations after rm removes the files.
  cleanup "${side_path[${side}]}"
  rm -rf "${side_path[${side}]}"
  git -C "${repo_root}" worktree prune
  git -C "${repo_root}" worktree add --detach \
    "${side_path[${side}]}" "${side_commit[${side}]}" >/dev/null
  # Use identical presets to exclude configuration changes from the comparison.
  # Copy the file because CMake resolves ${sourceDir} from its real path.
  # A symlink points to the current checkout instead of the worktree.
  cp "${repo_root}/CMakePresets.json" "${side_path[${side}]}/CMakePresets.json"
done

# ============================================================================
# Build both sides
# ============================================================================

# build_common.sh derives its paths from its location in each worktree.
# A symlink to the current checkout makes it use that checkout's paths.
run_side()
{
  local side="$1"
  shift
  local -a cmd=("$@")
  (
    cd "${side_path[${side}]}"
    # `build_common.sh` parses `$@` as its own arguments.
    set -- "${common_args[@]}"
    source ci/build_common.sh
    "${cmd[@]}"
  )
}

# Both sides use the same toolchain, so one environment report covers both builds.
run_side base print_environment_details

declare -A preset_dir=()

for side in base test; do
  # shellcheck disable=SC2031  # `build_common.sh` shadows PRESET locally.
  run_side "${side}" configure_preset "SASS ${side}" "${PRESET}"
  # The preset's binaryDir.
  # shellcheck disable=SC2031
  preset_dir[${side}]="${side_path[${side}]}/build/${CCCL_BUILD_INFIX:+${CCCL_BUILD_INFIX}/}${PRESET}"
done

# Per cub/benchmarks/CMakeLists.txt: <path>/<stem>.cu -> cub.<path>.<stem>.base.
# Source paths provide target names without querying the configured build tree.
matching_targets()
{
  find "$1/cub/benchmarks" -name '*.cu' -printf '%P\n' \
    | sed -e 's/\.cu$/.base/' -e 's|/|.|g' -e 's/^/cub./' \
    | grep -E -- "${filter_regex}" \
    | sort -u
}

# A comparison requires the benchmark target in both refs.
mapfile -t targets < <(
  comm -12 <(matching_targets "${side_path[base]}") \
           <(matching_targets "${side_path[test]}")
)
# comm succeeds for an empty intersection, so check the result explicitly.
# Passing --target without names builds all targets.
if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "No CUB benchmark target common to both sides matched: ${filter_regex}" >&2
  exit 1
fi
echo "Selected ${#targets[@]} benchmark target(s)."

for side in base test; do
  object_targets=()

  # The intermediate objects already contain the SASS needed for comparison.
  # Building object targets directly avoids the executable link step.
  # CMake writes their paths to the manifests during generation.
  # dump_side also uses these objects to exclude kernels from linked helper libraries.
  for target in "${targets[@]}"; do
    mapfile -t objects < "${preset_dir[${side}]}/cub/benchmarks/objects/${target}.objects"

    if [[ "${#objects[@]}" -eq 0 ]]; then
      echo "No object files listed for benchmark target: ${target}" >&2
      exit 1
    fi

    for object in "${objects[@]}"; do
      # Ninja names object targets relative to the build directory.
      object_targets+=("${object#"${preset_dir[${side}]}/"}")
    done
  done

  # shellcheck disable=SC2031  # `build_common.sh` shadows PRESET locally.
  run_side "${side}" build_preset "SASS ${side}" "${PRESET}" --target "${object_targets[@]}"
done

# ============================================================================
# Dump and compare
# ============================================================================

# nvcc includes path hashes in names for internal-linkage and anonymous-namespace entities.
# These hashes differ between worktrees. cu++filt removes them to prevent false SASS differences.
#
# shellcheck disable=SC2329 # Invoked indirectly by `run_command`.
dump_side() {
  local side="$1"
  # Linked binaries include nvbench_helper kernels unrelated to the benchmark.
  # Dump only the CUDA objects for each benchmark target to exclude those kernels.
  # bash -c starts a new shell, so it needs its own pipefail setting.
  # Without pipefail, cu++filt can mask a cuobjdump failure and leave an empty dump.
  local dump_cmd
  # shellcheck disable=SC2016  # Variables expand in the child shell.
  printf -v dump_cmd '
    set -euo pipefail
    mapfile -t objects < %q/{}.objects
    if [[ "${#objects[@]}" -eq 0 ]]; then
      echo "No object files listed for benchmark target: {}" >&2
      exit 1
    fi
    for object in "${objects[@]}"; do
      cuobjdump -sass -sort "$object"
    done | cu++filt > %q/{}.sass
  ' "${preset_dir[${side}]}/cub/benchmarks/objects" "${artifact_dir}/${side}"

  printf '%s\n' "${targets[@]}" | xargs --verbose -P "$(nproc)" -I{} bash -c "${dump_cmd}"
}

for side in base test; do
  run_command "🔍 Dump SASS ${side}" dump_side "${side}"
done

# render_report.py needs the git refs and the architecture list from the build configuration.
# It cannot recover these from the SASS comparison report.
report_arch="$(
  awk -F= '/^CMAKE_CUDA_ARCHITECTURES:/ {print $2}' "${preset_dir[test]}/CMakeCache.txt"
)"

jq -n \
  --arg base_ref "${BASE_REF}" \
  --arg test_ref "${TEST_REF}" \
  --arg arch "${report_arch}" \
  '$ARGS.named' > "${artifact_dir}/result/meta.json"

# `compare_sass.py` returns 1 when the SASS changed, and status 2 for other failures.
compare_status=0
run_command "📊 Compare SASS" \
  python3 "${script_dir}/compare_sass.py" \
  --base-dir "${artifact_dir}/base" \
  --test-dir "${artifact_dir}/test" \
  --output-dir "${artifact_dir}/result" \
  --verbose || compare_status=$?

if [[ "${compare_status}" -ge 2 ]]; then
  echo "The SASS comparison failed with status ${compare_status}." >&2
  exit "${compare_status}"
fi

echo
echo "Wrote report: ${artifact_dir}/result/report.json"

if [[ "${RENDER}" -eq 1 ]]; then
  run_command "📝 Render report" \
    python3 "${script_dir}/render_report.py" \
    --report "${artifact_dir}/result/report.json" \
    --meta "${artifact_dir}/result/meta.json" \
    --output "${artifact_dir}/result/summary.md"

  echo
  cat "${artifact_dir}/result/summary.md"
  echo
  echo "Wrote summary: ${artifact_dir}/result/summary.md"
fi

print_time_summary

if [[ "${compare_status}" -ne 0 ]]; then
  echo "The SASS changed. See ${artifact_dir}/result/." >&2
fi
