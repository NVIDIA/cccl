#!/usr/bin/env bash

# Compare the SASS of the CUB benchmarks between two git refs.
#
# The script adds a worktree for each ref, builds the selected benchmark targets
# in both, dumps the disassembly of every built binary with `cuobjdump -sass`,
# and compares the result.
#
# The exit status tells you only if the comparison ran. It is 0 when the script
# wrote a report, and it is not 0 when the script did not write a report.
#
# The exit status does not tell you if the SASS changed. `set -e` ends the script
# with the status of the command that failed, and a failed benchmark build gives
# status 1. Thus status 1 cannot show a SASS change. To find the result, read
# `any(.targets[].changed)` from result/report.json.
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
# can't make this readonly because pretty_printing.sh also declares a variable of the same
# name...
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

# Take the sass-specific options and leave the rest for `build_common.sh`, which
# parses `$@` as its own.
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
      # The CI job has the filters as a JSON array, from ci/matrix.yaml. Taking them
      # directly saves the workflow from unpacking them into flags. An empty array must
      # not become one empty filter, which matches everything. `mapfile` does not see the
      # exit status of jq, so an empty result covers both a parse error and an empty
      # array.
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
# One alternation, so a target is matched with a single grep.
filter_regex="$(IFS='|'; echo "${TARGET_FILTERS[*]}")"
readonly filter_regex

# `build_common.sh` declares readonly globals, so it can only be sourced once per
# process. Each side sources its own copy in `run_side`; the parent needs the
# logging helpers only.
# shellcheck source=ci/pretty_printing.sh
source "${ci_dir}/pretty_printing.sh"

# ============================================================================
# Test the comparison scripts
# ============================================================================

if [[ "${CI:-false}" != 'false' ]]; then
  run_command "🗜️  Install pytest for CI" python3 -m pip install -U pytest
fi

# Before the builds, because a broken script would otherwise be found only after them.
run_command "🧪 Test SASS scripts" python3 -m pytest "${script_dir}"

# ============================================================================
# Set up both worktrees
# ============================================================================

# The base ref can be a remote branch that was not fetched yet. `rev-parse` does not name
# the ref it rejected, so print both here.
git -C "${repo_root}" fetch --no-tags origin "${BASE_REF}" >/dev/null 2>&1 || true
echo "Resolving ${BASE_REF} and ${TEST_REF}..."
base_commit="$(git -C "${repo_root}" rev-parse --verify "${BASE_REF}^{commit}")"
test_commit="$(git -C "${repo_root}" rev-parse --verify "${TEST_REF}^{commit}")"

mkdir -p "${OUTPUT_DIR}"/{base,test,result}
artifact_dir="$(cd "${OUTPUT_DIR}" && pwd)"
readonly artifact_dir

# A fixed path, never `mktemp -d`. The path of the compilation unit reaches the
# preprocessed source, so it is part of the sccache key. With a fresh random path
# per run, no run could ever hit what an earlier run stored, and every object was
# compiled cold on both sides.
readonly worktree_root="${repo_root}/build/sass-worktrees"
readonly base_path="${worktree_root}/base"
readonly test_path="${worktree_root}/test"

echo "Base ref:  ${BASE_REF} (${base_commit})"
echo "Test ref:  ${TEST_REF} (${test_commit})"
echo "Preset:    ${PRESET}"
echo "Artifacts: ${artifact_dir}"

# Remove the worktrees on a normal exit and on the signals CI sends when a job is
# cancelled; without the signal traps a cancelled job leaves them registered.
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
  # The EXIT trap calls this with no argument, and the script then continues to its own
  # exit. A signal trap gives the status to exit with. Remove the traps first, so that the
  # exit below does not call this function a second time.
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
  # The path is fixed, so a run that was killed can have left it behind. The prune drops
  # the registration that `rm -rf` leaves stale.
  cleanup "${side_path[${side}]}"
  rm -rf "${side_path[${side}]}"
  git -C "${repo_root}" worktree prune
  git -C "${repo_root}" worktree add --detach \
    "${side_path[${side}]}" "${side_commit[${side}]}" >/dev/null
  # Pin the build configuration to the current tree, so a preset change is not measured as
  # a code change. Copy, never symlink: CMake resolves ${sourceDir} from the real path of
  # this file, and a symlink would point it at the current checkout instead of the
  # worktree.
  cp "${repo_root}/CMakePresets.json" "${side_path[${side}]}/CMakePresets.json"
done

# ============================================================================
# Build both sides
# ============================================================================

# `build_common.sh` derives its paths from where it is sourced. Never symlink
# into a worktree: that resolves back to the current checkout.
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

# Both sides use the same toolchain, so one report is enough.
run_side base print_environment_details

declare -A preset_dir=()

for side in base test; do
  # shellcheck disable=SC2031  # `build_common.sh` shadows PRESET locally.
  run_side "${side}" configure_preset "SASS ${side}" "${PRESET}"
  # The preset's binaryDir; `run_side` only reads these, never assigns them.
  # shellcheck disable=SC2031
  preset_dir[${side}]="${side_path[${side}]}/build/${CCCL_BUILD_INFIX:-}/${PRESET}"
done

# Per cub/benchmarks/CMakeLists.txt: <path>/<stem>.cu -> cub.<path>.<stem>.base.
# Reading the source tree means no configured build tree is needed here.
matching_targets()
{
  find "$1/cub/benchmarks" -name '*.cu' -printf '%P\n' \
    | sed -e 's/\.cu$/.base/' -e 's|/|.|g' -e 's/^/cub./' \
    | grep -E -- "${filter_regex}" \
    | sort -u
}

# Only the targets both sides have can be compared. A target that only one side
# has is reported separately by compare_sass.py.
mapfile -t targets < <(
  comm -12 <(matching_targets "${side_path[base]}") \
           <(matching_targets "${side_path[test]}")
)
# An empty result is a successful `comm`, thus this code must find it. If it does not,
# `--target` gets no target and the build makes all the targets.
if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "No CUB benchmark target common to both sides matched: ${filter_regex}" >&2
  exit 1
fi
echo "Selected ${#targets[@]} benchmark target(s)."

for side in base test; do
  # shellcheck disable=SC2031  # `build_common.sh` shadows PRESET locally.
  run_side "${side}" build_preset "SASS ${side}" "${PRESET}" --target "${targets[@]}"
done

# ============================================================================
# Dump and compare
# ============================================================================

# `cu++filt` strips the path hash that nvcc puts in the name of an internal-linkage or
# anonymous-namespace entity. Both differ between the two worktrees, so without it every
# such kernel compares as changed.
#
# shellcheck disable=SC2329 # Invoked indirectly by `run_command`.
dump_side() {
  local side="$1"
  # `pipefail` again, because `bash -c` starts a fresh shell. Without it a failed
  # `cuobjdump` writes an empty dump and still reports success.
  local dump_cmd="set -eou pipefail; cuobjdump -sass -sort '${preset_dir[${side}]}/bin/{}' | cu++filt > '${artifact_dir}/${side}/{}.sass'"

  printf '%s\n' "${targets[@]}" | xargs --verbose -P "$(nproc)" -I{} bash -c "${dump_cmd}"
}

for side in base test; do
  run_command "🔍 Dump SASS ${side}" dump_side "${side}"
done

# `render_report.py` cannot work out the refs that were compared and the architectures the
# build really used.
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
