#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  echo "This script must be run in a GitHub Actions environment." >&2
  exit 1
fi

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/.." && pwd)"
readonly repo_root

cd "$repo_root"

if ! ci/util/workflow/has_consumers.sh; then
  echo "No consumers found for this job. Exiting." >&2
  exit 0
fi

artifact="z_libcudacxx-test-artifacts-${DEVCONTAINER_NAME:?}-${JOB_ID:?}"

# BUILD_INFIX is undefined on windows CI.
build_dir_regex="build${CCCL_BUILD_INFIX:+/$CCCL_BUILD_INFIX}/libcudacxx[^/]*"
lit_executable_regex="$build_dir_regex/libcudacxx/test/libcudacxx/test/.*Output/.*\.exe$"

# Minimum CTest/lit metadata needed to run from the unpacked build tree.
ci/util/artifacts/stage.sh "$artifact" \
    "$build_dir_regex/build\.ninja$" \
    "$build_dir_regex/.*rules\.ninja$" \
    "$build_dir_regex/CMakeCache\.txt$" \
    "$build_dir_regex/.*VerifyGlobs\.cmake$" \
    "$build_dir_regex/.*CTestTestfile\.cmake$" \
    "$build_dir_regex/libcudacxx/test/libcudacxx/lit\.site\.cfg$" \
    > /dev/null

# Test executables plus shared libraries/smoke binaries used by CTest/lit.
ci/util/artifacts/stage.sh "$artifact" \
    "$build_dir_regex/bin/.*" \
    "$build_dir_regex/lib/.*" \
    "$lit_executable_regex" \
    > /dev/null

if ! find . -type f -regex "\./$lit_executable_regex" -print -quit | grep -q .; then
  echo "No lit test executables found for artifact '$artifact'." >&2
  exit 1
fi

# Windows builds generate binaries for header tests that are never executed.
ci/util/artifacts/unstage.sh \
    "$artifact" \
    "$build_dir_regex/.*\.headers\..*" > /dev/null || :

ci/util/artifacts/upload_stage_packed.sh "$artifact"
