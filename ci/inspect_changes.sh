#!/bin/bash

# Github action script to identify which subprojects are dirty in a PR

set -euo pipefail

# Print error message with script name and line number
trap 'echo "Error in ${BASH_SOURCE[0]}:${LINENO}"' ERR
set -o errtrace  # Ensure ERR trap propagates in functions and subshells

# Usage: inspect_changes.sh <base_sha> <head_sha>
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <base_sha> <head_sha> [<summary.md>]"
  exit 1
fi

base_sha=$1
head_sha=$2
summary_md=${3:-}

if [[ -n "$summary_md" ]]; then
  echo "Summary will be written to $summary_md"
  > "${summary_md}"
fi

# Unshallow repo to make it possible to trace the common ancestor.
if git rev-parse --is-shallow-repository &>/dev/null && git rev-parse --is-shallow-repository | grep -q true; then
  git fetch origin --unshallow -q
fi

# Github gives the SHA as the current HEAD of the target ref, not the common ancestor.
# Find the common ancestor and use that for the base.
base_sha=$(git rev-parse "$base_sha" || echo "$base_sha")
head_sha=$(git rev-parse "$head_sha" || echo "$head_sha")
git fetch origin $base_sha -q
git fetch origin $head_sha -q
base_sha=$(git merge-base $base_sha $head_sha)

# Define a list of subproject directories by their subdirectory name:
subprojects=(
  cccl
  packaging
  libcudacxx
  cub
  thrust
  cudax
  stdpar
  python
  cccl_c_parallel
  cccl_c_stf
  c2h
  nvbench_helper
)

# ...and their dependencies.
# Mapped as: key project is rebuilt if any value project is dirty.
declare -A dependencies=(
  [cccl]=""
  [packaging]="cccl libcudacxx cub thrust cudax"
  [libcudacxx]="cccl"
  [cub]="cccl libcudacxx thrust c2h nvbench_helper"
  [thrust]="cccl libcudacxx cub nvbench_helper"
  [cudax]="cccl libcudacxx thrust cub c2h nvbench_helper"
  [stdpar]="cccl libcudacxx cub thrust"
  [python]="cccl libcudacxx cub cccl_c_parallel"
  [cccl_c_parallel]="cccl libcudacxx cub thrust c2h"
  [cccl_c_stf]="cccl libcudacxx cudax c2h"
  [c2h]="cccl libcudacxx cub thrust"
  [nvbench_helper]="cccl libcudacxx cub thrust"
)

declare -A project_names=(
  [cccl]="CCCL Infrastructure"
  [packaging]="CCCL Packaging"
  [libcudacxx]="libcu++"
  [cub]="CUB"
  [thrust]="Thrust"
  [cudax]="CUDA Experimental"
  [stdpar]="stdpar"
  [python]="python"
  [cccl_c_parallel]="CCCL C Parallel Library"
  [cccl_c_stf]="CCCL C CUDASTF Library"
  [c2h]="Catch2Helper"
  [nvbench_helper]="NVBench Helper"
)

# By default, the project directory is assumed to be the same as the subproject name,
# but can be overridden here. The `cccl` project is special, and checks for files outside
# of any subproject directory.
#
# *** No trailing slashes, they break things. ***
#
declare -A project_dirs=(
  [packaging]='("examples" "test/cmake")'
  [cccl_c_parallel]='("c/parallel")'
  [cccl_c_stf]='("c/experimental/stf")'
  [stdpar]='("test/stdpar")'
)

# Changes to files / directories listed here are ignored when checking if the
# CCCL Infrastructure has been modified.
# These are checked as regexes that match the beginning of the file path.
ignore_paths=(
  ".clang-format"
  ".clangd"
  ".devcontainer/img"
  ".devcontainer/README.md"
  ".git-blame-ignore-revs"
  ".github/actions/docs-build"
  ".github/CODEOWNERS"
  ".github/copilot-instructions.md"
  ".github/copy-pr-bot.yaml"
  ".github/ISSUE_TEMPLATE"
  ".github/PULL_REQUEST_TEMPLATE.md"
  ".github/workflows/backport-prs.yml"
  ".github/workflows/build-docs.yml"
  ".github/workflows/build-rapids.yml"
  ".github/workflows/project_automation" # All project automation workflows
  ".github/workflows/triage_rotation.yml"
  ".github/workflows/update_branch_version.yml"
  ".github/workflows/verify-devcontainers.yml"
  ".gitignore"
  "ci-overview.md"
  "AGENTS.md"
  "CITATION.md"
  "CODE_OF_CONDUCT.md"
  "CONTRIBUTING.md"
  "docs"
  "LICENSE"
  "README.md"
  "SECURITY.md"
)

# Usage checks:
for subproject in "${subprojects[@]}"; do
  # Check that the subproject directory exists
  if [ "$subproject" != "cccl" ]; then
    # project_dirs[$subproject] may be a list of paths, or fallback to $subproject
    if [[ -n "${project_dirs[$subproject]:-}" ]]; then
      eval "dirs=${project_dirs[$subproject]}"
    else
      dirs=("$subproject")
    fi
    for dir in "${dirs[@]}"; do
      if [ ! -d "$dir" ]; then
        echo "Error: Subproject '$subproject' directory '$dir' does not exist."
        exit 1
      fi
    done
  fi

  # If the subproject has dependencies, check that they are valid
  for dependency in ${dependencies[$subproject]}; do
    if [ "$dependency" != "cccl" ]; then
      if [[ ! " ${subprojects[@]} " =~ " ${dependency} " ]]; then
        echo "Error: Dependency '$dependency' for subproject '$subproject' does not exist."
        exit 1
      fi
    fi
  done
done

write_output() {
  local key="$1"
  local value="$2"
  echo "$key=$value" | tee --append "${GITHUB_OUTPUT:-/dev/null}"
}

tee_to_summary() {
  if [ -n "${summary_md}" ]; then
    tee -a "${summary_md}"
  else
    cat
  fi
}

dirty_files() {
  git diff --name-only "${base_sha}" "${head_sha}"
}

full_build_projects=() # Actually dirty -- use eg. `pull_request` matrix
lite_build_projects=() # Upstream dependencies are dirty -- use eg. `pull_request_lite` matrix

# in_array <needle> <haystack...>
in_array() {
  local needle=$1; shift
  local x
  for x in "$@"; do
    [[ $x == "$needle" ]] && return 0
  done
  return 1
}

project_is_full_build() {
  local subproject="$1"
  in_array "$subproject" "${full_build_projects[@]}"
}

project_is_lite_build() {
  local subproject="$1"
  in_array "$subproject" "${lite_build_projects[@]}"
}

# Returns true if any non-excluded files outside of subproject directories are dirty.
# These are treated as changes to the core infrastructure, and a full dirty rebuild is triggered.
core_infra_is_dirty() {
  # Build a list of exclusions (subproject dirs and ignore_paths) for grep
  exclusions=()
  for subproject in "${subprojects[@]}"; do
    if [[ ${subproject} == "cccl" ]]; then
      continue
    fi
    # project_dirs[$subproject] may be a list of custom paths
    if [[ -n "${project_dirs[$subproject]:-}" ]]; then
      eval "dirs=${project_dirs[$subproject]}"
      for dir in "${dirs[@]}"; do
        exclusions+=("$dir")
      done
    else
      # Otherwise, the directory is the same as the subproject name
      exclusions+=("$subproject")
    fi
  done

  # Manual exclusions:
  exclusions+=("${ignore_paths[@]}")

  # Build grep pattern: ^(dir1|dir2|...|file1|file2)
  grep_pattern="^($(IFS="|"; echo "${exclusions[*]}"))"

  cccl_infra_dirty_files=$(dirty_files | grep -v -E "${grep_pattern}")

  [[ -n "${cccl_infra_dirty_files}" ]]
}

# subdir_is_dirty <subdir>
subdir_is_dirty() {
  local subdir="$1"
  subdir_dirt="$(dirty_files | grep -E "^${subdir}/")"
  [[ -n "$subdir_dirt" ]]
}

project_is_dirty() {
  local subproject="$1"
  subproject_dirs=()
  if [[ -n "${project_dirs[$subproject]:-}" ]]; then
    eval "subproject_dirs=${project_dirs[$subproject]}"
  else
    subproject_dirs=("$subproject")
  fi

  for dir in "${subproject_dirs[@]}"; do
    if subdir_is_dirty "$dir"; then
      return 0
    fi
  done
  return 1
}

main() {

  echo "Base SHA: ${base_sha}"
  git log --oneline -1 "${base_sha}" | sed 's/^/  /'
  echo "HEAD SHA: ${head_sha}"
  git log --oneline -1 "${head_sha}" | sed 's/^/  /'
  echo

  # Print the list of subprojects and all of their dependencies:
  echo "Subprojects: ${subprojects[*]}"
  echo
  echo "Dependencies:"
  for subproject in "${subprojects[@]}"; do
    printf "  - %-27s -> %s\n" "$subproject (${project_names[$subproject]})" "${dependencies[$subproject]}"

  done
  echo

  # Determine which projects have dirt.
  # First check core infra -- this will trigger a full rebuild of everything.
  echo "Checking for changes..."
  if core_infra_is_dirty; then
    echo "- Core infrastructure changes detected -- all projects will be rebuilt."
    full_build_projects=("${subprojects[@]}")
  else
    # Check each subproject for changes:
    for subproject in "${subprojects[@]}"; do
      if [[ ${subproject} == "cccl" ]]; then
        # Special case handled above.
        continue
      fi
      if project_is_dirty "$subproject"; then
        echo "- Changes detected in subproject '$subproject' (${project_names[$subproject]})"
        full_build_projects+=("$subproject")
      fi
    done

    # Now add all dependees of dirty projects to the lite build:
    for dirty_project in "${full_build_projects[@]}"; do
      for subproject in "${subprojects[@]}"; do
        if [[ " ${dependencies[$subproject]} " == *" $dirty_project "* ]]; then
          if ! project_is_full_build "$subproject" && ! project_is_lite_build "$subproject"; then
            echo "- Upstream dependency change detected: '$subproject' (${project_names[$subproject]}) depends on dirty project '$dirty_project'"
            lite_build_projects+=("$subproject")
          fi
        fi
      done
    done
  fi
  echo

  echo "Github Action Outputs:"
  write_output "FULL_BUILD" "${full_build_projects[*]}"
  write_output "LITE_BUILD" "${lite_build_projects[*]}"
  echo

  echo "::group::Project Change Summary"
  echo "<details><summary><h3>👃 Inspect Project Changes</h3></summary>" | tee_to_summary
  echo | tee_to_summary

  echo "| Project                     | Status     |" | tee_to_summary
  echo "|-----------------------------|------------|" | tee_to_summary
  for subproject in "${subprojects[@]}"; do
    if project_is_full_build "$subproject"; then
      status="Dirty"
    elif project_is_lite_build "$subproject"; then
      status="Dirty Deps"
    else
      status="Clean"
    fi
    printf "| %-27s | %-10s |\n" "${project_names[$subproject]}" "$status" | tee_to_summary
  done

  echo | tee_to_summary
  echo "<details><summary><h4>👉 Dirty Files</h4></summary>" | tee_to_summary
  echo | tee_to_summary
  dirty_files | sed 's/^/  - /' | tee_to_summary
  echo | tee_to_summary
  echo "</details>" | tee_to_summary

  echo "</details>" | tee_to_summary
  echo "::endgroup::"
}

main "$@"
