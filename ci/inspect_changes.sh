#!/bin/bash

# Github action script to identify which subprojects are dirty in a PR

set -euo pipefail

# Print error message with script name and line number
trap 'echo "Error in ${BASH_SOURCE[0]}:${LINENO}"' ERR
set -o errtrace  # Ensure ERR trap propagates in functions and subshells

# Usage: inspect_changes.sh <base_sha> <head_sha>
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base_sha> <head_sha>"
  exit 1
fi

base_sha=$1
head_sha=$2

# Unshallow repo to make it possible to trace the common ancestor.
if git rev-parse --is-shallow-repository &>/dev/null && git rev-parse --is-shallow-repository | grep -q true; then
  git fetch origin --unshallow -q
fi

# Github gives the SHA as the current HEAD of the target ref, not the common ancestor.
# Find the common ancestor and use that for the base.
head_sha=$(git rev-parse "$head_sha" || echo "$head_sha")
base_sha=$(git rev-parse "$base_sha" || echo "$base_sha")
git fetch origin $head_sha -q
git fetch origin $base_sha -q
base_sha=$(git merge-base $head_sha $base_sha)

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

tee_to_step_summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    tee -a "${GITHUB_STEP_SUMMARY}"
  else
    cat
  fi
}

dirty_files() {
  git diff --name-only "${base_sha}" "${head_sha}"
}

# Returns 0 if any files outside of the subproject directories have changed (i.e., CCCL infra is dirty), 1 otherwise
# This marks the `cccl` project dirty.
core_infra_is_dirty() {
  # Build a list of exclusions (subproject dirs and ignore_paths) for grep
  exclusions=()
  for subproject in "${subprojects[@]}"; do
    if [[ ${subproject} == "cccl" ]]; then
      continue
    fi
    # project_dirs[$subproject] may be a list of paths
    if [[ -n "${project_dirs[$subproject]:-}" ]]; then
      eval "dirs=${project_dirs[$subproject]}"
      for dir in "${dirs[@]}"; do
        exclusions+=("$dir")
      done
    else
      exclusions+=("$subproject")
    fi
  done

  # Manual exclusions:
  exclusions+=("${ignore_paths[@]}")

  # Build grep pattern: ^(dir1|dir2|...|file1|file2)
  grep_pattern="^($(IFS="|"; echo "${exclusions[*]}"))"

  cccl_infra_dirty_files=$(dirty_files | grep -v -E "${grep_pattern}")

  if [[ -n "${cccl_infra_dirty_files}" ]]; then
    return 0
  else
    return 1
  fi
}

# subdir_is_dirty <subdir>
# Returns 0 if any files in the subdirectory have changed
subdir_is_dirty() {
  local subdir="$1"

  subdir_dirt="$(dirty_files | grep -E "^${subdir}/")"

  if [[ -n "$subdir_dirt" ]]; then
    return 0
  else
    return 1
  fi
}

# project_or_deps_are_dirty <subproject>
# Returns 0 if the subproject or any of its dependencies are dirty, 1 otherwise
project_or_deps_are_dirty() {
  local subproject="$1"

  local dirty_flag="${subproject^^}_DIRTY"
  if [[ ${!dirty_flag} -ne 0 ]]; then
    return 0
  fi

  for dependency in ${dependencies[$subproject]}; do
    dirty_flag="${dependency^^}_DIRTY"
    if [[ ${!dirty_flag} -ne 0 ]]; then
      return 0
    fi
  done

  return 1
}

main() {
  # Print the list of subprojects and all of their dependencies:
  echo "Subprojects: ${subprojects[*]}"
  echo
  echo "Dependencies:"
  for subproject in "${subprojects[@]}"; do
    printf "  - %-27s -> %s\n" "$subproject (${project_names[$subproject]})" "${dependencies[$subproject]}"

  done
  echo

  echo "Base SHA: ${base_sha}"
  echo "HEAD SHA: ${head_sha}"
  echo

  check="+/-"
  no_check="   "
  get_checkmark() {
    if [[ $1 -eq 0 ]]; then
      echo "$no_check"
    else
      echo "$check"
    fi
  }

  # Print the list of files that have changed:
  echo "::group::Dirty files"
  dirty_files | sed 's/^/  - /'
  echo "::endgroup::"
  echo

  echo "<details><summary><h3>👃 Inspect Changes</h3></summary>" | tee_to_step_summary
  echo | tee_to_step_summary

  echo -e "### Modifications in project?\n" | tee_to_step_summary
  echo "|     | Project" | tee_to_step_summary
  echo "|-----|---------" | tee_to_step_summary

  CCCL_DIRTY=0
  if core_infra_is_dirty; then
    CCCL_DIRTY=1
  fi
  checkmark="$(get_checkmark ${CCCL_DIRTY})"
  echo "| ${checkmark} | ${project_names[cccl]}" | tee_to_step_summary

  # Check for changes in each subprojects directory:
  for subproject in "${subprojects[@]}"; do
    if [[ ${subproject} == "cccl" ]]; then
      # Special case handled above.
      continue
    fi

    # project_dirs[$subproject] may be a list of paths, or fallback to $subproject
    if [[ -n "${project_dirs[$subproject]:-}" ]]; then
      eval "dirs=${project_dirs[$subproject]}"
    else
      dirs=("$subproject")
    fi
    dirty=0
    for dir in "${dirs[@]}"; do
      if subdir_is_dirty "$dir"; then
        dirty=1
        break
      fi
    done

    declare ${subproject^^}_DIRTY=${dirty}
    checkmark="$(get_checkmark ${dirty})"

    echo "| ${checkmark} | ${project_names[$subproject]}" | tee_to_step_summary
  done
  echo | tee_to_step_summary

  echo -e "### Modifications in project or dependencies?\n" | tee_to_step_summary
  echo "|     | Project" | tee_to_step_summary
  echo "|-----|---------" | tee_to_step_summary

  for subproject in "${subprojects[@]}"; do
    dirty=0
    if project_or_deps_are_dirty "${subproject}"; then
      dirty=1
    fi
    declare ${subproject^^}_OR_DEPS_DIRTY=${dirty}
    checkmark="$(get_checkmark ${dirty})"
    echo "| ${checkmark} | ${project_names[$subproject]}" | tee_to_step_summary
  done

  echo "</details>" | tee_to_step_summary

  declare -a dirty_subprojects=()
  for subproject in "${subprojects[@]}"; do
    var_name="${subproject^^}_OR_DEPS_DIRTY"
    if [[ ${!var_name} -ne 0 ]]; then
      dirty_subprojects+=("$subproject")
    fi
  done

  write_output "DIRTY_PROJECTS" "${dirty_subprojects[*]}"
}

main "$@"
