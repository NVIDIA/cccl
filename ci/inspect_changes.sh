#!/bin/bash

# Github action script to identify which subprojects are dirty in a PR

set -u

# Usage: inspect_changes.sh <base_sha> <head_sha>
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base_sha> <head_sha>"
  exit 1
fi

base_sha=$1
head_sha=$2

# Github gives the SHA as the current HEAD of the target ref, not the common ancestor.
# Find the common ancestor and use that for the base.
git fetch origin --unshallow -q
git fetch origin $base_sha -q
base_sha=$(git merge-base $head_sha $base_sha)

# Define a list of subproject directories:
subprojects=(
  libcudacxx
  cub
  thrust
)

# ...and their dependencies:
declare -A dependencies=(
  [libcudacxx]="cccl"
  [cub]="cccl libcudacxx thrust"
  [thrust]="cccl libcudacxx cub"
)

write_output() {
  local key="$1"
  local value="$2"
  echo "$key=$value" | tee --append "${GITHUB_OUTPUT:-/dev/null}"
}

dirty_files() {
  git diff --name-only "${base_sha}" "${head_sha}"
}

# Return 1 if any files outside of the subproject directories have changed
inspect_cccl() {
  subprojs_grep_expr=$(
    IFS="|"
    echo "(${subprojects[*]})/"
  )

  if dirty_files | grep -v -E "${subprojs_grep_expr}" | grep -q "."; then
    return 1
  else
    return 0
  fi
}

# inspect_subdir <subdir>
# Returns 1 if any files in the subdirectory have changed
inspect_subdir() {
  local subdir="$1"

  if dirty_files | grep -E "^${subdir}/" | grep -q '.'; then
    return 1
  else
    return 0
  fi
}

# add_dependencies <subproject>
# if the subproject or any of its dependencies are dirty, return 1
add_dependencies() {
  local subproject="$1"

  # Check if ${subproject^^}_DIRTY is set to 1, return 1 if it is.
  local dirty_flag=${subproject^^}_DIRTY
  if [[ ${!dirty_flag} -ne 0 ]]; then
    return 1
  fi

  for dependency in ${dependencies[$subproject]}; do
    dirty_flag="${dependency^^}_DIRTY"
    if [[ ${!dirty_flag} -ne 0 ]]; then
      return 1
    fi
  done

  return 0
}

# write_subproject_status <subproject>
# Write the output <subproject_uppercase>_DIRTY={true|false}
write_subproject_status() {
  local subproject="$1"
  local dirty_flag=${subproject^^}_DIRTY

  if [[ ${!dirty_flag} -ne 0 ]]; then
    write_output "${dirty_flag}" "true"
  else
    write_output "${dirty_flag}" "false"
  fi
}

main() {
  # Print the list of subprojects and all of their dependencies:
  echo "Subprojects: ${subprojects[*]}"
  echo
  echo "Dependencies:"
  for subproject in "${subprojects[@]}"; do
    echo "  - ${subproject} -> ${dependencies[$subproject]}"
  done
  echo

  echo "Base SHA: ${base_sha}"
  echo "HEAD SHA: ${head_sha}"
  echo

  # Print the list of files that have changed:
  echo "Dirty files:"
  dirty_files | sed 's/^/  - /'
  echo ""

  echo "Modifications in project?"
  # Assign the return value of `inspect_cccl` to the variable `CCCL_DIRTY`:
  inspect_cccl
  CCCL_DIRTY=$?
  echo "$(if [[ ${CCCL_DIRTY} -eq 0 ]]; then echo " "; else echo "X"; fi) - CCCL Infrastructure"

  # Check for changes in each subprojects directory:
  for subproject in "${subprojects[@]}"; do
    inspect_subdir $subproject
    declare ${subproject^^}_DIRTY=$?
    echo "$(if [[ ${subproject^^}_DIRTY -eq 0 ]]; then echo " "; else echo "X"; fi) - ${subproject}"
  done
  echo

  echo "Modifications in project or dependencies?"
  for subproject in "${subprojects[@]}"; do
    add_dependencies ${subproject}
    declare ${subproject^^}_DIRTY=$?
    echo "$(if [[ ${subproject^^}_DIRTY -eq 0 ]]; then echo " "; else echo "X"; fi) - ${subproject}"
  done
  echo

  for subproject in "${subprojects[@]}"; do
    write_subproject_status ${subproject}
  done
}

main "$@"
