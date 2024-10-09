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

# Define a list of subproject directories by their subdirectory name:
subprojects=(
  cccl
  libcudacxx
  cub
  thrust
  cudax
  pycuda
  cccl_c_parallel
)

# ...and their dependencies:
declare -A dependencies=(
  [cccl]=""
  [libcudacxx]="cccl"
  [cub]="cccl libcudacxx thrust"
  [thrust]="cccl libcudacxx cub"
  [cudax]="cccl libcudacxx"
  [pycuda]="cccl libcudacxx cub thrust cccl_c_parallel"
  [cccl_c_parallel]="cccl libcudacxx cub thrust"
)

declare -A project_names=(
  [cccl]="CCCL Infrastructure"
  [libcudacxx]="libcu++"
  [cub]="CUB"
  [thrust]="Thrust"
  [cudax]="CUDA Experimental"
  [pycuda]="pycuda"
  [cccl_c_parallel]="CCCL C Parallel Library"
)

# By default, the project directory is assumed to be the same as the subproject name,
# but can be overridden here. The `cccl` project is special, and checks for files outside
# of any subproject directory.
declare -A project_dirs=(
  [pycuda]="python/cuda_cooperative"
  [cccl_c_parallel]="c/parallel"
)

# Usage checks:
for subproject in "${subprojects[@]}"; do
  # Check that the subproject directory exists
  if [ "$subproject" != "cccl" ]; then
    subproject_dir=${project_dirs[$subproject]:-$subproject}
    if [ ! -d "$subproject_dir" ]; then
      echo "Error: Subproject '$subproject' directory '$subproject_dir' does not exist."
      exit 1
    fi
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

# Return 1 if any files outside of the subproject directories have changed
inspect_cccl() {
  exclusions_grep_expr=$(
    declare -a exclusions
    for subproject in "${subprojects[@]}"; do
      if [[ ${subproject} == "cccl" ]]; then
        continue
      fi
      exclusions+=("${project_dirs[$subproject]:-$subproject}")
    done

    # Manual exclusions:
    exclusions+=("docs")

    IFS="|"
    echo "^(${exclusions[*]})/"
  )

  if dirty_files | grep -v -E "${exclusions_grep_expr}" | grep -q "."; then
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

  # Assign the return value of `inspect_cccl` to the variable `CCCL_DIRTY`:
  inspect_cccl
  CCCL_DIRTY=$?
  checkmark="$(get_checkmark ${CCCL_DIRTY})"
  echo "| ${checkmark} | ${project_names[cccl]}" | tee_to_step_summary

  # Check for changes in each subprojects directory:
  for subproject in "${subprojects[@]}"; do
    if [[ ${subproject} == "cccl" ]]; then
      # Special case handled above.
      continue
    fi

    inspect_subdir ${project_dirs[$subproject]:-$subproject}
    local dirty=$?
    declare ${subproject^^}_DIRTY=${dirty}
    checkmark="$(get_checkmark ${dirty})"
    echo "| ${checkmark} | ${project_names[$subproject]}" | tee_to_step_summary
  done
  echo | tee_to_step_summary

  echo -e "### Modifications in project or dependencies?\n" | tee_to_step_summary
  echo "|     | Project" | tee_to_step_summary
  echo "|-----|---------" | tee_to_step_summary

  for subproject in "${subprojects[@]}"; do
    add_dependencies ${subproject}
    local dirty=$?
    declare ${subproject^^}_DIRTY=${dirty}
    checkmark="$(get_checkmark ${dirty})"
    echo "| ${checkmark} | ${project_names[$subproject]}" | tee_to_step_summary
  done

  echo "</details>" | tee_to_step_summary

  declare -a dirty_subprojects=()
  for subproject in "${subprojects[@]}"; do
    var_name="${subproject^^}_DIRTY"
    if [[ ${!var_name} -ne 0 ]]; then
      dirty_subprojects+=("$subproject")
    fi
  done

  write_output "DIRTY_PROJECTS" "${dirty_subprojects[*]}"
}

main "$@"
