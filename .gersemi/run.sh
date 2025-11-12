#!/bin/bash

# Run gersemi on all CMake files, limited to tracked files in git.

set -euo pipefail

if ! command -v gersemi &> /dev/null; then
  echo "gersemi could not be found. Please install gersemi to use this script:"
  echo "  pip install gersemi"
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Patterns for git ls-files:
declare -a patterns=(
  "*.cmake"
  "CMakeLists.txt"
  "**/CMakeLists.txt"
)

declare -a tracked_files
while IFS= read -r file; do
  tracked_files+=("$file")
done < <(git ls-files "${patterns[@]}")

cd "${repo_root}"
gersemi -i -- "${tracked_files[@]}"

# Count the total number of lines in the files:
total_lines=0
for file in "${tracked_files[@]}"; do
  if [[ -f "$file" ]]; then
    line_count=$(wc -l < "$file")
    total_lines=$((total_lines + line_count))
  fi
done
echo "Formatted ${#tracked_files[@]} files with a total of ${total_lines} lines."
