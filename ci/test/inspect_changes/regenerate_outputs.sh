#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../../.." && pwd)
inspect_changes="${project_root}/ci/inspect_changes.py"

if [[ ! -x "${inspect_changes}" ]]; then
  echo "Error: ${inspect_changes} not found or not executable." >&2
  exit 1
fi

cd "${script_dir}"

for dirty_file in *.dirty_files; do
  test_name=${dirty_file%.dirty_files}
  output_file="${test_name}.output"
  echo "Regenerating ${output_file}"
  python "${inspect_changes}" --file "${dirty_file}" \
    | awk '/^FULL_BUILD=/{print}/^LITE_BUILD=/{print}' > "${output_file}"
done
