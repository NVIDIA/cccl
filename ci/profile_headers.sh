#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_csv=""
run_ctadvisor=0
declare -A output_time_map=()
declare -A loc_by_pp=()
declare -A included_by_count_map=()

usage() {
  cat <<'EOF'
Usage: ci/profile_headers.sh --output-csv <path>
  --output-csv <path>        Required
  --ctadvisor                Print ctadvisor report
EOF
}

status() { echo "[profile-headers] $*" >&2; }

while (($#)); do
  case "$1" in
    --output-csv) output_csv="$2"; shift 2 ;;
    --ctadvisor) run_ctadvisor=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$output_csv" ]] || { echo "error: --output-csv is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
command -v cloc >/dev/null || { echo "error: cloc not found" >&2; exit 1; }
if (( run_ctadvisor )); then
  command -v ctadvisor >/dev/null || { echo "error: ctadvisor not found" >&2; exit 1; }
fi

pushd "$repo_root" >/dev/null

status "Configuring and building preset 'profile-headers'..."
cmake --preset "profile-headers" && cmake --build --preset "profile-headers"

shopt -s nullglob globstar
build_root="${repo_root}/build"
if [[ -n "${CCCL_BUILD_INFIX:-}" ]]; then
  build_root="${build_root}/${CCCL_BUILD_INFIX}"
fi
build_dir="${build_root}/profile-headers"
compdb_dir="${build_dir}"

ninja_log="${compdb_dir}/.ninja_log"
if [[ -f "${ninja_log}" ]]; then
  while IFS=$'\t' read -r output duration; do
    [[ -n "${output}" ]] || continue
    output_time_map["${output}"]="${duration}"
    if [[ "${output}" != /* ]]; then
      output_time_map["${compdb_dir}/${output}"]="${duration}"
    fi
  done < <(awk 'BEGIN{FS="\t"} $1 !~ /^#/ && NF >= 4 {print $4 "\t" ($2-$1)}' "${ninja_log}")
fi

declare -a selected_tus=()
declare -a selected_pps=()
status "Discovering preprocessed header TUs..."
pp_paths=("${build_dir}"/**/headers/**/*.cpp4.ii)
(( ${#pp_paths[@]} > 0 )) || { echo "error: no preprocessed header TUs found under ${build_dir}" >&2; exit 1; }
for pp in "${pp_paths[@]}"; do
  tu_abs="${pp%.cpp4.ii}.cu"
  selected_tus+=("${tu_abs}")
  selected_pps+=("${pp}")
done
status "Found ${#selected_tus[@]} header TUs."

status "Counting direct include references across CCCL headers..."
while IFS=$'\t' read -r header count; do
  [[ -n "${header}" ]] || continue
  included_by_count_map["${header}"]="${count}"
done < <(
  find \
    "${repo_root}/cub" \
    "${repo_root}/thrust" \
    "${repo_root}/libcudacxx/include" \
    "${repo_root}/cudax/include" \
    -type f \
    \( -name '*.h' -o -name '*.hpp' -o -name '*.cuh' -o ! -name '*.*' \) \
    -print0 \
    | xargs -0 grep -h -E '^[[:space:]]*#include[[:space:]]*[<"][^">]+[">]' \
    | sed -E 's/^[[:space:]]*#include[[:space:]]*[<"]([^">]+)[">].*$/\1/' \
    | sort \
    | uniq -c \
    | awk '{count=$1; $1=""; sub(/^ /, "", $0); print $0 "\t" count}'
)

cloc_tmp="$(mktemp)"
status "Counting transitive LOC with cloc..."
cloc --csv --by-file --skip-uniqueness --processes "$(nproc --all --ignore=2)" --force-lang=C++,ii "${selected_pps[@]}" > "${cloc_tmp}"
while IFS=',' read -r _lang file _blank _comment code; do
  [[ "${file}" == "filename" || -z "${file}" ]] && continue
  loc_by_pp["${file}"]="${code}"
done < "${cloc_tmp}"
rm -f "${cloc_tmp}"

mkdir -p "$(dirname "$output_csv")"
echo "header_path,compile_time_ms,transitive_loc,included_by_header_count" > "$output_csv"
status "Writing CSV rows..."

for i in "${!selected_tus[@]}"; do
  tu_abs="${selected_tus[$i]}"
  header_path="${tu_abs#*/headers/*/}"
  header_path="${header_path%.cu}"
  pp="${selected_pps[$i]}"

  transitive_loc="${loc_by_pp["${pp}"]:-0}"

  compile_time_ms=""
  tu_after_build="${tu_abs#${build_dir}/}"
  build_subdir="${tu_after_build%%/headers/*}"
  rel_after_headers="${tu_after_build#*/headers/}"
  target_name="${rel_after_headers%%/*}"
  rel_tu_path="${rel_after_headers#*/}"
  object_rel="${build_subdir}/CMakeFiles/${target_name}.dir/headers/${target_name}/${rel_tu_path}.o"
  compile_time_ms="${output_time_map["${object_rel}"]:-}"
  if [[ -z "${compile_time_ms}" ]]; then
    compile_time_ms="${output_time_map["${build_dir}/${object_rel}"]:-}"
  fi

  included_by_header_count="${included_by_count_map["${header_path}"]:-0}"
  echo "${header_path},${compile_time_ms},${transitive_loc},${included_by_header_count}" >> "$output_csv"
done

status "Done. Profile headers CSV: $output_csv"

if (( run_ctadvisor )); then
  ctadvisor_trace_dir="${build_dir}/header_testing/device_time_trace"
  json_traces=("${ctadvisor_trace_dir}"/**/*.json)
  (( ${#json_traces[@]} > 0 )) || { echo "error: no ctadvisor traces found under ${ctadvisor_trace_dir}" >&2; exit 1; }
  status "Running ctadvisor over ${#json_traces[@]} traces..."
  ctadvisor --trace-file-path "${ctadvisor_trace_dir}" --header-advisor-entries 20 --thread-number "$(nproc --all --ignore=2)"
fi

popd >/dev/null
