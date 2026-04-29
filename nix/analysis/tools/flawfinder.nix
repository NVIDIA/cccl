# flawfinder — CWE-oriented security scanning for C/C++.
#
# Scans raw source files (no compilation database needed).
# Reports potential security vulnerabilities with CWE identifiers.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-flawfinder";
    runtimeInputs = with pkgs; [ flawfinder coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== flawfinder Analysis ==="

      # Scan all C/C++/CUDA source files
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
                   -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
                   -o -name '*.cu' -o -name '*.cuh' \) \
        ${excludes} \
        -print0 | xargs -0 \
      flawfinder \
        --minlevel=1 \
        --columns \
        --context \
        --singleline \
        --dataonly \
        --csv \
        > "$output_dir/report.csv" 2>&1 || true

      # Also produce human-readable report
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
                   -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
                   -o -name '*.cu' -o -name '*.cuh' \) \
        ${excludes} \
        -print0 | xargs -0 \
      flawfinder \
        --minlevel=1 \
        --columns \
        --context \
        --singleline \
        > "$output_dir/report.txt" 2>&1 || true

      # Extract hit count
      findings=$(grep -oP 'Hits = \K[0-9]+' "$output_dir/report.txt" | awk '{s+=$1} END {print s+0}')
      echo "$findings" > "$output_dir/count.txt"

      echo "flawfinder: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "flawfinder" runner
