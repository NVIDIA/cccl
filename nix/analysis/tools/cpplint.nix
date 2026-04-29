# cpplint — Google C++ style checking.
#
# Scans raw source files (no compilation database needed).
# Catches style issues, missing includes, and common C++ pitfalls.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-cpplint";
    runtimeInputs = with pkgs; [ cpplint coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== cpplint Analysis ==="

      # Find all C/C++ source and header files
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
                   -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
                   -o -name '*.cu' -o -name '*.cuh' \) \
        ${excludes} \
        -print0 | xargs -0 \
      cpplint \
        --filter=-legal/copyright,-build/include_subdir,-build/header_guard,-whitespace,-readability/todo \
        --quiet \
        --counting=detailed \
        > "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c ': ' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "cpplint: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "cpplint" runner
