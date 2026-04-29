# cppcheck — deep static analysis for C/C++.
#
# Produces both XML and text reports.
# Enables all checks with sensible suppressions for known noisy patterns.
#
{ pkgs, lib, src, analysisLib, compileDb }:

let
  runner = pkgs.writeShellApplication {
    name = "run-cppcheck";
    runtimeInputs = with pkgs; [ cppcheck coreutils gnugrep ];
    text = ''
      compile_db="$1"
      # shellcheck disable=SC2034
      source_dir="$2"
      output_dir="$3"

      echo "=== cppcheck Analysis ==="

      # XML report for machine parsing
      cppcheck \
        --project="$compile_db/compile_commands.json" \
        --enable=all \
        --std=c++17 \
        --inconclusive \
        --force \
        --max-ctu-depth=8 \
        --suppress=missingInclude \
        --suppress=unusedFunction \
        --suppress=unmatchedSuppression \
        --suppress=missingIncludeSystem \
        --inline-suppr \
        --xml \
        2> "$output_dir/report.xml" || true

      # Human-readable text report
      cppcheck \
        --project="$compile_db/compile_commands.json" \
        --enable=all \
        --std=c++17 \
        --inconclusive \
        --force \
        --max-ctu-depth=8 \
        --suppress=missingInclude \
        --suppress=unusedFunction \
        --suppress=unmatchedSuppression \
        --suppress=missingIncludeSystem \
        --inline-suppr \
        2> "$output_dir/report.txt" || true

      findings=$(grep -c '\(error\|warning\|style\|performance\|portability\)' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "cppcheck: $findings findings"
    '';
  };
in
analysisLib.mkCompileDbReport "cppcheck" runner compileDb
