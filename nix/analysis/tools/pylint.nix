# pylint — comprehensive Python linting.
#
# Catches a wide range of issues: errors, conventions, refactoring
# suggestions, and code quality metrics.
#
{ pkgs, lib, src, analysisLib }:

let
  runner = pkgs.writeShellApplication {
    name = "run-pylint";
    runtimeInputs = with pkgs; [ pylint coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== pylint Analysis ==="

      # Find Python files
      find "$source_dir" \
        -type f -name '*.py' \
        -not -path '*/build/*' \
        -not -path '*/.git/*' \
        -not -path '*/pkg/*' \
        -not -path '*/makefiles/*' \
        > "$output_dir/file-list.txt" 2>/dev/null || true

      file_count=$(wc -l < "$output_dir/file-list.txt" || echo "0")
      echo "Linting $file_count Python files..."

      # Disable import-error since many deps are unavailable in sandbox.
      xargs < "$output_dir/file-list.txt" \
        pylint \
        --disable=import-error,no-name-in-module,wrong-import-order \
        --enable=useless-else-on-loop,simplifiable-if-statement,consider-using-enumerate,consider-using-dict-items,consider-using-f-string,use-implicit-booleaness-not-comparison \
        --output-format=text \
        --score=no \
        > "$output_dir/report.txt" 2>&1 || true

      # JSON for triage
      xargs < "$output_dir/file-list.txt" \
        pylint \
        --disable=import-error,no-name-in-module,wrong-import-order \
        --enable=useless-else-on-loop,simplifiable-if-statement,consider-using-enumerate,consider-using-dict-items,consider-using-f-string,use-implicit-booleaness-not-comparison \
        --output-format=json \
        --score=no \
        > "$output_dir/report.json" 2>"$output_dir/pylint-stderr.log" || true

      findings=$(grep -c '^[^ ].*:' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "pylint: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "pylint" runner
