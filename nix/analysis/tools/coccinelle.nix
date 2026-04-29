# coccinelle — semantic patch-based pattern matching for C/C++.
#
# Uses vendored .cocci files for detecting common bug patterns:
# null dereferences, use-after-free, double-free, resource leaks.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;
  cocciRulesDir = ../rules/coccinelle;

  runner = pkgs.writeShellApplication {
    name = "run-coccinelle";
    runtimeInputs = with pkgs; [ coccinelle coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== Coccinelle Analysis ==="

      total_findings=0

      # Run each .cocci rule against all C/C++ source files
      for cocci_file in ${cocciRulesDir}/*.cocci; do
        rule_name=$(basename "$cocci_file" .cocci)
        echo "Running rule: $rule_name"

        find "$source_dir" \
          -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) \
          ${excludes} \
          -print0 | xargs -0 -P "$(nproc)" -I{} \
        spatch --sp-file "$cocci_file" --very-quiet {} \
          >> "$output_dir/report-$rule_name.txt" 2>/dev/null || true

        count=$(grep -c '.' "$output_dir/report-$rule_name.txt" 2>/dev/null || echo "0")
        total_findings=$((total_findings + count))
      done

      # Combine all rule reports
      cat "$output_dir"/report-*.txt > "$output_dir/report.txt" 2>/dev/null || true
      echo "$total_findings" > "$output_dir/count.txt"

      echo "coccinelle: $total_findings findings"
    '';
  };
in
analysisLib.mkSourceReport "coccinelle" runner
