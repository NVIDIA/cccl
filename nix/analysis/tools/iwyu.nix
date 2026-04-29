# include-what-you-use — header hygiene analysis.
#
# Detects missing includes (code uses symbols not included)
# and excess includes (headers included but not used).
# Requires compilation database.
#
{ pkgs, lib, src, analysisLib, compileDb }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-iwyu";
    runtimeInputs = with pkgs; [ include-what-you-use coreutils findutils gnugrep ];
    text = ''
      compile_db="$1"
      source_dir="$2"
      output_dir="$3"

      echo "=== include-what-you-use Analysis ==="

      # Find source files
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) \
        ${excludes} \
        -print0 | xargs -0 -P "$(nproc)" -I{} \
      iwyu_tool.py -p "$compile_db" -- {} \
        >> "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c 'should add\|should remove' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "iwyu: $findings findings"
    '';
  };
in
analysisLib.mkCompileDbReport "iwyu" runner compileDb
