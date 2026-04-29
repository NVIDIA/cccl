# clang-analyzer — Clang Static Analyzer (scan-build equivalent).
#
# Performs deep path-sensitive analysis for null pointer dereferences,
# memory leaks, use-after-free, and other interprocedural issues.
# Requires compilation database.
#
{ pkgs, lib, src, analysisLib, compileDb }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-clang-analyzer";
    runtimeInputs = with pkgs; [ clang-tools coreutils findutils gnugrep ];
    text = ''
      compile_db="$1"
      source_dir="$2"
      output_dir="$3"

      echo "=== Clang Static Analyzer ==="

      mkdir -p "$output_dir/plist"

      # Find source files and run clang --analyze on each
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) \
        ${excludes} \
        -print0 | xargs -0 -P "$(nproc)" -I{} \
      clang --analyze \
        -Xanalyzer -analyzer-output=text \
        -Xanalyzer -analyzer-checker=core,deadcode,security,unix,cplusplus,optin,alpha.core,alpha.deadcode,alpha.security,alpha.cplusplus,alpha.unix \
        -Xanalyzer -analyzer-config -Xanalyzer aggressive-binary-operation-simplification=true \
        -Xanalyzer -analyzer-config -Xanalyzer consider-single-element-arrays-as-flexible-array-members=true \
        -Xanalyzer -analyzer-config -Xanalyzer widen-loops=true \
        -p "$compile_db" \
        {} \
        >> "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c ': warning:' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "clang-analyzer: $findings findings"
    '';
  };
in
analysisLib.mkCompileDbReport "clang-analyzer" runner compileDb
