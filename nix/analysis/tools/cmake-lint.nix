# cmake-lint — CMake file linting via gersemi.
#
# Checks CMakeLists.txt and .cmake files for formatting issues.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-cmake-lint";
    runtimeInputs = with pkgs; [ gersemi coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== CMake Lint (gersemi) Analysis ==="

      HOME="$(mktemp -d)"
      export HOME

      # Find all CMake files
      find "$source_dir" \
        -type f \( -name 'CMakeLists.txt' -o -name '*.cmake' \) \
        ${excludes} \
        > "$output_dir/file-list.txt" 2>/dev/null || true

      file_count=$(wc -l < "$output_dir/file-list.txt" || echo "0")
      echo "Checking $file_count CMake files..."

      # gersemi --check returns non-zero if files need formatting
      xargs < "$output_dir/file-list.txt" \
        gersemi --check \
        > "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c 'would be reformatted\|Error' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "cmake-lint: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "cmake-lint" runner
