# clang-tidy — linting and modernization with compilation database.
#
# Uses CCCL's existing .clang-tidy config at the repo root.
# Handles .cu files as C++ with appropriate language flags.
#
{ pkgs, lib, src, analysisLib, compileDb }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-clang-tidy";
    runtimeInputs = with pkgs; [ clang-tools coreutils findutils gnugrep parallel ];
    text = ''
      compile_db="$1"
      source_dir="$2"
      output_dir="$3"

      echo "=== clang-tidy Analysis ==="
      echo "Using compilation database: $compile_db"

      # Find all C/C++/CUDA source files (not headers)
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
                   -o -name '*.cu' \) \
        ${excludes} \
        > "$output_dir/file-list.txt" 2>/dev/null || true

      file_count=$(wc -l < "$output_dir/file-list.txt" || echo "0")
      echo "Scanning $file_count source files..."

      # Use CCCL's .clang-tidy config (always present at repo root)
      effective_config="$source_dir/.clang-tidy"
      echo "Using project's .clang-tidy config"

      # Run clang-tidy with the compilation database
      parallel -j "$(nproc)" \
        clang-tidy -p "$compile_db" \
          --config-file="$effective_config" \
          --quiet {} \
        < "$output_dir/file-list.txt" \
        > "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c ': warning:\|: error:' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "clang-tidy: $findings findings"
    '';
  };
in
analysisLib.mkCompileDbReport "clang-tidy" runner compileDb
