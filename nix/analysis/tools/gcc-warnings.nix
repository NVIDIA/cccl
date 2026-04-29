# gcc-warnings — extended compiler warning flags.
#
# Compiles C/C++ source files with aggressive -W flags to catch issues
# that the default build misses. Since CCCL requires CUDA for a full build,
# this tool uses per-file g++ -fsyntax-only on .cc/.cpp/.c files.
#
{ pkgs, lib, src, analysisLib }:

let
  warningFlags = lib.concatStringsSep " " analysisLib.gccWarningFlags;
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-gcc-warnings";
    runtimeInputs = with pkgs; [ gcc coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== GCC Extended Warnings Analysis ==="

      # CCCL requires CUDA for full cmake build, so compile individual
      # .cc/.cpp/.c files with -fsyntax-only and extended warning flags.
      # Skip .cu files (need nvcc).
      find "$source_dir" \
        -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' \) \
        ${excludes} \
        > "$output_dir/file-list.txt" 2>/dev/null || true

      file_count=$(wc -l < "$output_dir/file-list.txt" || echo "0")
      echo "Checking $file_count files with extended warnings..."

      touch "$output_dir/report.txt"

      while IFS= read -r file; do
        # Determine language flag
        case "$file" in
          *.c)  lang_flag="-x c -std=c11" ;;
          *)    lang_flag="-x c++ -std=c++17" ;;
        esac

        # shellcheck disable=SC2086
        g++ -fsyntax-only \
          $lang_flag \
          ${warningFlags} \
          -I"$source_dir/libcudacxx/include" \
          -I"$source_dir/cub" \
          -I"$source_dir/thrust" \
          -I"$source_dir/cudax/include" \
          -I"$source_dir/c/include" \
          "$file" 2>&1 | \
          grep -E ': warning:|: error:' >> "$output_dir/report.txt" || true
      done < "$output_dir/file-list.txt"

      findings=$(grep -c ': warning:\|: error:' "$output_dir/report.txt" 2>/dev/null || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "gcc-warnings: $findings findings"
    '';
  };
in
analysisLib.mkBuildReport "gcc-warnings" runner
