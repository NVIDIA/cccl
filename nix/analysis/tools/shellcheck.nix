# shellcheck — shell script linting.
#
# Detects common bash/sh scripting pitfalls:
# unquoted variables, useless cat, unreachable code,
# incorrect test operators, etc.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-shellcheck";
    runtimeInputs = with pkgs; [ shellcheck coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== ShellCheck Analysis ==="

      find "$source_dir" \
        -type f -name '*.sh' \
        ${excludes} \
        -print0 | xargs -0 \
      shellcheck \
        --format=gcc \
        --severity=info \
        --shell=bash \
        > "$output_dir/report.txt" 2>&1 || true

      # JSON for triage
      find "$source_dir" \
        -type f -name '*.sh' \
        ${excludes} \
        -print0 | xargs -0 \
      shellcheck \
        --format=json \
        --severity=info \
        --shell=bash \
        > "$output_dir/report.json" 2>&1 || true

      findings=$(grep -c ': warning:\|: error:\|: note:' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "shellcheck: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "shellcheck" runner
