# bandit — Python security scanning.
#
# Detects common security issues in Python code:
# hardcoded passwords, SQL injection, shell injection,
# insecure hash functions, debug flags, etc.
#
{ pkgs, lib, src, analysisLib }:

let
  runner = pkgs.writeShellApplication {
    name = "run-bandit";
    runtimeInputs = with pkgs; [ bandit coreutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== bandit Analysis ==="

      bandit \
        -r "$source_dir" \
        -f json \
        --severity-level all \
        --confidence-level all \
        --exclude='build,pkg,makefiles,.git' \
        -x '*.pyc' \
        > "$output_dir/report.json" 2>"$output_dir/bandit-stderr.log" || true

      bandit \
        -r "$source_dir" \
        -f txt \
        --severity-level all \
        --confidence-level all \
        --exclude='build,pkg,makefiles,.git' \
        -x '*.pyc' \
        > "$output_dir/report.txt" 2>"$output_dir/bandit-stderr.log" || true

      findings=$(grep -c '"issue_severity"' "$output_dir/report.json" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "bandit: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "bandit" runner
