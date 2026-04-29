# semgrep — pattern-based static analysis for Python.
#
# Uses vendored rules for security-focused Python analysis.
# Detects: command injection, SQL injection, path traversal,
# insecure deserialization, hardcoded secrets, etc.
#
{ pkgs, lib, src, analysisLib }:

let
  rulesFile = ../rules/semgrep-python.yaml;

  runner = pkgs.writeShellApplication {
    name = "run-semgrep-python";
    runtimeInputs = with pkgs; [ semgrep coreutils gnugrep cacert ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== semgrep (Python) Analysis ==="

      export SEMGREP_ENABLE_VERSION_CHECK=0
      export SEMGREP_SEND_METRICS=off
      export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      export OTEL_TRACES_EXPORTER=none
      HOME="$(mktemp -d)"
      export HOME

      semgrep \
        --config "${rulesFile}" \
        --json \
        --no-git-ignore \
        --exclude='build' \
        --exclude='_deps' \
        --exclude='.devcontainer' \
        --exclude='lib' \
        --include='*.py' \
        "$source_dir" \
        > "$output_dir/report.json" 2>"$output_dir/semgrep-stderr.log" || true

      semgrep \
        --config "${rulesFile}" \
        --no-git-ignore \
        --exclude='build' \
        --exclude='_deps' \
        --exclude='.devcontainer' \
        --exclude='lib' \
        --include='*.py' \
        "$source_dir" \
        > "$output_dir/report.txt" 2>>"$output_dir/semgrep-stderr.log" || true

      findings=$(grep -c '"check_id"' "$output_dir/report.json" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "semgrep (Python): $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "semgrep-python" runner
