# semgrep — pattern-based static analysis for C/C++/CUDA.
#
# Uses vendored rules (no network access needed in nix sandbox).
# Rules cover: unsafe functions, memory management, race conditions,
# type safety, error handling, resource management, concurrency.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  rulesFile = ../rules/semgrep-cpp.yaml;
  cudaRulesFile = ../rules/semgrep-cuda.yaml;

  runner = pkgs.writeShellApplication {
    name = "run-semgrep-cpp";
    runtimeInputs = with pkgs; [ semgrep coreutils gnugrep cacert ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== semgrep (C/C++) Analysis ==="

      export SEMGREP_ENABLE_VERSION_CHECK=0
      export SEMGREP_SEND_METRICS=off
      export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      export OTEL_TRACES_EXPORTER=none
      HOME="$(mktemp -d)"
      export HOME

      # Run with vendored C/C++ and CUDA rules
      semgrep \
        --config "${rulesFile}" \
        --config "${cudaRulesFile}" \
        --json \
        --no-git-ignore \
        --exclude='build' \
        --exclude='_deps' \
        --exclude='.devcontainer' \
        --exclude='lib' \
        --include='*.c' --include='*.cc' --include='*.cpp' --include='*.cxx' \
        --include='*.h' --include='*.hh' --include='*.hpp' --include='*.hxx' \
        --include='*.cu' --include='*.cuh' \
        "$source_dir" \
        > "$output_dir/report.json" 2>"$output_dir/semgrep-stderr.log" || true

      # Also produce human-readable report
      semgrep \
        --config "${rulesFile}" \
        --config "${cudaRulesFile}" \
        --no-git-ignore \
        --exclude='build' \
        --exclude='_deps' \
        --exclude='.devcontainer' \
        --exclude='lib' \
        --include='*.c' --include='*.cc' --include='*.cpp' --include='*.cxx' \
        --include='*.h' --include='*.hh' --include='*.hpp' --include='*.hxx' \
        --include='*.cu' --include='*.cuh' \
        "$source_dir" \
        > "$output_dir/report.txt" 2>>"$output_dir/semgrep-stderr.log" || true

      findings=$(grep -c '"check_id"' "$output_dir/report.json" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "semgrep (C/C++): $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "semgrep-cpp" runner
