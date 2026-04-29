# ruff — fast Python linting and formatting checks.
#
# Covers: pyflakes, pycodestyle, isort, pep8-naming, flake8-bugbear,
# flake8-comprehensions, and many more rule sets.
#
{ pkgs, lib, src, analysisLib }:

let
  runner = pkgs.writeShellApplication {
    name = "run-ruff";
    runtimeInputs = with pkgs; [ ruff coreutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== ruff Analysis ==="

      # Run ruff with a broad selection of rules
      # --no-cache because nix store is read-only
      ruff check \
        --no-cache \
        --select=ALL \
        --ignore=ANN401,E501,COM812,ISC001,D100,D101,D102,D103,D104,D105,D107 \
        --output-format=concise \
        --exclude='build,_deps,.devcontainer,lib,wheelhouse' \
        "$source_dir" \
        > "$output_dir/report.txt" 2>&1 || true

      # JSON output for triage
      ruff check \
        --no-cache \
        --select=ALL \
        --ignore=ANN401,E501,COM812,ISC001,D100,D101,D102,D103,D104,D105,D107 \
        --output-format=json \
        --exclude='build,_deps,.devcontainer,lib,wheelhouse' \
        "$source_dir" \
        > "$output_dir/report.json" 2>&1 || true

      findings=$(wc -l < "$output_dir/report.txt" 2>/dev/null || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "ruff: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "ruff" runner
