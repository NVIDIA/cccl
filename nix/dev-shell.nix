# Development shell with all static analysis tools on PATH.
#
# Enter with: nix develop
#
{ pkgs }:

pkgs.mkShell {
  name = "cccl-analysis";

  packages = with pkgs; [
    # C/C++ analysis
    clang-tools       # clang-tidy, clang-format, clang --analyze
    cppcheck
    flawfinder
    cpplint
    include-what-you-use
    coccinelle

    # Semgrep
    semgrep

    # Python analysis
    ruff
    bandit
    pylint

    # General
    shellcheck
    yamllint
    gersemi           # CMake formatting/linting

    # Build tools (for compile_commands.json generation)
    cmake
    gcc
    gnumake

    # Utilities
    python3
    jq                # JSON processing
    parallel          # Parallel execution
  ];

  shellHook = ''
    echo ""
    echo "=== CCCL Static Analysis Shell ==="
    echo ""
    echo "Available analysis commands:"
    echo "  nix build .#analysis-quick       Quick checks (~5-15 min)"
    echo "  nix build .#analysis-standard    Standard checks (~30-60 min)"
    echo "  nix build .#analysis-deep        Full analysis (~1-3 hours)"
    echo ""
    echo "Individual tools:"
    echo "  nix build .#analysis-flawfinder"
    echo "  nix build .#analysis-cpplint"
    echo "  nix build .#analysis-cppcheck"
    echo "  nix build .#analysis-clang-tidy"
    echo "  nix build .#analysis-ruff"
    echo "  nix build .#analysis-semgrep-cpp"
    echo "  ... (see flake.nix for all targets)"
    echo ""

    # Disable semgrep telemetry
    export SEMGREP_ENABLE_VERSION_CHECK=0
    export SEMGREP_SEND_METRICS=off
  '';
}
