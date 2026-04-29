# Top-level analysis module.
#
# Imports all tool modules, compile-db generation,
# and composite targets. Exports a flat attrset of all analysis derivations.
#
{ pkgs, lib, src }:

let
  # Shared helpers
  analysisLib = import ./lib.nix { inherit pkgs lib src; };

  # Compile database generation
  compileDb = import ./compile-db.nix { inherit pkgs lib src; };

  # -- Individual tool targets --

  # C/C++ source-only tools
  flawfinder = import ./tools/flawfinder.nix { inherit pkgs lib src analysisLib; };
  cpplint = import ./tools/cpplint.nix { inherit pkgs lib src analysisLib; };
  semgrep-cpp = import ./tools/semgrep-cpp.nix { inherit pkgs lib src analysisLib; };
  coccinelle = import ./tools/coccinelle.nix { inherit pkgs lib src analysisLib; };

  # C/C++ compile-db tools
  clang-tidy = import ./tools/clang-tidy.nix { inherit pkgs lib src analysisLib compileDb; };
  cppcheck = import ./tools/cppcheck.nix { inherit pkgs lib src analysisLib compileDb; };
  iwyu = import ./tools/iwyu.nix { inherit pkgs lib src analysisLib compileDb; };
  clang-analyzer = import ./tools/clang-analyzer.nix { inherit pkgs lib src analysisLib compileDb; };

  # C/C++ build tools
  gcc-warnings = import ./tools/gcc-warnings.nix { inherit pkgs lib src analysisLib; };
  gcc-analyzer = import ./tools/gcc-analyzer.nix { inherit pkgs lib src analysisLib; };

  # Python tools
  ruff = import ./tools/ruff.nix { inherit pkgs lib src analysisLib; };
  bandit = import ./tools/bandit.nix { inherit pkgs lib src analysisLib; };
  pylint = import ./tools/pylint.nix { inherit pkgs lib src analysisLib; };
  semgrep-python = import ./tools/semgrep-python.nix { inherit pkgs lib src analysisLib; };

  # General tools
  shellcheck = import ./tools/shellcheck.nix { inherit pkgs lib src analysisLib; };
  yamllint = import ./tools/yamllint.nix { inherit pkgs lib src analysisLib; };
  cmake-lint = import ./tools/cmake-lint.nix { inherit pkgs lib src analysisLib; };

  # All individual tools as an attrset
  tools = {
    inherit
      flawfinder cpplint semgrep-cpp coccinelle
      clang-tidy cppcheck iwyu clang-analyzer
      gcc-warnings gcc-analyzer
      ruff bandit pylint semgrep-python
      shellcheck yamllint cmake-lint;
  };

  # -- Composite targets --

  composites = import ./composites.nix { inherit pkgs lib tools; };

in

# Export all targets with analysis- prefix
lib.mapAttrs' (name: value: {
  name = "analysis-${name}";
  inherit value;
}) tools

# Composite targets (already prefixed)
// {
  "analysis-quick" = composites.quick;
  "analysis-standard" = composites.standard;
  "analysis-deep" = composites.deep;
  "analysis-all" = composites.all;
}

# Compile database (useful for debugging)
// {
  "analysis-compile-db" = compileDb;
}
