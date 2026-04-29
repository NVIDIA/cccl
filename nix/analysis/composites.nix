# Composite analysis targets: quick, standard, deep.
#
# Each tier includes progressively more tools, trading speed for coverage.
#
{ pkgs, lib, tools }:

let
  # All tool names — used by deep and all targets
  allTools = lib.attrNames tools;

  # Helper to create a composite target that links to tool outputs
  # and prints a summary.
  mkComposite = name: description: toolSet:
    pkgs.runCommand "analysis-${name}" { } (
      let
        linkCommands = lib.concatMapStringsSep "\n" (t:
          "ln -s ${tools.${t}} $out/${t}"
        ) toolSet;

        summaryLines = lib.concatMapStringsSep "\n" (t:
          ''echo "${t}: $(cat ${tools.${t}}/count.txt 2>/dev/null || echo 'N/A') findings"''
        ) toolSet;
      in
      ''
        mkdir -p $out

        ${linkCommands}

        {
          echo "=== Analysis Summary (${name}) ==="
          echo "${description}"
          echo ""
          ${summaryLines}
          echo ""
          echo "Reports in: $out/<tool>/report.txt"
        } > $out/summary.txt

        cat $out/summary.txt
      ''
    );

in
{
  # Quick: source-only tools, no compilation needed (~2-5 min)
  quick = mkComposite "quick"
    "Fast checks for pre-commit and immediate feedback."
    [
      "flawfinder"
      "cpplint"
      "ruff"
      "shellcheck"
      "yamllint"
      "cmake-lint"
    ];

  # Standard: quick + compile-db tools (~10-20 min)
  standard = mkComposite "standard"
    "Comprehensive analysis balanced with runtime."
    [
      # Quick tools
      "flawfinder"
      "cpplint"
      "ruff"
      "shellcheck"
      "yamllint"
      "cmake-lint"
      # Compile-db tools
      "clang-tidy"
      "cppcheck"
      # Python tools
      "bandit"
      "pylint"
    ];

  # Deep: all tools (~30-60 min)
  deep = mkComposite "deep"
    "Exhaustive analysis including interprocedural and pattern-based tools."
    allTools;

  # All: explicit alias for deep — every tool, no exceptions
  all = mkComposite "all"
    "Every analysis tool (${toString (lib.length allTools)} tools). Same as deep."
    allTools;
}
