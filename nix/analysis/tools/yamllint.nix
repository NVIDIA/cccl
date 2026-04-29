# yamllint — YAML file linting.
#
# Checks for syntax errors, key duplication, line length,
# indentation consistency, and other YAML formatting issues.
#
{ pkgs, lib, src, analysisLib }:

let
  excludes = analysisLib.mkExcludeArgs analysisLib.defaultExcludes;

  runner = pkgs.writeShellApplication {
    name = "run-yamllint";
    runtimeInputs = with pkgs; [ yamllint coreutils findutils gnugrep ];
    text = ''
      source_dir="$1"
      output_dir="$2"

      echo "=== yamllint Analysis ==="

      find "$source_dir" \
        -type f \( -name '*.yaml' -o -name '*.yml' \) \
        ${excludes} \
        -print0 | xargs -0 \
      yamllint \
        -d '{extends: default, rules: {line-length: {max: 200}, truthy: disable}}' \
        -f parsable \
        > "$output_dir/report.txt" 2>&1 || true

      findings=$(grep -c ': \[error\]\|: \[warning\]' "$output_dir/report.txt" || echo "0")
      echo "$findings" > "$output_dir/count.txt"

      echo "yamllint: $findings findings"
    '';
  };
in
analysisLib.mkSourceReport "yamllint" runner
