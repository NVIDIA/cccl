# Shared helpers for building analysis tool derivations.
#
# Two patterns:
#   mkSourceReport  — tools that scan raw source (no compilation database)
#   mkCompileDbReport — tools that need compile_commands.json
#   mkBuildReport — tools that need a full build
#
{ pkgs, lib, src }:

{
  # Run a tool that operates on raw source files.
  # The runner script receives: <source-dir> <output-dir>
  mkSourceReport = name: runner:
    pkgs.runCommand "analysis-${name}" {
      nativeBuildInputs = [ runner ];
    } ''
      mkdir -p $out
      ${lib.getExe runner} ${src} $out
    '';

  # Run a tool that needs a compilation database.
  # The runner script receives: <compile-db-dir> <source-dir> <output-dir>
  mkCompileDbReport = name: runner: compileDb:
    pkgs.runCommand "analysis-${name}" {
      nativeBuildInputs = [ runner ];
    } ''
      mkdir -p $out
      ${lib.getExe runner} ${compileDb} ${src} $out
    '';

  # Run a tool that needs a full build (gcc-warnings, gcc-analyzer).
  # The runner script receives: <source-dir> <output-dir>
  mkBuildReport = name: runner:
    pkgs.runCommand "analysis-${name}" {
      nativeBuildInputs = [ runner ];
    } ''
      mkdir -p $out
      ${lib.getExe runner} ${src} $out
    '';

  # Shared GCC warning flags used by gcc-warnings and gcc-analyzer.
  gccWarningFlags = [
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wformat=2"
    "-Wformat-security"
    "-Wformat-overflow=2"
    "-Wformat-truncation=2"
    "-Wformat-signedness"
    "-Wshadow"
    "-Wcast-qual"
    "-Wcast-align=strict"
    "-Wwrite-strings"
    "-Wpointer-arith"
    "-Wconversion"
    "-Wsign-conversion"
    "-Wduplicated-cond"
    "-Wduplicated-branches"
    "-Wlogical-op"
    "-Wnull-dereference"
    "-Wdouble-promotion"
    "-Wfloat-equal"
    "-Walloca"
    "-Wvla"
    "-Wswitch-enum"
    "-Wswitch-default"
    "-Wimplicit-fallthrough=5"
    "-Wundef"
    "-Wunused"
    "-Wuninitialized"
    "-Wmaybe-uninitialized"
    "-Wstrict-overflow=4"
    "-Wstringop-overflow=4"
    "-Wstringop-truncation"
    "-Warray-bounds=2"
    "-Wattribute-alias=2"
    "-Wtrampolines"
    "-Wstack-protector"
    "-Werror=return-type"
    "-Werror=format-security"
    "-Werror=implicit-function-declaration"
  ];

  # File extensions considered C/C++ source (including GPU languages).
  cppExtensions = "*.c *.cc *.cpp *.cxx *.h *.hh *.hpp *.hxx *.cu *.cuh";

  # File extensions for source-only scanning (not headers).
  cppSourceExtensions = "*.c *.cc *.cpp *.cxx *.cu";

  # Directories to exclude from analysis.
  defaultExcludes = [
    "build"
    "build-*"
    ".git"
    "_deps"
    ".devcontainer"
    "lib"
    "__pycache__"
    "wheelhouse"
  ];

  # Generate find command fragments for excluding directories.
  mkExcludeArgs = excludes:
    lib.concatMapStringsSep " " (d: "-not -path '*/${d}/*'") excludes;
}
