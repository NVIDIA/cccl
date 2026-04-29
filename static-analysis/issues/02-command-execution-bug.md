# Issue: Accidental command execution in build_cuda_cccl_wheel.sh

**Template:** `[BUG]: if $(git rev-parse ...) executes output as command in build_cuda_cccl_wheel.sh`
**Component:** Infrastructure
**Severity:** Low (works by coincidence, but fragile)

## Description

In `ci/build_cuda_cccl_wheel.sh:26`, the shallow-repository check uses `if $(git rev-parse --is-shallow-repository)`, which executes the *output* of `git rev-parse` as a shell command rather than testing its exit code.

`git rev-parse --is-shallow-repository` prints the literal string `true` or `false` to stdout. The `$()` captures that string and executes it. This works because `true` and `false` are shell builtins that exit 0 and 1 respectively — but the mechanism is accidental. If `git rev-parse` ever produced unexpected output (error message, warning), that output would be executed as a command.

## Fix

```bash
# Current (line 26):
if $(git rev-parse --is-shallow-repository); then

# Proposed — test exit code directly via string comparison:
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
```

Note: `git rev-parse --is-shallow-repository` always exits 0 regardless of the answer (it prints "true" or "false" as text), so using it directly as `if git rev-parse --is-shallow-repository` would not work — the exit code doesn't differentiate. The string comparison is the correct approach.

## Verification

The behavior is identical for the two possible outputs ("true" / "false"). The fix is strictly safer because it handles unexpected output gracefully (the `[` test would fail closed rather than executing arbitrary text).

## Detection

Found via ShellCheck 0.11.0 (SC2091: "You appear to be executing a command output, not testing it"). Full analysis: `static-analysis/shellcheck.md`.
