# parse_error tests

## Purpose

- Validate that `ci/util/parse_error.py` reliably extracts error lines from
  configure, build, and test logs.
- Prevent regressions in the regexes, output formats, truncation rules, and
  multi‑file ordering.
- Provide ready‑made reference outputs (Markdown, GitHub summary, JSON) that
  make review diffs small and obvious when behavior changes.

## What’s here

- `*.log`: Minimal sample logs for common failure modes:
  - `configure*.log` — CMake configure diagnostics
  - `build.<tool>*.log` — compiler diagnostics (e.g., `build.clang.log`, `build.gcc.log`, `build.nvcc.log`, `build.nvcc.context.log`)
  - `ctest*.log` — CTest summary lines
  - `lit*.log` — LLVM lit failures and diagnostics
- `*.n{0|1}.{fmt}.{ext}`: Reference outputs produced by `parse_error.py` for
  each log, format, and `-n` setting used by the tests.
  - Formats: `md`, `json`.
  - Extensions: `.md` for `md`; `.json` for `json` (single extension).
  - Examples: `build.clang.n1.md`, `build.clang.n0.json`.
  - `nN`: `-n` arg; report first N errors; 0 is unlimited.
- `run_tests.py`: Enumerates all `*.log` files, runs `parse_error.py` with
  `-n 1` and `-n 0` across all formats, and compares the results to the
  references. It also verifies deterministic ordering for multi‑file input.

## How matching works (TL;DR)

- Regex sets live in `ci/util/parse_error.py`:
  - `CONFIGURE_PATTERNS`, `BUILD_PATTERNS`, `TEST_PATTERNS`.
  - They are evaluated in that order via `ERROR_PATTERNS`.
- Each regex should, when possible, capture these groups:
  - `file`: path or identifier of the source of the error
  - `line`: line number (numeric string)
  - `msg`: diagnostic text (should include the keyword such as `error` or
    `fatal` where applicable)
- On a match, the full original line is stored as `full` for use in the
  GitHub summary body.

## Output formats

- `--format md`: One line per match, ``- `file:line`: `message```.
- `--format md`: Plain Markdown with a shared summary line, a Location line,
  and a preformatted Full Error context block.
- `--format json`: JSON array of captured groups per match (`file`, `line`,
  `msg`, `full`).

## `-n` behavior

- `-n 1` (default): First matching line only.
- `-n 0`: All matching lines.

## Multi‑file input

- When multiple log files are provided, output is the concatenation of each
  file’s output in the same order as the filenames on the command line. The
  test suite verifies this by comparing combined output against the
  concatenation of the individual references.

## Running the tests

- Python:
  - `python3 "ci/test/parse_error/run_tests.py"`
- CTest (from repo root or a build tree configured with tests enabled):
  - `ctest -R cccl.ci.test.parse_error`

## Adding support for a new error

1) Add or update a regex
   - Edit `ci/util/parse_error.py` and add a `re.compile(...)` entry to the
     appropriate pattern list (`CONFIGURE_PATTERNS`, `BUILD_PATTERNS`, or
     `TEST_PATTERNS`).
   - Prefer anchored patterns (`^…$`) and keep them strict but minimal.
   - Capture `file`, `line`, and `msg` when available; omit when not present.
   - Include the error‑class keyword in `msg` where practical (e.g., `error`,
     `fatal`).

2) Create a minimal sample log
   - Add a new `*.log` under `ci/test/parse_error/` that contains a single,
     representative line that your regex should match. Keep it stable and
     deterministic; avoid machine‑specific paths when possible.

3) Generate the reference outputs
   - For a new log `ci/test/parse_error/mycase.log` and basename `mycase`:
     - Markdown (`-n 1` then `-n 0`):
       - `python3 "ci/util/parse_error.py" --format md -n 1 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n1.md.md"`
       - `python3 "ci/util/parse_error.py" --format md -n 0 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n0.md.md"`
     - Markdown:
       - `python3 "ci/util/parse_error.py" --format md -n 1 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n1.md.md"`
       - `python3 "ci/util/parse_error.py" --format md -n 0 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n0.md.md"`
     - JSON:
       - `python3 "ci/util/parse_error.py" --format json -n 1 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n1.json.json"`
       - `python3 "ci/util/parse_error.py" --format json -n 0 "ci/test/parse_error/mycase.log" > "ci/test/parse_error/mycase.n0.json.json"`

   Tip: To regenerate references for all logs and formats in bulk:

   ```bash
   for log in ci/test/parse_error/*.log; do base=$(basename "$log" .log); \
     for n in 1 0; do \
       python3 "ci/util/parse_error.py" --format md   -n "$n" "$log" > "ci/test/parse_error/${base}.n${n}.md"; \
       python3 "ci/util/parse_error.py" --format json -n "$n" "$log" > "ci/test/parse_error/${base}.n${n}.json"; \
      done; \
    done
    ```

    Manually review these to ensure that they are correct and as expected.

4) Verify
   - Run: `python3 "ci/test/parse_error/run_tests.py"`
   - Or via CTest: `ctest -R cccl.ci.test.parse_error`

## Guidelines for logs and patterns

- Keep logs minimal, but include the full error message.
- Regex evaluation order matters. If a new regex is more general, place it
  after more specific ones to avoid unintended matches.
