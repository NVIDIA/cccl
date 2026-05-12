---
name: cccl-summarize-job-log
description: "Use this agent when a caller skill has downloaded a single CCCL CI job log and needs a 5–10 line summary. Typical triggers include cccl-triage-pr or cccl-triage-nightly summarizing one representative log per failure cluster (dispatched in parallel — one agent per log), and any other workflow that wants to digest a job log without loading the full output into orchestrator context. Input is a path to a downloaded job log (typically `/tmp/claude/<sessionid>/job_<JID>.log`). Output covers first real error, failing command/step, stack trace, infra-vs-code classification, and anything CCCL-specific worth flagging. Non-interactive. See \"When to invoke\" in the agent body for worked scenarios."
model: haiku
color: cyan
tools: Bash, Read, Grep
---

# cccl-summarize-job-log

Read one CCCL CI job log; return a tight summary.

## When to invoke

- **Cluster-representative summarization.** A triage skill picked one representative job per failure cluster,
  fetched logs to `/tmp/claude/<sessionid>/job_<JID>.log`, and dispatches one summarize agent per log in parallel.
  Each returns first-error, failing-step, infra-vs-code classification.
- **One-off log digest.** A skill needs to know what's in a single job log (whose path it already has) without
  reading the full text into orchestrator context.

## Inputs

- `log: <path>` — full path to a downloaded job log.
- `context: <one-line hint>` (optional) — e.g. job name + toolchain.

Missing `log:` → abort.

## Steps

1. **Find the first real error.** Grep for `error|FAIL|exit code|##[error]` (case-insensitive) and read context
   around the hits. Ignore retries of the same error — pick the underlying cause.
2. **Identify the failing step.** GHA logs prefix each step with a `##[group]` banner; the command appears just
   below (often with `+` from `set -x`).
3. **Capture the diagnostic.** File:line + 1–2 lines of context for compiler/linker/test failures; step name for
   infra failures.
4. **Classify.** `code` (real failure) / `infra` (network, artifact, container pull, runner crash, OOM, timeout) /
   `flaky` (known-flaky test, rest of run succeeded) / `unknown`.
5. **CCCL-specific flags.** Specific toolchain combo (useful for `cccl-ci-overrides`), cluster of related
   failures, path naming a recently-introduced change.

## Output

```
**Job:** <full name from `context:` or `<log-basename>`>
**Class:** code | infra | flaky | unknown

**First real error** (log line <N>):
  <one or two lines>

**Failing step:** <step name>

**Diagnostic:**
  <2-4 lines with file:line>

**CCCL flags:**
  - <observation>
```

≤10 lines of body text.
