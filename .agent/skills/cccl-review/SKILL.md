---
name: cccl-review
description: Use when reviewing CCCL code changes (a working-tree diff, PR, or commit range); loads the cccl-style and cccl-test skills and the CCCL review guidelines.
---

# CCCL Review

## Workflow

1. Determine the changed files (e.g. `git diff --name-only <base>...<head>`, or the working-tree diff).
2. Read `../cccl-style/SKILL.md` and follow its workflow for the changed files.
3. If tests are changed or the change requires new tests, read `../cccl-test/SKILL.md` and follow its workflow.
4. Read `docs/cccl/development/review_guidelines.md` and check the diff against every guideline whose scope matches.
5. Report each finding with a `critical:` / `important:` / `suggestion:` prefix (most severe first), the `file:line` location, and the violated guideline or rule. Do not report style issues that pre-commit tooling already enforces.
