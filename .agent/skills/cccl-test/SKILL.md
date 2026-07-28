---
name: cccl-test
description: Use when writing, updating, reviewing, or validating CCCL tests; read common CCCL test guidance and the path-specific references named by this skill.
---

# CCCL Test

## Workflow

1. Read `references/common.md` for guidance that applies across CCCL tests.
2. For `libcudacxx/test/**/*`, also read `references/libcudacxx.md`.
3. If no path-specific reference exists, follow nearby tests and repository docs. Do not import test rules from another subproject.
4. Apply each reference only to its stated scope. Rules for one CCCL subproject do not automatically apply to another.
