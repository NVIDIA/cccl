---
name: cccl-style
description: Use when editing or reviewing CCCL code for style conventions; read common CCCL guidance and the path-specific references named by this skill.
---

# CCCL Style

## Workflow

1. Read `references/common.md` for guidance that applies across CCCL.
2. For `libcudacxx/include/**/*`, also read `references/libcudacxx.md`.
3. For `cudax/include/**/*`, also read `references/libcudacxx.md`.
4. If no path-specific reference exists, follow nearby code and repository docs. Do not import rules from another subproject.
5. Apply each reference only to its stated scope. Rules for one CCCL subproject do not automatically apply to another.
