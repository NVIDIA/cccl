---
name: cccl-style
description: Use when editing or reviewing CCCL code for style conventions; read common CCCL guidance and the path-specific references named by this skill.
---

# CCCL Style

## Workflow

1. Read `references/common.md` for guidance that applies across CCCL.
2. For `libcudacxx/include/**/*`, also read `references/libcudacxx.md`.
3. For `cudax/include/**/*`, also read `references/libcudacxx.md`.
4. For `cub/cub/device/**/*`, also read the documentation at `docs/cub/developer/device_scope.rst` and `docs/cub/device_wide.rst`.
5. For `cub/cub/block/**/*`, also read the documentation at `docs/cub/developer/block_scope.rst`.
6. For `cub/cub/warp/**/*`, also read the documentation at `docs/cub/developer/warp_level.rst`.
7. For `cub/cub/thread/**/*`, also read the documentation at `docs/cub/developer/thread_level.rst`.
8. For `cub/test/**/*`, also read the documentation at `docs/cub/developer/test_overview.md`.
9. If no path-specific reference exists, follow nearby code and repository docs. Do not import rules from another subproject.
10. Apply each reference only to its stated scope. Rules for one CCCL subproject do not automatically apply to another.
