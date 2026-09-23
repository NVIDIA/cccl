---
name: cccl-style
description: Use when editing or reviewing CCCL code for style conventions; read common CCCL guidance and the path-specific references named by this skill.
---

# CCCL Style

## Workflow

1. Read the [CCCL C++ Coding Guidelines](https://nvidia.github.io/cccl/unstable/cccl/development/coding_guidelines.html),
   which supersede everything.
2. Then read `references/common.md`.
3. For `cub/**/*`, also read `references/cub.md`.
4. For `libcudacxx/include/**/*`, also read `references/libcudacxx.md`.
5. For `cudax/include/**/*`, also read `references/libcudacxx.md`.
6. For `cub/cub/device/**/*`, also read the documentation at `docs/cub/developer/device_scope.rst` and `docs/cub/device_wide.rst`.
7. For `cub/cub/block/**/*`, also read the documentation at `docs/cub/developer/block_scope.rst`.
8. For `cub/cub/warp/**/*`, also read the documentation at `docs/cub/developer/warp_level.rst`.
9. For `cub/cub/thread/**/*`, also read the documentation at `docs/cub/developer/thread_level.rst`.
10. For `cub/test/**/*`, also read the documentation at `docs/cub/developer/test_overview.rst`.
11. If no path-specific reference exists, follow nearby code and repository docs. Do not import rules from another subproject.
12. Apply each reference only to its stated scope. Rules for one CCCL subproject do not automatically apply to another.
