# Documentation index — cccl

Top-level orientation documents. Relevant to any CCCL session as background context.

## Primary

| Path | What it covers |
|------|----------------|
| `README.md` | Welcome guide: mission, the three core libraries (CUB, Thrust, libcudacxx), quick links. |
| `CONTRIBUTING.md` | Getting started: fork, branch conventions, devcontainer setup, pre-commit, first PR. |
| `AGENTS.md` | Agent-specific instructions for building, testing, and contributing to CCCL. |
| `docs/index.rst` | Main Sphinx landing page; links to C++, Python, and maintainer doc sections. |
| `docs/cpp.rst` | C++ libraries landing page (CUB, Thrust, libcudacxx, cudax). |

## Adjacent

| Path | What it covers |
|------|----------------|
| `docs/cccl/config_macros.rst` | CMake and compile-time configuration options across CCCL libraries. |
| `docs/cccl/3.0_migration_guide.rst` | Breaking changes and upgrade path from CCCL 2.x to 3.x. |
| `docs/cccl/tma.rst` | Tensor Memory Accelerator (TMA) hardware feature: API and usage in CCCL. |
| `ci-overview.md` | CI environment, matrix.yaml structure, skip tags, override matrix, troubleshooting. |

## See also

- `cccl-build` `references/docs.md`, `cccl-test` `references/docs.md` — build/test documentation.
- Per-library skills for library-specific docs (cccl-cub, cccl-thrust, cccl-libcudacxx, cccl-cudax).
