# Agent Instructions

## Load the `cccl` skill first

Load the `cccl` skill via the Skill tool. It maps the available repo-local skills and agents and routes by user intent.

If you don't know what skills or agents are, load `cccl-agent-impl` first.

## What CCCL is

CCCL — the CUDA Core Compute Libraries — is a collection of CUDA C++ libraries and Python packages:

- **libcudacxx** — CUDA C++ Standard Library
- **CUB** — Block-level primitives
- **Thrust** — High-level parallel algorithms
- **cudax** — Experimental features
- **C Parallel Library** — C bindings for CCCL algorithms
- **cuda-cccl Python packages** — Python bindings for parallel + cooperative primitives

Built with CMake + Ninja via the presets in `CMakePresets.json`.

## Repository layout

```
cccl/
├── .agent/skills/      <- canonical skills (one dir per skill)
├── .agent/agents/      <- canonical agents (one file per agent)
├── .claude/skills      -> ../.agent/skills (directory symlink)
├── .claude/agents      -> ../.agent/agents (directory symlink)
├── .claude/settings.json
├── .devcontainer/      <- Docker containers for reproducible builds
├── .github/            <- workflows, copy-pr-bot
├── libcudacxx/         <- CUDA C++ Standard Library
├── cub/                <- CUB primitives
├── thrust/             <- Thrust algorithms
├── cudax/              <- experimental features
├── c/                  <- C Parallel library
├── python/cuda_cccl/   <- Python bindings
├── ci/                 <- build/test scripts + matrix.yaml
├── docs/               <- Sphinx documentation source
├── examples/           <- usage examples
├── AGENTS.md           <- this file
├── CLAUDE.md           -> AGENTS.md (symlink)
└── CMakePresets.json
```

`.agent/` is canonical; `.claude/skills` and `.claude/agents` symlink to it so both Claude Code and Codex find the
same files.

## Skill routing

The `cccl` skill carries the full table. Common entries:

- Commit uncommitted changes / wrap up a fix → `cccl-commit`
- Resplit / clean up a branch's commit history → `cccl-resplit-branch`
- Open / edit / comment on a PR / trigger CI → `cccl-pr`
- CI overview / matrix / skip tags / `/ok to test` → `cccl-ci`
- Triage failed CI → `cccl-triage-pr` or `cccl-triage-nightly`
- Benchmarks → `cccl-ci-benchmarks`
- Git bisect → `cccl-bisect`
- Devcontainers → `cccl-devcontainers`
- Targeted build/test (fast iteration) → `cccl-build-and-test-targets`
- Full-matrix C++ build/test scripts → `cccl-cpp-builds`
- Python packages (cuda-cccl) → `cccl-python`
- libcudacxx code style → `cccl-libcudacxx-style`
- SASS / PTX comparison → `cccl-sass-diff`
- Stuck on a decision → `cccl-clarify`

Reference docs: `CONTRIBUTING.md`, `ci-overview.md`, `docs/cccl/development/`.

## Known agent limitations

- Long-running builds (60+ min) and tests (30+ min) are normal — never cancel them. Use
  `cccl-build-and-test-targets` for fast iteration.

## Pre-commit

Run `pre-commit run --files <files>` before committing. CI's linters will fail otherwise.

## Reset before final merge

These files block merging in their non-default state:

- Non-empty `workflows.override` in `ci/matrix.yaml` → reset to empty.
- `[skip-*]` tags in the last commit message → remove and re-push.
- Modified `ci/bench.yaml` → restore to match `ci/bench.template.yaml`.
