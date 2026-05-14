# Agent Instructions

## Load the `cccl` skill first

Load the `cccl` skill via the Skill tool. It is the entry-point router for the `cccl-*` skill and agent
family, and carries the full intent → skill routing table. Every session begins here.

The `cccl` skill points at `references/skills-and-agents.md` for the complete catalog and invocation
mechanics.

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

`.agent/` is canonical; `.claude/skills` and `.claude/agents` symlink to it so both Claude Code and Codex find
the same files. Reference docs live in `CONTRIBUTING.md`, `ci-overview.md`, and `docs/cccl/development/`.

## Skill and agent naming

- Entry skills use the `cccl-` prefix and appear in `/cccl-` slash autocomplete.
- Detail skills use the `cccl_detail-` prefix (underscore between `cccl` and `detail`), auto-load via
  description match, and are excluded from slash autocomplete.

## Known limitations

Long-running builds (60+ min) and tests (30+ min) are normal — never cancel them. The `cccl-build` and
`cccl-test` skills cover fast-iteration targeted builds and tests.

## Pre-commit

Run `pre-commit run --files <files>` before committing. CI's linters will fail otherwise.

## Reset before final merge

These files block merging in their non-default state:

- Non-empty `workflows.override` in `ci/matrix.yaml` → reset to empty.
- `[skip-*]` tags in the last commit message → remove and re-push.
- Modified `ci/bench.yaml` → restore to match `ci/bench.template.yaml`.
