---
name: cccl
description: "Entry-point orientation for the CCCL repository. Surfaces the available CCCL-specific skills and agents and points at common entry phrases. Load this skill first in every CCCL session before doing other work. Use when starting any task in this repo, when unsure which CCCL skill to use, or when introduced to the repo cold."
---

# cccl

Skills live in `.agent/skills/`; agents live in `.agent/agents/`. `.claude/skills` and `.claude/agents` symlink to
those so Claude Code and Codex find the same files.

If you don't know how skill or agent invocation works, load `cccl-agent-impl` first.

## Where to start by intent

| Intent                                         | Load                                                |
|------------------------------------------------|-----------------------------------------------------|
| Commit uncommitted changes / wrap up a fix     | `cccl-commit`                                       |
| Resplit / clean up a branch's commit history   | `cccl-resplit-branch`                               |
| Open / edit / comment on a PR / trigger CI     | `cccl-pr`                                           |
| Diagnose CI on this PR / what's failing        | `cccl-triage-pr`                                    |
| Triage nightly / fix nightly CI                | `cccl-triage-nightly`                               |
| Stuck on a decision / should I X or Y          | `cccl-clarify`                                      |
| Post `/ok to test`                             | `cccl-ok-to-test` agent (called by a skill)         |
| Generate override matrix / skip tags           | `cccl-ci-overrides` agent (called by a skill)       |
| How does CI work / where is X CI defined       | `cccl-ci`                                           |
| Set up a benchmark on this PR                  | `cccl-ci-benchmarks`                                |
| Git bisect a regression                        | `cccl-bisect`                                       |
| Build / test in the devcontainer               | `cccl-devcontainers`, `cccl-build-and-test-targets` |
| Build cub / thrust / libcudacxx / cudax (full) | `cccl-cpp-builds`                                   |
| Work on / build / test the python bindings     | `cccl-python`                                       |
| Check for SASS/PTX changes                     | `cccl-sass-diff`                                    |
| libcudacxx code style                          | `cccl-libcudacxx-style`                             |

## Repo conventions

- **Scratch space**: `/tmp/claude/<sessionid>/`. Create with `mkdir -p`. Don't pipe; redirect to a file and Read.
- **CI** uses `ci/matrix.yaml` with optional `workflows.override` to scope PR jobs; `[skip-*]` commit tags scope
  further. Both block merging while present. See `cccl-ci`.
- **`/ok to test <SHA>`** is required from a maintainer for external PRs. The `cccl-ok-to-test` agent posts it.
