---
description: "Entry-point router for the cccl-* skill and agent family. Load first in any CCCL session. Routes by intent: commit, branch, PR, CI, build/test, libraries, infrastructure, benchmarks, docs. Triggers: \"where do I start\", \"which skill should I use\", \"new to this repo\", \"what cccl skill handles X\"."
---

# cccl

Entry-point router for the cccl-* skills and agents.

Skills live under `.agent/skills/<name>/SKILL.md`; agents under `.agent/agents/<name>.md`. `.claude/skills` and `.claude/agents` are directory symlinks to those.

Entry skills (`cccl-*`) are slash-completable workflow entry points. Detail skills (`cccl_detail-*`) auto-load via description match and are excluded from slash autocomplete. See `references/skills-and-agents.md` for the full catalog and invocation mechanics.

## Where to start by intent

| Intent                               | Load                     |
|--------------------------------------|--------------------------|
| Commit / wrap up a fix               | `cccl-commit`            |
| Resplit / clean up commit history    | `cccl-resplit-branch`    |
| Open / edit / comment on a PR        | `cccl-pr`                |
| Diagnose CI failures (PR or nightly) | `cccl-triage`            |
| Stuck on a decision                  | `cccl-clarify`           |
| CI overview / matrix / skip tags     | `cccl-ci`                |
| Benchmarks / perf comparisons        | `cccl-bench`             |
| Git bisect a regression              | `cccl-bisect`            |
| Build a target                       | `cccl-build`             |
| Run tests                            | `cccl-test`              |
| Launch a devcontainer                | `cccl-devcontainer`      |
| Cross-functional infra               | `cccl-infra`             |
| Python bindings                      | `cccl-python`            |
| SASS / PTX comparison                | `cccl-sass-diff`         |
| Pre-commit / linters / formatters    | `cccl-precommit`         |
| CMake presets / configuration        | `cccl-cmake`             |
| Sphinx docs / Doxygen                | `cccl-docs`              |
| libcudacxx                           | `cccl-libcudacxx`        |
| CUB                                  | `cccl-cub`               |
| Thrust                               | `cccl-thrust`            |
| cudax                                | `cccl-cudax`             |
| C Parallel Library                   | `cccl-c`                 |

## Additional resources

- `references/skills-and-agents.md` — full entry-skill catalog, detail-skill catalog, agent catalog, naming convention, invocation mechanics.
- `references/docs.md` — index of top-level CCCL orientation documentation.
