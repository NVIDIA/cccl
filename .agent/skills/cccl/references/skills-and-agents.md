# Skills and agents — catalog and invocation mechanics

## Skill invocation mechanics

Skills are loaded by the harness via description-match (every session turn, every word in `description:` is
indexed for intent matching). When a user phrase triggers a skill, the harness loads the SKILL.md body for
that session. References (`references/*.md`) are **not** loaded automatically — the skill body loads on demand
by instructing the orchestrator to read a specific path.

Key behaviors:

- `description:` loads every turn — keep it 30–60 words; long descriptions are an anti-pattern.
- Skill body loads every trigger — keep to the essential workflow spine; push edge cases and templates to `references/`.
- Invoke via the **Skill tool** with `skill: <name>`. Skills are not reentrant.
- Entry skills (`cccl-*`) appear in slash autocomplete (`/cccl-` prefix). Detail skills (`cccl_detail-*`) do not.

## Agent invocation mechanics

Agents live at `.agent/agents/<name>.md`. Dispatch via the **Agent tool** with `subagent_type: <name>` and an
explicit `model:` parameter — the per-call value overrides frontmatter. Model tier:

- `haiku` — mechanical tasks: log parsing, SHA verification, JSON extraction.
- `sonnet` — multi-file reasoning or judgment (e.g. generating override matrices).

CCCL agents are leaf agents: non-interactive (no `AskUserQuestion`), no spawning subagents. User dialogue
belongs in the calling skill.

## Entry-skill catalog

| Skill                 | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `cccl`                | Entry-point router; load first in any session                           |
| `cccl-infra`          | Cross-functional infrastructure: CTK bumps, release cycles, CI/cmake/precommit/devcontainer/github/examples workflows |
| `cccl-bisect`         | Git bisect a regression via cloud workflow or local devcontainer        |
| `cccl-build`          | Configure and build targets; single-target and subproject builds        |
| `cccl-test`           | Run tests: ctest, lit, targeted regex; smallest-scope-first             |
| `cccl-ci`             | CI overview: matrix, skip tags, agents, override matrix, bench stub     |
| `cccl-clarify`        | Decision-point escalation for ambiguous choices or tricky tradeoffs     |
| `cccl-commit`         | Interactive commit prep: survey, split, chunks, message, commit         |
| `cccl-devcontainer`   | Launch devcontainers; Docker workflow; container selection               |
| `cccl-pr`             | Open, edit, comment on PRs; push + trigger CI                           |
| `cccl-python`         | Python bindings (cuda-cccl): build, test, publish                       |
| `cccl-resplit-branch` | Rebase and resplit a branch's commit history into a clean series        |
| `cccl-sass-diff`      | SASS/PTX comparison between two refs                                    |
| `cccl-triage`         | Diagnose CI failures on a PR or the latest nightly; optionally fix      |
| `cccl-docs`           | Sphinx docs, Doxygen, gen_docs.bash, docs-deploy.yml                    |
| `cccl-cmake`          | CMake presets, configuration options, usage entry point                 |
| `cccl-precommit`      | Pre-commit hooks, clang-format, ruff, gersemi, codespell, shellcheck    |
| `cccl-bench`          | Write benchmarks, nvbench, cccl.bench Python harness, CI bench requests |
| `cccl-libcudacxx`     | libcudacxx tour: stdlib, style, headers, internal macros                |
| `cccl-cub`            | CUB tour: block primitives, algorithms, device API, tests               |
| `cccl-thrust`         | Thrust tour: algorithms, policies, execution backends, tests            |
| `cccl-cudax`          | cudax tour: experimental features, containers, streams                  |
| `cccl-c`              | C Parallel Library: C bindings for CCCL algorithms                      |

## Detail-skill catalog

Detail skills auto-load via description match. They do **not** appear in slash autocomplete.

| Skill                             | Purpose                                                                          |
|-----------------------------------|----------------------------------------------------------------------------------|
| `cccl_detail-cmake`               | Buildsystem internals, arch variants, CMake helpers index                        |
| `cccl_detail-ci`                  | CI internals deep dive: workflow-build action, skip-tag plumbing, downstream     |
| `cccl_detail-release`             | Versioning pipeline: `cccl-version.json`, `update_version.sh`, release workflows, Python wheel versioning |
| `cccl_detail-github`              | Templates, CODEOWNERS, copy-pr-bot, CodeRabbit, non-CI workflows                 |
| `cccl_detail-examples`            | Top-level `examples/` CPM-consumption tests via `cccl_add_compile_test`          |
| `cccl_detail-test-params`         | `%PARAM%` convention via `cmake/CCCLTestParams.cmake`                            |
| `cccl_detail-cpp-macros`          | `_CCCL_*` internal macros: compiler detection, visibility, ABI, diagnostics      |
| `cccl_detail-devcontainer-matrix` | `make_devcontainers.sh`, 60+ container configs from `ci/matrix.yaml`, `verify-devcontainers.yml` |

## Agent catalog

| Agent                       | Model    | Purpose                                                            |
|-----------------------------|----------|--------------------------------------------------------------------|
| `cccl-ci-overrides`         | `sonnet` | Generate CI override matrix and skip-tag recommendations           |
| `cccl-ci-fetch-failures`    | `haiku`  | Fetch and parse failed CI job logs; return structured failure list |
| `cccl-ci-summarize-job-log` | `haiku`  | Summarize a single CI job log; return structured digest            |

## Naming convention

Format: `cccl[_detail]-<subarea>[-<topic>]*`. Entry skills use the `cccl-` prefix (slash-completable). Detail and
helper skills use `cccl_detail-` — the underscore-detail marker sits between `cccl` and the first kebab dash,
keeping them out of `/cccl-` autocomplete. Multiple topic suffixes allowed for hierarchical nesting (e.g.
`cccl_detail-cmake-helpers`). When naming, drop tokens that duplicate the parent directory path; do not
abbreviate semantic content.

## Where these files live

- Skills: `.agent/skills/<name>/SKILL.md`
- Skill references: `.agent/skills/<name>/references/*.md` (on-demand only; not auto-loaded)
- Agents: `.agent/agents/<name>.md`
- `.claude/skills` and `.claude/agents` are directory symlinks to `.agent/skills` and `.agent/agents` respectively
