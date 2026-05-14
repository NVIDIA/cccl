---
description: |
  CCCL GitHub-side infrastructure — issue and PR templates, CODEOWNERS review routing,
  copy-pr-bot (internal mirror and CI trust), CodeRabbit AI review configuration,
  non-CI automation workflows (backport, triage rotation, project sync, release tooling,
  docs deploy, blackduck), and release changelog config.
  Triggers: "how does CODEOWNERS work", "add someone to copy-pr-bot", "issue templates",
  "coderabbit config", "how does backporting work", "PR template".
---

Reference skill — CCCL's GitHub-side repository infrastructure. Covers templates, bots, and automation workflows that are not part of the build/test CI pipeline. For CI matrix, skip tags, and `/ok to test`, see `cccl-ci`.

## Issue templates

Six structured templates under `.github/ISSUE_TEMPLATE/`:

| File                  | Title prefix | Auto-label | Purpose                                           |
|----------------------|---|---|---|
| `1-bug_report.yml`    | `[BUG]:`     | —          | Bug type dropdown, component, reproducer, system info |
| `2-feature_request.yml` | `[FEA]:`   | —          | Area dropdown, problem + solution + alternatives |
| `3-doc_request.yml`   | `[DOC]:`     | —          | New vs correction, link to incorrect/missing docs |
| `infra_ticket.yml`    | `[INFRA]:`   | `infra`    | CMake/GHA/CI/devcontainer requests; Slack for critical issues |
| `devex_ticket.yml`    | `[DEVEX]:`   | `devex`    | Low-priority developer-experience improvements |
| `config.yml`          | —            | —          | Enables blank issues; links to Discussions and Discord |

All templates auto-add to `NVIDIA/6` GitHub project board. Load-bearing fields: **component/area dropdown** (routes triage), **duplicate confirmation checkbox** (required on bug/doc), **reproduction link** (optional but preferred on bugs).

## PR template

`.github/PULL_REQUEST_TEMPLATE.md` — three load-bearing elements:

- `closes <!-- Link issue here -->` — PR title feeds CHANGELOG; issue link drives project board sync.
- Tests checkbox — reviewers check this; CI does not enforce it.
- Documentation checkbox — signal to reviewer, not gating.

PR title is included verbatim in the release CHANGELOG (via `.github/release.yml` label-to-category mapping).

## CODEOWNERS

`.github/CODEOWNERS` routes required reviews by path:

| Pattern                                             | Team                                   |
|-----------------------------------------------------|----------------------------------------|
| `thrust/`, `cub/`, `libcudacxx/`, `cudax/`, `c/`, `python/` | Per-library `cccl-*-codeowners` teams |
| `.github/`, `ci/`, `.devcontainer/`, `.pre-commit-config.yaml`, `.clang-format`, `.clangd`, `c2h/`, `nvbench_helper/`, `.vscode`, `.coderabbit.yaml` | `cccl-infra-codeowners` |
| `**/CMakeLists.txt`, `**/cmake/`                    | `cccl-cmake-codeowners` (cudax test CMakeLists overrides to cudax team) |
| `benchmarks/`, `**/benchmarks`                      | `cccl-benchmark-codeowners` |
| `README.md`, `docs/`, `examples/`                   | `cccl-codeowners` (general)            |

All teams are under the `@nvidia/` org prefix.

## copy-pr-bot

`.github/copy-pr-bot.yaml` — GitHub App that mirrors every public PR to NVIDIA's internal repo so internal CI runners can safely execute it (public runners cannot access private infrastructure).

Config knobs:
- `enabled: true` — bot is active.
- `auto_sync_draft: false` — draft PRs are not mirrored until marked ready.
- `additional_trustees` — GitHub usernames whose PRs are automatically trusted for mirroring without a manual `/ok to test` approval. Currently `ahendriksen` and `gonzalobg`.

The `/ok to test` trigger that fires CI on the mirrored PR is handled separately — see `cccl-ci`.

## CodeRabbit

`.coderabbit.yaml` at repo root (owned by `cccl-infra-codeowners`). AI review bot that posts inline comments on PRs targeting `main` or `branch/X.Y.x` release branches.

Key configuration:
- **Profile:** `chill` — avoids nitpicky style comments.
- **Comment prefixes:** `suggestion:` / `important:` / `critical:` — three-level severity. No praise, no headings, no emoji.
- **Drafts:** skipped (`auto_review.drafts: false`).
- **Ignored bots:** `copy-pr-bot`, `dependabot[bot]`, `github-actions[bot]`, `nv-automation-bot`.
- **Per-path instructions:** each major subdirectory (`libcudacxx/`, `cub/`, `thrust/`, `cudax/`, `c/`, `python/`, `benchmarks/`, `docs/`, `ci/`, `.github/`) has a path instruction block tuning review focus.
- **Knowledge base:** ingests `AGENTS.md`, `CONTRIBUTING.md`, the libcudacxx style skill, and key RST docs.
- **Finishing touches:** Autofix enabled; docstring/unit-test/simplify generation disabled.

## Non-CI workflows

Workflows under `.github/workflows/` that are not CCCL build/test pipelines:

| Workflow                            | Trigger                | Role                                              |
|-------------------------------------|----------------------|--------------------------------------------------|
| `backport-prs.yml`                  | PR merged or `/backport` comment | Opens backport PR to release branches |
| `triage_rotation.yml`               | Issue opened (non-member) | Auto-assigns external issues for triage |
| `project_automation_sync_pr_issues.yml` | PR opened/edited     | Syncs linked issues to project board |
| `project_automation_set_in_progress.yml` | PR opened/edited    | Moves PR + issues to "In Progress" |
| `project_automation_set_in_review.yml` | PR event            | Moves to "In Review" state             |
| `project_automation_set_roadmap.yml` | Issue/PR event       | Sets roadmap column on project board   |
| `docs-deploy.yml`                   | Push to `main` or manual | Builds and deploys Sphinx docs to GitHub Pages |
| `update-branch-version.yml`         | Manual               | Bumps version numbers in a release branch |
| `release-create-new.yml`            | Manual               | Creates new release branch + initial PR |
| `release-update-rc.yml`             | Manual               | Updates release candidate state        |
| `release-finalize.yml`              | Manual               | Finalizes and tags a release           |
| `release-wheels.yml`                | Manual               | Publishes Python wheels to PyPI        |
| `blackduck-sca.yml`                 | Push to `main` or manual | Black Duck software composition analysis (security/license) |
| `build-rapids.yml`                  | Manual/scheduled      | Downstream compatibility — builds RAPIDS against CCCL |
| `build-pytorch.yml`                 | Manual/scheduled      | Downstream compatibility — builds PyTorch against CCCL |
| `build-matx.yml`                    | Manual/scheduled      | Downstream compatibility — builds MatX against CCCL |
| `verify-devcontainers.yml`          | PR/push              | Verifies devcontainer image builds     |

Release workflows have a sequenced README at `.github/workflows/release-README.md`.

## Release changelog

`.github/release.yml` — GitHub's auto-generated release notes config. Maps PR labels to CHANGELOG sections: Thrust/CUB, libcudacxx, cuda.coop, cuda.compute, Documentation, Other Changes. PR title is the changelog entry — keep titles descriptive.

## Additional resources

- `references/docs.md` — index of GitHub-side infrastructure documentation (branching, backport, release workflows).
