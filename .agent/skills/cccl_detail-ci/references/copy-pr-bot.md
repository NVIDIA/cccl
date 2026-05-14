# copy-pr-bot and /ok to test

## Why it exists

CCCL uses NVIDIA self-hosted GPU runners. For security, PR code from external
contributors must not run on those runners until reviewed. `copy-pr-bot`
enforces this gate by copying approved code to a separate branch that the
`pull_request` workflow actually watches.

## Trigger: the pull_request workflow

```yaml
on:
  push:
    branches:
      - "pull-request/[0-9]+"
```

The workflow does not trigger on `pull_request` events. It triggers on pushes
to branches named `pull-request/<N>`. The bot creates and updates these
branches; direct pushes are also accepted for internal contributors.

## External contributor flow

1. Contributor opens a PR from a fork or feature branch.
2. Reviewer inspects the changes.
3. Reviewer comments `/ok to test <SHA>` where `<SHA>` is the PR head SHA.
4. copy-pr-bot verifies the SHA matches `github.event.pull_request.head.sha`.
5. Bot pushes the code to `pull-request/<PR number>`.
6. The `pull_request` workflow triggers.

The SHA verification prevents reviewers from accidentally approving a commit
that was pushed after their review.

## Internal contributor flow

Internal contributors with SSH-signed commits do not need `/ok to test`. A
signed push to the PR branch (which is typically already a `pull-request/<N>`
branch for internal contributors) triggers CI automatically.

SSH signing setup:

```bash
git config --global gpg.format ssh
git config --global user.signingKey ~/.ssh/<KEY>.pub
git config --global commit.gpgsign true
```

Upload the key as a **Signing Key** (not just authentication key) at
`github.com/settings/keys`.

## additional_trustees

`.github/copy-pr-bot.yaml`:

```yaml
additional_trustees:
  - ahendriksen
  - gonzalobg
```

Users listed here may use `/ok to test` in addition to the default set
(NVIDIA org members with write access). The `auto_sync_draft` field controls
whether draft PRs are auto-synced; it is `false` in CCCL.

## Concurrency and re-runs

The `pull_request` workflow has:

```yaml
concurrency:
  group: ${{ github.workflow }}-on-${{ github.event_name }}-from-${{ github.ref_name }}
  cancel-in-progress: true
```

A new `/ok to test` (new push to the `pull-request/<N>` branch) cancels any
in-progress run for that PR.

## No automatic run for new commits

Each new commit on an external PR requires a new `/ok to test <SHA>`. The SHA
must match the new head exactly. This prevents a reviewer from approving old
code that has since been superseded.
