.. _cccl-pr-review:

PR Review Workflow
==================

This workflow gives Codex and human reviewers a consistent way to review CCCL
pull requests. The goal is to make the review start from the issue and the PR
intent, not from an isolated diff.

Quick Start
-----------

Start from the checked-out PR branch in the repository root. Use the repo-local
``review-cccl`` skill with an explicit mode:

.. code-block:: text

   /review-cccl context
   /review-cccl feedback
   /review-cccl adversarial

If no mode is provided, the workflow defaults to ``context``.

Context mode is the first pass. It gathers PR context and explains the change,
but does not produce review comments:

.. code-block:: bash

   ci/util/pr_review_context.sh

Ask Codex to summarize the linked issue, PR intent, implementation strategy,
and a sensible file review order. Codex should not produce review findings in
this phase.

By default, the helper fetches ``upstream/main``, computes the merge base with
``HEAD``, prints commits, diff summaries, changed files, and any linked GitHub
issues it can find in commit messages or the branch name. Use ``--no-fetch`` if
the base ref is already fresh.

Use an explicit PR or issue when the local branch cannot reveal it:

.. code-block:: bash

   ci/util/pr_review_context.sh --pr 1234
   ci/util/pr_review_context.sh --issue 1234

After the human reviewer has inspected the PR, use ``/review-cccl feedback`` to
request a normal review pass. For high-risk changes, use
``/review-cccl adversarial`` after the PR context is understood.

Review Protocol
---------------

1. Read ``AGENTS.md`` first.
2. Run ``ci/util/pr_review_context.sh`` or manually gather the same context.
3. If no issue or PR context is discoverable, clearly flag that the review is
   missing required context. Do not silently review a context-free diff.
4. Read the linked issue description and PR description before reviewing the
   implementation.
5. Summarize the exact problem the PR is solving, how the PR attempts to solve
   it, and a sensible file review order.
6. Pause for the human reviewer to inspect the PR. Answer code-explanation
   questions during this phase, but do not produce review findings yet.
7. When the user asks for a review pass, review for correctness, performance,
   API compatibility, maintainability, and test coverage.
8. Report findings first, ordered by severity, with file and line references.
   Keep summaries secondary to concrete review comments.

Useful Git commands:

.. code-block:: bash

   git fetch upstream main
   base="$(git merge-base HEAD upstream/main)"
   git log --oneline "$base"..HEAD
   git diff --stat "$base"..HEAD
   git diff "$base"..HEAD

Useful GitHub CLI commands:

.. code-block:: bash

   gh pr view --json number,title,body,closingIssuesReferences,commits,files
   gh issue view <issue-number>

Review Focus
------------

When asked for a review pass, focus on correctness, performance, and consistency
with existing code. Let the changed files determine which risks matter, but pay
special attention to CCCL-specific contracts such as CUDA/device behavior,
iterator and aliasing semantics, overload constraints, API compatibility,
temporary storage, streams, tuning/env propagation, and test coverage when they
are relevant.

TODO: Refer to skills in .agent (not .agents)