How To Plan Work with GitHub Issues
===================================

CCCL uses `GitHub issues <https://github.com/NVIDIA/cccl/issues>`__ and the
`CCCL GitHub Project <https://github.com/orgs/NVIDIA/projects/6>`__ as the source of truth to make
planned work visible, understandable, and coordinated.

This guide explains how CCCL plans work, records that
work in closable issues, and keeps priorities and ownership visible over time.

CCCL plans work in monthly sprints, with each sprint identifying the issues the
team intends to prioritize during that 4-week window.

Understand current planned work
-------------------------------

The `Current Sprint <https://github.com/orgs/NVIDIA/projects/6/views/47>`__ view in the GitHub Project contains
the planned work the team has agreed to prioritize now. It is not a complete
list of all engineering activity.

Planned work intentionally leaves capacity for reviews, support, debugging, and
other interrupt-driven work.

In-progress work is not automatically carried forward into the next planning
cycle. Each cycle is planned from current priorities.

Write a closable issue
----------------------

Every issue should answer:

   "This issue can be closed when..."

Examples:

- a bug is fixed;
- a refactoring is complete;
- a benchmark is added;
- a design decision is documented;
- follow-on issues are created;
- an investigation summary is written.

Large efforts can start as tracking issues whose scope is still evolving. As
concrete work becomes clear, create sub-issues for work with independent close
conditions.

Propose planned work
--------------------

Sprint planning happens monthly. Team leads review the roadmap, current priorities,
and incoming requests to identify the highest-impact work for the upcoming
four-week planning window.

If you think work should be considered for the next planning cycle, raise it
during planning or ask a team lead to add it to the
`Sprint Planning <https://github.com/orgs/NVIDIA/projects/6/views/58>`__ view.

During planning, the team reviews proposed issues, confirms ownership,
identifies gaps or dependencies, and checks that issues are actionable and
up-to-date.

Confirmed planned work is moved to the
`Current Sprint <https://github.com/orgs/NVIDIA/projects/6/views/47>`__ view.

Keep assigned issues current
----------------------------

The assignee is responsible for driving the issue forward and keeping it
accurate while the work evolves.

Update the issue when:

- the title no longer describes the work clearly;
- the close condition changes;
- the work has been split into sub-issues;
- important context, decisions, or blockers appear;
- the issue is no longer relevant and should be closed.

Use issue comments for questions, decisions, blockers, and context that
others may need to find later. If a comment changes the issue's scope or close
condition, update the issue body as well.

Complete assigned issues
------------------------

Use the issue's close condition to decide when the work is done. Often, this
means opening a PR that completes the work described by the issue. Follow the
:doc:`contributing guidelines </cccl/contributing>` for how to prepare and
submit the PR.

Link the PR to the issue so GitHub can close the issue when the PR merges. For
example, include ``Fixes #123`` in the PR description.

Linked PRs also keep the GitHub Project status current. A linked draft PR keeps
the issue ``In Progress``. When the PR is ready for review, automation moves the
issue to ``Review``.

Plan release-critical work
--------------------------

Upcoming releases have corresponding
`GitHub milestones <https://github.com/NVIDIA/cccl/milestones>`__.

Team leads create release milestones and assign issues to them during planning.
Add an issue to a release milestone when the work is
important to complete and include in that release.

Not all current planned work is tied to a particular release.
