Backport Process
================

This page documents how changes are made to release branches.

In an ideal world, the release tag ``vX.Y.Z`` would match the exact commit where
``branch/X.Y.Z`` was created from ``main``, with no additional fixes needed.
In practice, important bugs are sometimes found after that cut, and we need
surgical fixes in a prior release branch without taking every newer change from
``main``. These targeted fixes are called *backports*: the fix is made via a standard
PR to the ``main`` branch, then automation opens an equivalent PR against the relevant
release branch.

To land a fix in a release branch, follow these steps:

#. Make the fix via a normal PR against ``main``.
#. If the fix meets the criteria below for inclusion in a release branch, add
   the label ``backport branch/<MAJOR>.<MINOR>.x``.
#. Once the PR is merged into ``main``, automation opens a backport PR targeting
   ``branch/<MAJOR>.<MINOR>.x``.
#. Before merging into the release branch, the backport PR must pass CI when
   built against the *tip* of the target release branch (no stale PRs allowed).

Backport Criteria
----------------

Use the questions below to decide if a change is worth backporting:

- Does this fix a correctness bug (wrong result, UB, memory safety, data race, deadlock)?
- Does this fix a crash?
- Does this fix a regression?
- How likely are users to be affected by this? How many?
- Is there a reasonable workaround?
- How risky is this change?

Examples *not* worth backporting:

- Fixes to tests that do not impact the functionality of the library.
- Fixes to infrastructure that do not impact the functionality of the library.
