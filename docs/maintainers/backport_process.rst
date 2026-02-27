Backport Process
================

This page documents how changes are made to release branches.

Process overview
----------------

In general, changes are made to the `main` branch and then backported via automation to release branches.

To land a fix in a release branch, follow these steps:
1. Make the fix via a normal PR against `main`.
2. If the fix meets the criteria below to include in a release branch, add a label:
   `backport branch/<MAJOR>.<MINOR>.x`
3. Once the PR is merged into `main`, automation opens a backport PR targeting `branch/<MAJOR>.<MINOR>.x`
4. Before merging into the release branch, the backport PR must pass CI when built against the *tip* of the target release branch (no stale PRs allowed)

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
- Fixes to tests that do not impact the functionality of the library
- Fixes to infrastructure that does not impact the functionality of the library
