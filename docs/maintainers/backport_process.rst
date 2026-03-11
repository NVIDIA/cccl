How to commit fixes to release branches (backport process)
==========================================================

This guide explains when and how to commit fixes to release branches.

After a release branch is created and before a release tag is finalized,
maintainers may need to apply fixes to that release branch.

To keep ``main`` as the source of truth, each fix starts as a PR against
``main``, then automation opens an equivalent *backport* PR against the
relevant release branch.


Backport Criteria
-----------------

Before starting a backport, use the questions below to decide if a change is worth backporting:

- Does this fix a correctness bug (wrong result, UB, memory safety, data race, deadlock)?
- Does this fix a crash?
- Does this fix a regression?
- Are users actively asking for this fix to be backported?
- How likely are users to be affected by this? How many?
- Is there a reasonable workaround?
- How risky is this change?

Examples *not* worth backporting:

- Fixes to tests that do not impact the functionality of the library.
- Fixes to infrastructure that does not impact the functionality of the library.

Steps
-----

#. Create a PR with the fix against ``main`` via a PR following our :doc:`contributing guidelines </cccl/contributing>`.
#. Add the label ``backport branch/X.Y.x`` to the PR.

   - If the PR is already merged, you can still trigger a backport by commenting
     ``/backport branch/X.Y.x`` on the merged PR.
#. After merge to ``main``, confirm automation opens a backport PR targeting
   ``branch/X.Y.x``.
#. Review the generated backport PR for correctness and resolve any conflicts.
#. Ensure all CI checks have passed and merge the backport PR into the target
   release branch.

   - Only members of the GitHub team `cccl-release-owners <https://github.com/orgs/NVIDIA/teams/cccl-release-owners>` can merge PRs to release branches.
