.. _infra-ci-skip-tags:

Skip tags
=========

Skip tags scope a pull-request CI run. Place a tag in the last commit message before pushing.
The next PR run reads the last commit message and filters job groups accordingly. Combine tags
with the override matrix in ``ci/matrix.yaml`` for finer control.

All ``[skip-*]`` and ``[bench-only]`` tags block merge while present. Remove them from the last
commit before merging.

Tag reference
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Tag
     - Skips
     - Blocks merge
   * - ``[skip-matrix]``
     - All build and test jobs from ``ci/matrix.yaml``.
     - Yes
   * - ``[skip-vdc]``
     - All "Validate Devcontainer" jobs.
     - Yes
   * - ``[skip-docs]``
     - The documentation verification build.
     - Yes
   * - ``[skip-tpt]`` / ``[skip-third-party-testing]``
     - All third-party canary builds (MatX, PyTorch, RAPIDS).
     - Yes
   * - ``[skip-rapids]``
     - RAPIDS canary builds only.
     - Yes
   * - ``[skip-matx]``
     - MatX canary builds only.
     - Yes
   * - ``[skip-pytorch]``
     - PyTorch canary builds only.
     - Yes
   * - ``[bench-only]``
     - Equivalent to ``[skip-matrix][skip-vdc][skip-docs][skip-tpt]``.
     - Yes

``[skip-tpt]`` and ``[skip-third-party-testing]`` are aliases for the same tag.

``[bench-only]`` shorthand
--------------------------

``[bench-only]`` expands to ``[skip-matrix][skip-vdc][skip-docs][skip-tpt]``. It skips all
non-benchmark job groups. Benchmarks are triggered separately by modifying ``ci/bench.yaml``
relative to ``ci/bench.template.yaml``.

Usage
-----

Place one or more tags in the last commit message before pushing::

    git commit -m "Run PR benchmarks [bench-only]"
    git commit -m "README tidy-up [skip-matrix][skip-vdc][skip-docs][skip-third-party-testing]"

Tags act on the last commit message of the branch. To change which tags apply, amend the last
commit and re-push.

Remove all skip tags from the last commit before merging. The full CI suite must run on the final
commit for the PR to land.
