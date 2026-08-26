.. _infra-ci-override-matrix:

Override matrix
===============

The override matrix scopes a pull request's CI to a chosen subset of jobs.
When the ``workflows.override`` key in ``ci/matrix.yaml`` is non-empty, it
replaces the entire ``pull_request`` matrix. The PR runs only the override
jobs, and branch protection blocks the merge until the override is empty
again.

Use the override matrix to:

- Test a new compiler's nightly / weekly jobs from the PR before merging.
- Validate a compiler-specific or GPU-specific fix against one combo.
- Test CI infrastructure changes that need only a few jobs to validate.
- Debug a nightly failure by running only the combos that failed.

The override matrix is a temporary scoping tool, not a permanent matrix
edit. Every override entry must be removed before the PR lands.

Add an override entry
---------------------

The override lives at the top of ``ci/matrix.yaml`` under
``workflows.override``. The default value is empty. Override entries use the
same syntax as ``pull_request`` entries.

**Step 1. Pick the combo to test.** Identify the exact job, project,
compiler, CTK version, and GPU you need. For a compiler-specific fix, this is
one ``cxx`` value. For a nightly failure, copy the failing entry from the
``nightly`` workflow.

**Step 2. Add the entry under** ``override``. Edit ``ci/matrix.yaml`` and
add one mapping under the ``override:`` key:

.. code-block:: yaml

   workflows:
     override:
       - {jobs: ['test'], project: 'thrust', std: 'max', ctk: '<ctk>', cxx: '<compiler>', gpu: '<gpu>'}

     pull_request:
       - <...>

Choose ``ctk``, ``cxx``, and ``gpu`` values from the existing
``pull_request`` entries in ``ci/matrix.yaml``. Each field scopes the run:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``jobs``
     - Job types to run. A ``test`` entry auto-generates any build jobs it
       depends on.
   * - ``project``
     - Which project to build or test (``thrust``, ``cub``, ``libcudacxx``, ``cudax``, ...).
   * - ``std``
     - C++ standard. ``max`` selects the highest standard the combo supports.
   * - ``ctk``
     - CUDA Toolkit version. A ``<major>.X`` suffix selects the newest image for that major version.
   * - ``cxx``
     - Host compiler. A single value runs one compiler; an array expands to several jobs.
   * - ``gpu``
     - GPU runner model. Required for ``test`` jobs.

Field defaults and the full tag list live in the ``tags`` section of
``ci/matrix.yaml``.

**Step 3. Trim turnaround with targeted builds.** A full project build is
slow. To build and run a single test target instead, use ``project:
'target'`` and pass ``args`` to ``ci/util/build_and_test_targets.sh``:

.. code-block:: yaml

   workflows:
     override:
       - {jobs: ['run_gpu'], project: 'target', ctk: '<ctk>', cxx: '<compiler>', gpu: '<gpu>',
          args: '--preset <preset> --build-targets "<target>" --ctest-targets "<target>"'}

The ``run_cpu`` and ``run_gpu`` jobs map directly to
``build_and_test_targets.sh``. See that script for the available ``args``,
covered in :doc:`/cccl/development/build_and_bisect_tools`.

**Step 4. Reduce overhead further with skip tags.** Combine the override
with :ref:`[skip-*] tags <infra-ci-skip-tags>` in the last commit message to drop
devcontainer, docs, and third-party canary jobs:

.. code-block:: bash

   git commit -m "Debug <compiler> <project> failure [skip-vdc][skip-docs][skip-tpt]"

Run scoped CI and merge
-----------------------

The override is temporary: scoped jobs first, full matrix before merge.

1. **Add the override entry and push.** Only the override jobs run; the
   ``pull_request`` matrix is skipped while ``workflows.override`` is non-empty.

2. **Iterate until the override jobs pass.** Each push reruns only those
   jobs, keeping turnaround short.

3. **Empty the override and push again.** With ``workflows.override`` reset
   to empty, the full ``pull_request`` matrix runs — the suite that gates merge.

4. **Merge once the full matrix is green.**
