.. _infra-ci-override-matrix:

Custom matrices and the PR override
===================================

The authoritative matrix is ``ci/matrix.yaml`` on CCCL's ``ci`` branch.
There are two ways to run a focused subset, with deliberately different scopes:

* The source branch's ``custom`` workflow reads a candidate matrix from
  ``ci/matrix.yaml`` on a ref in the caller repository. It runs that matrix's
  ``nightly`` definition as additional, on-demand validation.
* The shared ``workflows.override`` definition on ``@ci`` replaces the normal
  ``pull_request`` matrix for every public source PR. It is reserved for
  coordinated changes to rolling CI state.

For ordinary iteration, use a custom matrix. Do not put a source-PR-specific
override on the shared ``ci`` branch.

Run a candidate matrix
----------------------

Create a branch in the repository that will dispatch the workflow. The branch
must contain the candidate file at exactly ``ci/matrix.yaml``; callers select a
ref, not an arbitrary path. The branch may live in public ``NVIDIA/cccl`` or in
the internal CI companion.

**Step 1. Focus the candidate's nightly definition.** Copy the combinations you
want to test into ``workflows.nightly`` in the candidate ``ci/matrix.yaml``.
Entries use the same syntax as the rolling matrix:

.. code-block:: yaml

   workflows:
     nightly:
       - {jobs: ['test'], project: 'thrust', std: 'max', ctk: '<ctk>', cxx: '<compiler>', gpu: '<gpu>'}

Useful fields include:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``jobs``
     - Job types to run. A ``test`` entry generates any build jobs it depends on.
   * - ``project``
     - Project to build or test (``thrust``, ``cub``, ``libcudacxx``, ``cudax``, ...).
   * - ``std``
     - C++ standard. ``max`` selects the highest standard the combination supports.
   * - ``ctk``
     - CUDA Toolkit version. A ``<major>.X`` suffix selects the newest image for that major version.
   * - ``cxx``
     - Host compiler. An array expands to several jobs.
   * - ``gpu``
     - GPU runner model. Required for GPU test jobs.

Field defaults and the complete vocabulary live in the matrix's ``tags`` and
``jobs`` sections.

For a tight single-target run, use ``project: 'target'`` and forward ``args``
to ``../cccl-ci/ci/util/build_and_test_targets.sh``:

.. code-block:: yaml

   workflows:
     nightly:
       - {jobs: ['run_gpu'], project: 'target', ctk: '<ctk>', cxx: '<compiler>', gpu: '<gpu>',
          args: '--preset <preset> --build-targets "<target>" --ctest-targets "<target>"'}

The ``run_cpu`` and ``run_gpu`` jobs map directly to
``build_and_test_targets.sh``. Its options are covered in
:doc:`/cccl/development/build_and_bisect_tools`.

**Step 2. Push and dispatch.** Open **Actions → custom → Run workflow** on the
source branch whose SHA should be tested, and pass the candidate branch as
``matrix_branch``. The wrapper calls the reusable implementation on ``@ci`` and
passes the candidate branch as ``matrix_ref``. The implementation always reads
``ci/matrix.yaml`` from that ref.

**Step 3. Inspect the requested jobs.** Confirm the compiler, CTK, GPU, and
arguments in the generated jobs and iterate on the candidate matrix as needed.
This validates matrix data against the current ``@ci`` implementation; it does
not dynamically load workflows or actions from the candidate branch.

The rolling PR override
-----------------------

The ``workflows.override`` list in ``@ci:ci/matrix.yaml`` is still understood by
the public pull-request workflow. When non-empty, it replaces
``workflows.pull_request`` for all public source PRs, and the aggregate ``CI``
job remains non-mergeable until the override is empty again.

Because ``@ci`` is shared across every source and release branch, use this only
as a coordinated rolling-CI operation. Keep ``override: []`` in ordinary CI
changes, and restore it immediately after any intentional use.
