.. _infra-ci-gha-workflows:

GitHub Actions workflows
========================

Every CCCL workflow lives in ``.github/workflows/``. They fall into five purposes:
the CI matrix, manual developer tools, the release pipeline, documentation
deployment, and repository automation. This page catalogs each workflow, then
explains the workflow-file mechanics that turn one push into hundreds of jobs.
For the high-level CI narrative — triggering, change detection, the ``ci:`` gate —
see :ref:`infra-ci-overview`.

Workflow reference
------------------

CI matrix
~~~~~~~~~

Four entry-point workflows select matrix keys and run them. All support manual
dispatch; the PR workflow also runs on push, and nightly/weekly workflows run on
schedules. Six dispatcher workflows unfold the generated job list into GitHub
matrix strategies.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``.github/workflows/ci-workflow-pull-request.yml``
     - Both
     - Runs the ``pull_request`` matrix on pushes to a ``pull-request/<N>`` branch or manual dispatch.
   * - ``.github/workflows/ci-workflow-nightly.yml``
     - Both
     - Runs the broader ``nightly`` matrix on a weekday schedule, with AI failure analysis and Slack notifications.
   * - ``.github/workflows/ci-workflow-weekly.yml``
     - Both
     - Runs the exhaustive ``weekly`` matrix on a Sunday schedule, with AI failure analysis and Slack notifications.
   * - ``.github/workflows/ci-workflow-custom.yml``
     - Manual
     - Runs the requested workflow keys in full without change detection; defaults to ``nightly``.
   * - ``.github/workflows/workflow-dispatch-standalone-group-linux.yml``
     - Automatic
     - Dispatches an array of standalone Linux jobs as a matrix.
   * - ``.github/workflows/workflow-dispatch-standalone-group-windows.yml``
     - Automatic
     - Dispatches an array of standalone Windows jobs as a matrix.
   * - ``.github/workflows/workflow-dispatch-two-stage-group-linux.yml``
     - Automatic
     - Dispatches an array of producer/consumer Linux chains as a matrix.
   * - ``.github/workflows/workflow-dispatch-two-stage-group-windows.yml``
     - Automatic
     - Dispatches an array of producer/consumer Windows chains as a matrix.
   * - ``.github/workflows/workflow-dispatch-two-stage-linux.yml``
     - Automatic
     - Executes one producer/consumer chain on Linux.
   * - ``.github/workflows/workflow-dispatch-two-stage-windows.yml``
     - Automatic
     - Executes one producer/consumer chain on Windows.

Custom workflow
^^^^^^^^^^^^^^^

The custom workflow accepts a ``workflows`` input with a space-separated list
of workflow keys from the matrix file. It defaults to ``nightly``. For example,
``nightly weekly`` runs both workflows. All selected workflows run in full,
without change detection.

.. _infra-ci-matrix-selection:

Choosing matrix files
^^^^^^^^^^^^^^^^^^^^^

All four entry points accept two optional ``workflow_dispatch`` inputs. Matrix
files must be in the same repository as the workflow run and follow the
:ref:`matrix.yaml format <infra-ci-matrix-yaml>`.

Setting either input selects a single matrix file and overrides ``CI_MATRICES``.
An omitted ref or path uses the corresponding default below.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Input
     - Meaning
     - Default when blank
   * - ``matrix_ref``
     - Commit SHA, branch, or tag containing the matrix file.
     - The commit that triggered the run (``github.sha``).
   * - ``matrix_path``
     - Path to the matrix file, relative to the repository root.
     - ``ci/matrix.yaml``.

When both inputs are blank, PR, nightly, and weekly runs default to the
``pull_request``, ``nightly``, and ``weekly`` profiles, respectively, in the
``CI_MATRICES`` repository variable. Custom runs use the default matrix.

Nightly and weekly manual dispatches also accept an optional ``ci_profile``
input to select a different profile. A blank value uses ``nightly`` or
``weekly``, respectively.

The variable maps profile names to named matrix definitions in ``path@ref``
form. Matrix names may contain only ASCII letters, digits, hyphens, and
underscores. For example:

.. code-block:: json

   {
     "nightly": {
       "extended": "ci/matrices/extended.yaml@ci-configs"
     },
     "pull_request": {
       "standard": "ci/matrix.yaml",
       "extended": "ci/matrices/extended.yaml@ci-configs"
     }
   }

An omitted path or ref uses ``ci/matrix.yaml`` or the triggering commit,
respectively. Each resolved path/ref pair must have one name within a profile;
different names for the same pair cause an error. If the variable is unset or
the selected profile defines no matrices, CI uses the single default matrix.

Selecting a matrix ref changes the configuration; the workflow's branch or tag
still selects the code to build and test. PR, nightly, and weekly runs use their
usual workflow keys; custom runs use the ``workflows`` input.

For example, run the full ``nightly`` matrix on code from ``my-feature`` using
``ci/custom-matrix.yaml`` from the ``ci-matrices`` branch:

.. code-block:: bash

   gh workflow run ci-workflow-custom.yml --ref my-feature \
     -f workflows=nightly                                  \
     -f matrix_ref=ci-matrices                             \
     -f matrix_path=ci/custom-matrix.yaml

When starting the PR workflow manually, select its ``pull-request/<N>`` branch
as the workflow ref.

Manual tools
~~~~~~~~~~~~

Contributors invoke these through ``workflow_dispatch`` inputs on the Actions tab.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``.github/workflows/bench.yml``
     - Both
     - Compares benchmark performance between two refs; also called by the PR workflow's bench dispatch.
   * - ``.github/workflows/git-bisect.yml``
     - Manual
     - Runs an automated ``git bisect`` across a commit range to locate a regression.

Documentation
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``.github/workflows/docs-deploy.yml``
     - Both
     - Builds the Sphinx docs and publishes them to GitHub Pages under a version-derived path.


CI orchestration
----------------

The build-workflow action
~~~~~~~~~~~~~~~~~~~~~~~~~

The "build workflow from matrix" job runs the ``build-workflow`` action at
``.github/actions/workflow-build/``. Its script ``build-workflow.py`` reads
each selected matrix file independently, using that file's tags, jobs, and
exclusions together with the change-detection output from ``ci/inspect_changes.py``.
The resulting jobs are merged into ``workflow/workflow.json``, combining shared
builds and tests and assigning unique job IDs across the run. That file holds
named groups — "CUB GCC", for example — each carrying a ``standalone`` array and
a ``two_stage`` array of fully expanded job specs.

A second script, ``prepare-workflow-dispatch.py``, splits ``workflow.json`` into
``dispatch.json``. This is the structure the GHA workflow is actually dispatched over.

The four-bucket split
~~~~~~~~~~~~~~~~~~~~~

``dispatch.json`` carries four buckets, keyed by operating system crossed with job
structure:

.. code-block:: text

    linux_standalone     linux_two_stage
    windows_standalone   windows_two_stage

Each bucket holds an ordered ``keys`` list of group names and a ``jobs`` map from
each key to its job specs:

.. code-block:: text

    {
      "linux_two_stage": {
        "keys": ["CUB GCC", "Thrust Clang", ...],
        "jobs": {
          "CUB GCC":      [{ "producers": [...], "consumers": [...] }],
          "Thrust Clang": [{ "producers": [...], "consumers": [...] }]
        }
      },
      "linux_standalone":   { "keys": [...], "jobs": {...} },
      "windows_two_stage":  { "keys": [...], "jobs": {...} },
      "windows_standalone": { "keys": [...], "jobs": {...} }
    }

``ci-workflow-pull-request.yml`` dispatches four parallel matrix jobs, one per
bucket. Each fans out over its bucket's ``keys`` and calls the matching dispatcher
workflow — ``workflow-dispatch-<structure>-group-<os>.yml``.

Standalone jobs
~~~~~~~~~~~~~~~

A standalone job builds and tests on one runner, exchanging no artifacts with other
jobs. The group dispatcher
(``workflow-dispatch-standalone-group-<os>.yml``) fans out over the bucket's job
array; each matrix entry runs one CI script — ``ci/build_<project>.sh`` or
``ci/test_<project>.sh`` — start to finish. See :ref:`infra-ci-scripts` for details.

Two-stage jobs
~~~~~~~~~~~~~~

A typical two-stage workflow splits a build job from one or more test jobs. Each stage
has different workloads and hardware requirements, and this division allows us to schedule
each on appropriate hardware.

The producer runs on a GPU-less, CPU-heavy runner.
It builds and uploads binaries through GHA artifacts (:ref:`infra-ci-artifacts`) or sccache.

Each consumer runs on a CPU-light, GPU-attached runner.
They fetch the pre-compiled test binaries and execute them.
