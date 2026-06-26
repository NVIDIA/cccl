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

Three trigger workflows select a matrix key and run it. Six dispatcher workflows
unfold the generated job list into GitHub matrix strategies. Contributors rarely
invoke any of these directly; they fire on push or schedule.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``.github/workflows/ci-workflow-pull-request.yml``
     - Automatic
     - Runs the ``pull_request`` matrix on every push to a ``pull-request/<N>`` branch.
   * - ``.github/workflows/ci-workflow-nightly.yml``
     - Both
     - Runs the broader ``nightly`` matrix on a weekday schedule, with Slack notifications.
   * - ``.github/workflows/ci-workflow-weekly.yml``
     - Both
     - Runs the exhaustive ``weekly`` matrix on a Sunday schedule, with Slack notifications.
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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "build workflow from matrix" job runs the ``build-workflow`` action at
``.github/actions/workflow-build/``. Its script ``build-workflow.py`` reads
``ci/matrix.yaml`` together with the change-detection output from
``ci/inspect_changes.py`` and produces ``workflow/workflow.json``. That file holds
named groups — "CUB GCC", for example — each carrying a ``standalone`` array and a
``two_stage`` array of fully expanded job specs.

A second script, ``prepare-workflow-dispatch.py``, splits ``workflow.json`` into
``dispatch.json``. This is the structure the GHA workflow is actually dispatched over.

The four-bucket split
~~~~~~~~~~~~~~~~~~~~~~~

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
