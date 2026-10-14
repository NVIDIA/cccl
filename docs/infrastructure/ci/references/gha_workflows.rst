.. _infra-ci-gha-workflows:

GitHub Actions workflows
========================

Every CCCL workflow lives in ``.github/workflows/``. They fall into five purposes:
the CI matrix, manual developer tools, the release pipeline, documentation
deployment, and repository automation. This page catalogs each workflow, then
explains the workflow-file mechanics that turn one push into hundreds of jobs.
For the high-level CI narrative — triggering, change detection, the ``ci:`` gate —
see :ref:`infra-ci-overview`.

Reusable workflows
------------------

``ci-workflow-pull-request.yml``, ``ci-workflow-nightly.yml``, and
``ci-workflow-weekly.yml`` support ``workflow_call`` alongside their push,
schedule, or manual triggers. Each defines its CI jobs, concurrency settings,
and permissions in the same file, so other workflows can call a CI profile
directly.
``ci-workflow-custom.yml`` provides manual dispatch of ``ci-matrix-build.yml``,
passing its ``workflows`` and ``matrix_branch`` inputs. The ``workflows`` input
defaults to ``nightly``. Other workflows call ``ci-matrix-build.yml`` directly.
Documentation deployment, release management, and repository maintenance
workflows define their own triggers and jobs.

Developer tools and third-party builds define their jobs in the same workflows
that expose ``workflow_dispatch`` and ``workflow_call``: ``bench.yml``,
``git-bisect.yml``, ``build-and-test-python-wheels.yml``, ``build-matx.yml``,
``build-pytorch.yml``, and ``build-rapids.yml``. They can be run manually from the
Actions tab or called directly by other workflows.

The matrix builder and helper workflows expose only ``workflow_call``.
Nested workflow calls use ``$/.github/workflows/...``
to reference implementations in the workflow's own repository and commit.

For example, nightly matrix jobs follow this chain:

.. code-block:: text

    ci-workflow-nightly.yml
      -> ci-matrix-build.yml
        -> workflow-dispatch-two-stage-group-linux.yml
          -> workflow-dispatch-two-stage-linux.yml

Callers must grant the token permissions needed by the called jobs and pass
secrets through ``secrets: inherit`` or an explicit secret mapping. CI
profiles control concurrency for their runs.

The PR workflow includes a ``CI`` job used by branch protection. It checks the
required jobs and runs even after failure or cancellation. Advisory job failures
remain outside this gate.

Workflow reference
------------------

CI matrix
~~~~~~~~~

The CI profiles select a matrix key and run it. Six reusable dispatcher workflows
unfold the generated job list into GitHub matrix strategies. Contributors rarely
invoke any of these directly; they fire on push or schedule.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``ci-workflow-pull-request.yml``
     - Automatic / reusable
     - Runs the ``pull_request`` matrix on every push to a ``pull-request/<N>`` branch.
   * - ``ci-workflow-nightly.yml``
     - Scheduled / manual / reusable
     - Runs the broader ``nightly`` matrix on a weekday schedule, with AI failure analysis and Slack notifications.
   * - ``ci-workflow-weekly.yml``
     - Scheduled / manual / reusable
     - Runs the exhaustive ``weekly`` matrix on a Sunday schedule, with AI failure analysis and Slack notifications.
   * - ``ci-workflow-custom.yml``
     - Manual
     - Runs selected workflows from a matrix on another branch in the same
       repository; defaults to ``nightly``.
   * - ``ci-matrix-build.yml``
     - Reusable
     - Builds and dispatches the selected matrix and summarizes its results.
   * - ``workflow-dispatch-standalone-group-linux.yml``
     - Reusable
     - Dispatches an array of standalone Linux jobs as a matrix.
   * - ``workflow-dispatch-standalone-group-windows.yml``
     - Reusable
     - Dispatches an array of standalone Windows jobs as a matrix.
   * - ``workflow-dispatch-two-stage-group-linux.yml``
     - Reusable
     - Dispatches an array of producer/consumer Linux chains as a matrix.
   * - ``workflow-dispatch-two-stage-group-windows.yml``
     - Reusable
     - Dispatches an array of producer/consumer Windows chains as a matrix.
   * - ``workflow-dispatch-two-stage-linux.yml``
     - Reusable
     - Executes one producer/consumer chain on Linux.
   * - ``workflow-dispatch-two-stage-windows.yml``
     - Reusable
     - Executes one producer/consumer chain on Windows.

Developer tools and third-party builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These workflows support both manual dispatch and calls from other workflows.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``bench.yml``
     - Manual / reusable
     - Compares benchmark performance between two refs; also called by the PR workflow.
   * - ``git-bisect.yml``
     - Manual / reusable
     - Runs an automated ``git bisect`` across a commit range to locate a regression.
   * - ``build-and-test-python-wheels.yml``
     - Manual / reusable
     - Builds and tests Python wheels using the ``python-wheels`` matrix.
   * - ``build-matx.yml``
     - Manual / reusable
     - Builds MatX against CCCL; also called by PR and nightly CI.
   * - ``build-pytorch.yml``
     - Manual / reusable
     - Builds PyTorch against CCCL; also called by PR and nightly CI.
   * - ``build-rapids.yml``
     - Manual / reusable
     - Builds RAPIDS against CCCL; also called by PR and nightly CI.

Documentation
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Workflow
     - Trigger
     - Purpose
   * - ``docs-deploy.yml``
     - Both
     - Builds the Sphinx docs and publishes them to GitHub Pages under a version-derived path.


CI orchestration
----------------

The build-workflow action
~~~~~~~~~~~~~~~~~~~~~~~~~

The "build workflow from matrix" job runs the ``build-workflow`` action at
``.github/actions/workflow-build/``. Its script ``build-workflow.py`` reads
``ci/matrix.yaml`` together with the change-detection output from
``ci/inspect_changes.py`` and produces ``workflow/workflow.json``. That file holds
named groups — "CUB GCC", for example — each carrying a ``standalone`` array and a
``two_stage`` array of fully expanded job specs.

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
