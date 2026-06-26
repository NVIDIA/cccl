.. _infra-ci-reproducing-locally:

Reproducing CI locally
======================

A failing CI job prints everything needed to reproduce it. Open the failed
job from the PR's checks list and expand the log.

On failure: reproduction block
-------------------------------

When a job exits non-zero, the runner emits an
**Instructions to Reproduce CI Failure Locally** block. It contains two steps:

**Step 1** — clone the repository at the exact SHA under test:

.. code-block:: bash

   git clone --branch <branch> --single-branch https://github.com/NVIDIA/cccl.git \
     && cd cccl && git checkout <sha>

**Step 2** — launch the same container and re-run the same command:

.. code-block:: bash

   .devcontainer/launch.sh -d -c <cuda> -H <host> -- <command>

Copy both lines verbatim. The ``-c`` and ``-H`` values are the CTK version and
host compiler for this job. ``<command>`` is the ``ci/*.sh`` invocation the
runner used, with all flags.

.. note::

   GPU test jobs omit the GPU flag from the printed command. Add ``--gpus all``
   when reproducing a test job on a machine with a GPU.

Job Inputs block
----------------

The **Job Inputs** group near the top of every log records the full job
configuration, printed before the container launches:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``Job command``
     - The ``ci/*.sh`` script and flags the runner will execute
   * - ``JOB_ID``
     - Unique job identifier; pass it to ``create_mock_job_env.sh`` to reproduce the job's
       environment locally
   * - ``JOB_CUDA``
     - CTK version (e.g. ``13.3``)
   * - ``JOB_HOST``
     - Host compiler identifier (e.g. ``gcc15``)
   * - ``JOB_IMAGE``
     - Full RAPIDS devcontainer image tag pulled for this job
   * - ``JOB_RUNNER``
     - Runner label; labels containing ``-gpu-`` mean the job used a GPU
   * - ``JOB_ENVIRONMENT``
     - Extra environment variables injected into the container

These are the same values that feed the ``launch.sh`` invocation in the
failure block.

Override matrix entry
---------------------

The **Override matrix entry** block, also printed before the container
launches, contains a YAML snippet:

.. code-block:: yaml

    - {jobs: [...], project: '...', ctk: '...', cxx: '...', ...}

Paste this into the ``workflows.override`` list in ``ci/matrix.yaml`` and push
to re-run only that specific job in CI without waiting for the full matrix.
See :ref:`infra-ci-override-matrix` for the override workflow.

Mock the job environment
------------------------

.. note::

   This step is only needed when debugging the CCCL workflow / artifact scripts (rare).
   More project contributors can safely skip this.

To run the ``ci/util/workflow/`` and ``ci/util/artifacts/`` scripts outside CI —
debugging artifact upload, download, or producer resolution locally — recreate the
job's environment with ``create_mock_job_env.sh``:

.. code-block:: bash

   ci/util/create_mock_job_env.sh <run_id> <job_id>

Run it inside a devcontainer. It takes the ``<run_id>`` and ``<job_id>`` from the job
log (``GITHUB_RUN_ID`` and ``JOB_ID``), sets the ``GITHUB_*`` variables CI exports,
sources the workflow and artifact helpers, clears stale local artifact directories, and
drops into a shell that mimics the in-container CI environment. Every job log prints the
exact command under ``Mock with:``.

Tighten the loop
----------------

The full ``ci/build_*.sh`` and ``ci/test_*.sh`` scripts build and run an
entire project's test suite. Once the container reproduces the failure, use
``ci/util/build_and_test_targets.sh`` to build and run only the failing target:

.. code-block:: bash

   .devcontainer/launch.sh --docker --cuda 13.0 --host gcc14 --gpus all -- \
     ci/util/build_and_test_targets.sh \
       --preset cub-cpp20 \
       --build-targets "cub.test.iterator" \
       --ctest-targets "cub.test.iterator"

This rebuilds one target instead of the full project. For preset, target, and
lit-test flags, and for ``ci/util/git_bisect.sh`` to find the introducing commit,
see :doc:`/cccl/development/build_and_bisect_tools`.
