.. _infra-ci-overview:

CI overview
===========

CCCL runs its build and test matrix on NVIDIA's self-hosted GitHub Actions runners.
The pipeline turns one push into hundreds of jobs across CUDA Toolkit versions,
host compilers, GPU architectures, C++ standards, and operating systems.

Source branches own the event-triggered entry workflows, tests, and
source-coupled change detection. Their thin entry workflows pass the source
commit SHA to reusable workflows on the ``ci`` branch. That rolling
branch owns the CI implementation, matrix, actions, and scripts. Locally and on
runners the two checkouts remain separate: source is ``cccl`` and CI is
``cccl-ci``; jobs mount them at ``/home/coder/cccl`` and
``/home/coder/cccl-ci``, respectively.

Triggering and the copy-pr-bot security model
---------------------------------------------

CCCL's runners have access to NVIDIA infrastructure, so arbitrary fork code must
never run on them unreviewed. The ``copy-pr-bot`` GitHub App enforces this. CI does
not trigger on ``pull_request`` events. It triggers on pushes to ``pull-request/<N>``
branches in the main repository, where ``<N>`` is the PR number. ``copy-pr-bot``
owns those branches and only writes vetted commits to them.

For external contributors, CI does not begin until a maintainer leaves an
``/ok to test [commit SHA]`` comment. ``copy-pr-bot`` verifies the SHA, copies the
approved code to the ``pull-request/<N>`` branch, and the push to that branch starts
CI. Every new commit on an external PR needs a fresh ``/ok to test``.

For NVIDIA enterprise members with signed commits, the bot establishes identity from
the commit signature and copies pushes automatically. CI begins immediately on each
push, with no comment required. See the `copy-pr-bot documentation
<https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/>`_ for signing setup.
Draft PRs do not auto-trigger — ``auto_sync_draft: false`` in
``.github/copy-pr-bot.yaml`` holds the bot back until the PR is ready.

SSH signing keys
----------------

`Signed commits <https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits>`_
are required for any internal NVIDIA contributors who want the convenience of CI running
automatically whenever a commit is pushed to a branch (i.e., doesn't require using ``/ok to test``).

This is not required for external contributions, which will always require an explicit
``/ok to test [commit SHA]`` comment from an approved account for each CI run.

To enable commit signing using your existing ssh key, set the following git options:

.. code-block:: bash

    git config --global gpg.format ssh
    git config --global user.signingKey ~/.ssh/YOUR_PUBLIC_KEY_FILE_HERE.pub

    # These settings are optional. They tell git to automatically sign all new commits and tags.
    # If these are set to false, use `git commit -S` to manually sign each commit.
    git config --global commit.gpgsign true
    git config --global tag.gpgsign true

Git is now configured to sign commits with your ssh key.

To complete the process, upload the public key to your `GitHub Signing Keys
<https://github.com/settings/keys>`_ in your browser or using the ``gh`` CLI tool:

.. code-block:: bash

    gh ssh-key add ~/.ssh/YOUR_PUBLIC_KEY_FILE_HERE.pub --type signing

Make sure that the key is uploaded to 'Signing Keys', not just 'Authentication Keys'.
The same key may be used for both.

Building the job matrix
-----------------------

``@ci:ci/matrix.yaml`` is the authoritative public matrix definition, described
in detail at :ref:`infra-ci-matrix-yaml`. The ``build-workflow`` action at
``@ci:.github/actions/workflow-build/`` reads it and produces the concrete job
list for the run. The action selects a workflow key by trigger:
``pull_request`` for PRs, ``nightly`` and ``weekly`` for scheduled runs.

Before expansion, source-owned ``ci/inspect_changes.py`` compares the PR against
its merge base and reports which projects changed, the mechanism covered at
:ref:`infra-ci-change-detection`. The action filters the ``pull_request`` matrix to the
changed projects and pulls reduced ``pull_request_lite`` entries for projects only
affected through a dependency. A source change classified as core infrastructure
marks every project dirty and runs the full matrix. The expansion mechanics — dependency
propagation, tag defaults, list explosion, ``job_map`` — are in
``@ci:ci/matrix.yaml``'s ``tags``, ``jobs``, and ``projects`` sections and the
``@ci`` ``build-workflow.py`` action.

Expansion produces ``workflow.json``: named groups (for example, "CUB GCC"), each
carrying a ``standalone`` array and a ``two_stage`` array. A second step splits these
into four buckets along two axes — operating system and structure:

.. code-block:: text

    linux_standalone     linux_two_stage
    windows_standalone   windows_two_stage

The reusable ``@ci`` pull-request workflow consumes the four buckets as four
parallel matrix dispatches, one per
``workflow-dispatch-<structure>-group-<os>.yml``. The split exists
because Linux and Windows need different runner images and shell tooling, and because
standalone and two-stage jobs have different dependency shapes. The workflow file
mechanics are covered in full at :ref:`infra-ci-gha-workflows`.

Standalone and two-stage jobs
-----------------------------

A standalone job builds and tests in one runner, sharing nothing with other jobs.

A two-stage job splits build from test into a generic single-producer multiple-consumer model.
A single producer job creates artifacts (usually on a cheap non-GPU runner), which are then
consumed in one or more test jobs, which may use GPUs if needed.

Accelerating build times with sccache
-------------------------------------

CCCL's CI uses a `heavily modified fork of sccache <https://github.com/rapidsai/sccache>`_ with
improved CUDA support to cache compiler artifacts for files that haven't changed and dramatically
accelerate build times. Local builds inside `CCCL's
Dev Containers <https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md>`_ can share the
same cache such that local builds and CI jobs mutually benefit from accelerated build times.
Follow the `GitHub Authentication
<https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md#optional-authenticate-with-github-for-sccache>`_
guide to enable this feature.

PR Branch Protections
---------------------

The reusable pull request workflow produces a single sentinel job named "CI"
that gates PR mergeability. In the source entry workflow, GitHub displays it as
``Run pull request CI / CI``.
This sentinel job depends on and checks for success of every required top-level job in the
workflow.

This test also checks for skip tags, override matrix modifications, open benchmark requests, etc.
and will block merging until the tree is restored to a mergeable state.

Override and skip enforcement
-----------------------------

A non-empty ``workflows.override`` in the shared ``@ci:ci/matrix.yaml`` replaces
the ``pull_request`` matrix globally, detailed at
:ref:`infra-ci-override-matrix`. It is reserved for coordinated rolling-CI
operations. To validate candidate matrix entries without changing every source
PR, use the source ``custom`` workflow described at
:ref:`infra-ci-adding-coverage`. The shared override deliberately blocks merge
until it is emptied.

Skip tags in the last commit message on the branch, documented at :ref:`infra-ci-skip-tags`, drop job
groups for the next run. They also block merge while present, forcing a clean final run:

.. code-block:: bash

    git commit -m "README tidy-up [skip-matrix][skip-vdc][skip-docs][skip-tpt]"
    git commit -m "Run PR benchmarks [bench-only]"

The recognized tags and their semantics are catalogued in :ref:`infra-ci-skip-tags`.
``[bench-only]`` is shorthand for the common benchmark-request combination.

Reproducing a failure locally
-----------------------------

CI jobs run build and test scripts from ``@ci:ci/`` inside the devcontainers
described in the source checkout's ``.devcontainer/README.md``. A failing job's
log names the exact container and script invocation. Pull the same container
and run the same ``/home/coder/cccl-ci/ci/build_<project>.sh`` or
``/home/coder/cccl-ci/ci/test_<project>.sh`` line to reproduce the CI
environment, as walked through at :ref:`infra-ci-reproducing-locally`. For
targeted single-test iteration,
``/home/coder/cccl-ci/ci/util/build_and_test_targets.sh`` builds and runs a
named subset, covered at :ref:`infra-ci-targeted-builds`; for a regression
hunt, see :doc:`/cccl/development/build_and_bisect_tools`.
