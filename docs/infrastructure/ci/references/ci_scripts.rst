.. _infra-ci-scripts:

CI scripts
==========

The ``ci/`` directory holds the build, test, benchmark, and utility scripts that CI jobs
invoke. The same scripts reproduce a CI environment locally — a failing job's log prints the
exact script and arguments it ran. Scripts are organized by role:

- ``ci/`` root — per-project build and test entry points.
- ``ci/util/`` — shared utilities, plus the artifact and workflow plumbing jobs use to pass data.
- ``ci/bench/`` — benchmark drivers.
- ``ci/windows/`` — PowerShell equivalents of the per-project scripts for Windows runners.

Per-project build and test scripts
----------------------------------

Each project has a pair of scripts following a fixed naming convention:
``ci/build_<project>.sh`` configures and builds it, ``ci/test_<project>.sh`` builds and runs
its tests. Both source ``ci/build_common.sh`` for argument parsing and CMake setup, so every
project shares one flag interface. Building tests needs no GPU; running them (usually) does.

Specialized build scripts cover configurations that fall outside the per-project pattern —
stdpar, clang-tidy, Python wheels, NVRTC, and codegen verification among them. They live
beside the per-project scripts in ``ci/`` and source the same common setup.

The flags come from ``ci/build_common.sh``: the host and CUDA compilers (``-cxx``, ``-cuda``),
the C++ standard (``-std``), target architectures (``-arch``), forwarded CMake options
(``-cmake-options``), and ``-configure`` to stop after configuration.
``PARALLEL_LEVEL`` controls build parallelism. Run a script with ``-h`` for the authoritative
flag list and current defaults.

When run locally, the test scripts will invoke the build script to ensure that the targets are
available. In CI, they may download GHA artifacts instead.

::

    ./ci/test_cub.sh  -cxx g++ -std 17 -arch "70;80;90"

For fast local iteration on a single target rather than a whole project, see
:doc:`/cccl/development/build_and_bisect_tools`.

Utility scripts: ci/util/
-------------------------

``ci/util/`` collects tooling shared across jobs: a targeted build-and-test runner
(``build_and_test_targets.sh``), automated ``git bisect`` over a build/test command, command
retry, peak-memory monitoring, and a mock job environment (``create_mock_job_env.sh``) that
lets the artifact and workflow scripts run outside GitHub Actions. Run any script with ``-h``
for its options.

Two subdirectories carry the producer/consumer plumbing for two-stage jobs:
``ci/util/artifacts/`` uploads and downloads the files passed between jobs, and
``ci/util/workflow/`` resolves producer/consumer relationships for the current run. Both are
covered at :ref:`infra-ci-artifacts`.

``ci/inspect_changes.py`` reports which projects are dirty between two refs and drives full
versus lite matrix selection; see :ref:`infra-ci-change-detection`.

Benchmark scripts: ci/bench/
----------------------------

``ci/bench/`` holds the benchmark drivers: ``bench.sh`` builds and runs the suite for a
configuration, and the ``compare_*`` scripts build two refs or two paths and diff the results.
The comparison workflow is ``.github/workflows/bench.yml``. PR request syntax lives in
``ci/bench.yaml``; ``ci/bench/README.md`` documents local usage and artifact layout.
