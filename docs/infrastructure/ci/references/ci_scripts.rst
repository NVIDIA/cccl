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

Testing Python in a minimal container
-------------------------------------

``cuda.compute`` is meant to work with nothing installed beyond its declared pip
dependencies -- no host compiler and no system CUDA toolkit. The devcontainer supplies
both, so a test running there cannot distinguish "we depend only on our wheels" from "we
happened to find ``gcc`` and ``/usr/local/cuda`` lying around".

The Python test lanes therefore fetch the wheel in the devcontainer (which needs ``gh``)
and then run the test payload in a sibling container holding nothing but Python, launched
through the host's docker daemon -- by ``ci/util/python/run_in_minimal_container.sh`` on
Linux and ``ci/windows/run_in_minimal_container.ps1`` on Windows. This is the same
docker-outside-of-docker arrangement the wheel builds already use. An undeclared
dependency fails there instead of passing silently. The same applies to both the v1
(NVRTC) and v2 (HostJIT) backends.

The two platforms differ only where they must. Linux hands the sibling the specific GPUs
the driver reports, since ``--gpus all`` would reach GPUs belonging to other jobs on a
shared runner. Windows exposes GPUs as a whole device class and only under process
isolation, and its image must match the host kernel -- the devcontainer images are
LTSC 2022, so the sibling defaults to ``mcr.microsoft.com/windows/servercore:ltsc2022``.
Neither image ships the interpreter the lane asked for: ``uv`` installs that, exactly as it
does in the devcontainer. The Linux image supplies a Python only to bootstrap ``uv``;
Server Core has none at all.

Windows needs one more thing. ``python:3.14-slim`` still ships glibc and libstdc++,
because every C/C++ Python extension links against them; Server Core ships neither of
the Windows equivalents, since ``msvcp140.dll`` and ``vcruntime140*.dll`` come from the
MSVC redistributable rather than from Windows itself. Without them numba and
``cccl.c.parallel.dll`` both fail to load. The Windows payload installs the
redistributable before running anything, which leaves the comparison the lane exists to
make intact: still no compiler, still no CUDA toolkit.

Each lane is therefore two scripts: an entry point that provisions the wheel and
dispatches (``ci/test_<lane>.sh``, or ``ci/windows/test_<lane>.ps1``), and a payload that
must survive in the minimal image (``ci/util/python/run_<lane>_tests.sh``, or
``ci/windows/run_<lane>.ps1``). Set ``CCCL_MINIMAL_CONTAINER=0`` to run the payload in the
devcontainer instead, which is useful locally and for comparing the two environments.

These lanes deliberately stay in the devcontainer, because they need what it provides:

* ``py_ctk_mode: sysctk`` -- exists specifically to test against a *system-provided* CUDA
  toolkit.
* ``test_headers`` -- compiles C++, so it needs a host compiler.
* ``python_tsan`` -- ``LD_PRELOAD``\ s the runner's ``libtsan``, located via ``gcc``.
* ``test_py_stf`` -- ``cuda-stf`` is a separate wheel with its own producer and test
  script, which does not use this path.

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
