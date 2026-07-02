.. _infra-install-build-test:

Install, Build, Test
====================

CCCL provides several developer interfaces for working with the codebase.
Purpose-built scripts exist that drive routine work, and using the wrong tool can waste hours of developer time.
For example, the ``ci/test_*.sh`` scripts can take hours to run a full validation suite, while the
``ci/util/build_and_test_targets.sh`` tool configures, builds, and runs a small subset in seconds.
Prebuilt development containers simplify working in specific toolchains and environments.
Core members can launch GitHub Actions benchmarking / bisection workflows that run on cloud infrastructure.

Pick a path by goal
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 38 14 14

   * - Goal
     - Tools
     - Type
     - Availability
   * - Install CCCL headers to a prefix
     - ``ci/install_cccl.sh``
     - Script
     - Public
   * - Build and run a specific test
     - ``ci/util/build_and_test_targets.sh``
     - Script
     - Public
   * - Build or test an entire project
     - ``ci/build_<project>.sh`` / ``ci/test_<project>.sh``
     - Script
     - Public
   * - Bisect a regression
     - ``ci/util/git_bisect.sh``, git-bisect.yml
     - Script, GHA
     - Public / members
   * - Request a benchmark comparison
     - ``ci/bench/bench.sh``, bench.yml, ``ci/bench.yaml``
     - Script, GHA, PR tool
     - Public / members
   * - Custom build
     - ``cmake --preset``
     - Script
     - Public

Install only
~~~~~~~~~~~~

``ci/install_cccl.sh <prefix>`` copies CCCL's headers and CMake config files into a prefix
directory. CCCL is header-only, so the install has no build step and finishes in seconds.

Use it when a downstream project needs ``find_package(CCCL)`` against a fixed checkout, or when you
want CCCL on a system include path without cloning into the consumer's tree.

Build and run a specific test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ci/util/build_and_test_targets.sh`` configures one preset, builds the Ninja targets you name,
and runs the CTest or lit tests you name. It is the fast-iteration path for a single test or a
handful of targets.

::

    ./ci/util/build_and_test_targets.sh \
      --preset cub-cpp20 \
      --build-targets "cub.test.iterator" \
      --ctest-targets "cub.test.iterator"

Use it when you are fixing one test and want a tight edit-build-run loop. Building tests does not
require a GPU; running them does.

Need a specific CTK or host compiler? Launch the matching container with ``.devcontainer/launch.sh``
first, then run the script inside it. Valid toolchain combinations are in the
``devcontainers:`` section of ``ci/matrix.yaml``; launching is covered at
:ref:`infra-devcontainer-launching`.

Build or test an entire project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ci/build_<project>.sh`` and ``ci/test_<project>.sh`` build or test a full project across a host
compiler, C++ standard, and architecture set. These are the scripts CI runs, so they reproduce a
CI job exactly.

::

    ./ci/build_cub.sh -cxx g++ -std 17 -arch "75;80;90"
    ./ci/test_cub.sh  -cxx g++ -std 17 -arch "75;80;90"

Use them to reproduce a CI failure or to validate a project end to end before pushing. A full
project build takes hours; a targeted ``build_and_test_targets.sh`` run takes minutes. Test scripts
require a GPU.

Need a specific toolchain? Run these inside a devcontainer launched with ``.devcontainer/launch.sh``
(:ref:`infra-devcontainer-launching`). CI failure logs print the exact container and arguments to
reproduce the job.

Bisect a regression
~~~~~~~~~~~~~~~~~~~

``ci/util/git_bisect.sh`` walks the commit history between a good and a bad ref, building and
testing each candidate, until it pins the commit that introduced a regression. It takes the same
``--preset``, ``--build-targets``, and ``--ctest-targets`` arguments as
``build_and_test_targets.sh``.

::

    ./ci/util/git_bisect.sh \
      --preset cub-cpp20 \
      --build-targets "cub.test.iterator" \
      --ctest-targets "cub.test.iterator" \
      --good-ref v1.13.0 \
      --bad-ref origin/main

Use it when a test passes on an old ref and fails on ``main`` and you need the offending commit.
``--repeat N`` re-runs multiple times to help catch intermittent failures.

Need a specific toolchain? Run the script inside a devcontainer launched with
``.devcontainer/launch.sh`` (:ref:`infra-devcontainer-launching`).

Members can run the same bisect remotely on CI machines through the `Git Bisect workflow
<https://github.com/NVIDIA/cccl/actions/workflows/git-bisect.yml>`_: choose a runner, set the good
and bad refs and target arguments, and dispatch. Source is ``.github/workflows/git-bisect.yml``.

Request a benchmark comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare benchmark results between two refs from the browser, a local script, or a pull request.

Members can dispatch the `Benchmark Compare workflow
<https://github.com/NVIDIA/cccl/actions/workflows/bench.yml>`_ from the browser: choose "Run
workflow", set the base and test refs and the CUB and Python filters, and it runs on the CI GPU
pool. Source is ``.github/workflows/bench.yml``.

Locally, ``ci/bench/bench.sh <base> <test>`` runs the same comparison against checked-out refs::

    ./ci/bench/bench.sh origin/main HEAD --cub-filter "^cub\.bench\.copy\.memcpy\.base$"

It wraps ``ci/bench/compare_git_refs.sh`` and ``ci/bench/compare_paths.sh``; call those directly
when you already have two checkouts.

To benchmark inside a PR, edit ``ci/bench.yaml`` to set GPUs and filters and push; PR CI detects the
diff from ``ci/bench.template.yaml`` and dispatches the jobs. Reset ``ci/bench.yaml`` to match the
template before merging. Argument behavior and artifact layout live in ``ci/bench/README.md``.

Full CMake control
~~~~~~~~~~~~~~~~~~

``cmake --preset <name>`` configures a build directory directly.
This is discouraged, but available for custom workflows / tool integrations.

::

    cmake --preset all-dev
    cmake --build --preset all-dev
    ctest --preset all-dev

List the presets with ``cmake --list-presets``. The ``all-dev`` preset enables every library,
test, and example against your native GPU; per-library presets like ``cub-cpp20`` scope the build
to one library and standard. The :ref:`infra-cmake-preset-reference` catalogs the full preset set.
