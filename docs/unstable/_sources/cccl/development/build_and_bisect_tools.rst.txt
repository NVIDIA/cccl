.. _build-and-bisect-tools:

Build and Bisect Utilities
==========================

``build_and_test_targets.sh``
-----------------------------

:file:`ci/util/build_and_test_targets.sh` configures, builds, and tests selected
CMake targets.

Options
~~~~~~~
- ``--preset <name>`` - choose a CMake preset.
- ``--cmake-options <str>`` - extra arguments for the preset configuration.
- ``--configure-override <cmd>`` - run a custom configuration command instead of
  a preset. When used, ``--preset`` and ``--cmake-options`` are ignored.
- ``--build-targets <targets>`` - space separated Ninja targets. If omitted,
  nothing builds.
- ``--ctest-targets <regex>`` - space separated CTest ``-R`` patterns. If
  omitted, nothing runs.
- ``--lit-precompile-tests <paths>`` - space separated libcudacxx lit test paths
  to precompile (no run). Paths are relative to ``libcudacxx/test/libcudacxx/``.
- ``--lit-tests <paths>`` - space separated libcudacxx lit test paths to execute.
  Paths are relative to ``libcudacxx/test/libcudacxx/``.
- ``--custom-test-cmd <cmd>`` - arbitrary command executed after build/tests.

Combine with ``.devcontainer/launch.sh -d`` to reproduce CI commands inside a
container and choose a CUDA toolkit and host compiler:
``.devcontainer/launch.sh -d [--cuda <XX.Y>] [--host <compiler>] [--gpus all] -- <script>``

Examples
~~~~~~~~
Build a single CUB test locally::

  ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator"

Build the same test for SM90 using a CMake option::

  ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --cmake-options "-DCMAKE_CUDA_ARCHITECTURES=90" \
    --build-targets "cub.cpp20.test.iterator"

Build the test for SM90 with a configure override::

  ci/util/build_and_test_targets.sh \
    --configure-override "ci/build_cub.sh -configure -arch 90" \
    --build-targets "cub.cpp20.test.iterator"

Build **and run** a single CUB test locally::

  ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator" \
    --ctest-targets "cub.cpp20.test.iterator"

Build and run a single CUB test in a devcontainer with specific CTK and host::

  .devcontainer/launch.sh -d --cuda 12.3 --host gcc12 --gpus all -- \
    ci/util/build_and_test_targets.sh \
      --preset cub-cpp20 \
      --build-targets "cub.cpp20.test.iterator" \
      --ctest-targets "cub.cpp20.test.iterator"

Precompile the libcudacxx lit suite::

  ci/util/build_and_test_targets.sh \
    --preset libcudacxx \
    --build-targets libcudacxx.test.lit.precompile

Precompile a single libcudacxx lit test (no execution)::

  ci/util/build_and_test_targets.sh \
    --preset libcudacxx \
    --lit-precompile-tests \
      "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"

Execute one or more libcudacxx lit tests::

  ci/util/build_and_test_targets.sh \
    --preset libcudacxx \
    --lit-tests \
      "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"

``git_bisect.sh``
-----------------

:file:`ci/util/git_bisect.sh` wraps ``git bisect`` around the build/test helper.
It accepts all ``build_and_test_targets.sh`` options plus:

- ``--good-ref <rev>`` - Optional; known good commit, tag, or branch. ``-Nd`` means
  "N days ago." Defaults to the latest release version tag.
- ``--bad-ref <rev>`` - Optional; known bad commit. ``-Nd`` means "N days ago."
  Defaults to ``origin/main``.

Examples
~~~~~~~~
Local CUB bisection from latest release to origin/main::

  ci/util/git_bisect.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator" \
    --ctest-targets "cub.cpp20.test.iterator"

Devcontainer CUB bisection from last week::

  .devcontainer/launch.sh -d --cuda 12.3 --host gcc12 --gpus all -- \
    ci/util/git_bisect.sh \
      --preset cub-cpp20 \
      --build-targets "cub.cpp20.test.iterator" \
      --ctest-targets "cub.cpp20.test.iterator" \
      --good-ref -7d

Compute-sanitizer example for regression introduced between 3-4 weeks ago::

  .devcontainer/launch.sh -d --cuda 12.9 --host gcc13 --gpus all \
    --env CCCL_TEST_MODE=compute-sanitizer-initcheck \
    --env C2H_SEED_COUNT_OVERRIDE=1 \
    -- ci/util/git_bisect.sh \
      --preset "cub-cpp20" \
      --build-targets "cub.cpp20.test.iterator" \
      --ctest-targets "cub.cpp20.test.iterator" \
      --good-ref -28d \
      --bad-ref -21d

Workflow/Bisect
---------------

A ``Workflow/Bisect`` GitHub Actions job runs ``git_bisect.sh`` on a remote
runner. Launch it from **Actions → Git Bisect → Run workflow**. Provide
any desired runner label, refs, preset, targets, or launch arguments. The job
log streams bisect progress, and the run's **Summary** page renders the final
Markdown report with culprit commit, PR, reproduction steps, and more.
