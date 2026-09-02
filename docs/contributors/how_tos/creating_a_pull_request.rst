.. _contributors-creating-a-pull-request:

Creating a Pull Request
=========================

Once you pushed your local branch to your fork, you can create a pull request (PR) to CCCL's GitHub:
Refer to `GitHub's documentation
<https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
for more information on creating a pull request.

The PR description
~~~~~~~~~~~~~~~~~~~~~~~

Describe the motivation, purpose and context of the changes in the pull request description.
If your change is related to a GitHub issue, please link to it in the description.

.. TODO(bgruber): we should expand a bit here what kind of additional information we would like to see, like benchmark results, etc.

Continuous Integration (CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CCCL's CI pipeline tests across various CUDA versions, compilers, and GPU architectures. For external
contributors, the CI pipeline will not begin until a maintainer leaves an ``/ok to test`` comment. For
members of the NVIDIA GitHub enterprise, the CI pipeline will begin immediately. For a detailed overview
of CCCL's CI, see :doc:`CI overview </infrastructure/ci/references/ci_overview>`.

There is a CI check for pre-commit, called `pre-commit.ci <https://pre-commit.ci/>`_. This enforces
that all linters (such as ``clang-format``) pass. If pre-commit.ci is failing, you can comment
``pre-commit.ci autofix`` on a pull request to trigger the auto-fixer. The auto-fixer will push a commit
to your pull request that applies changes made by pre-commit hooks.

Contributors are expected to investigate failing CI jobs, fix the corresponding errors,
and work towards getting all CI runs to pass successfully.
This is not a prerequisite for getting review feedback,
but helps increase the likelihood of maintainers contributing their time.

Documentation Preview
~~~~~~~~~~~~~~~~~~~~~~~

Documentation previews allow reviewers to see how changes will appear on the live documentation site
before merging. Previews are automatically generated for all pull requests and updated with every
commit. To skip building documentation for a PR, include ``[skip-docs]`` in your commit message.

The preview URL will be posted as a comment on your PR and automatically cleaned up when the PR is
closed.

Checking for Performance Regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance stability is a key goal for CCCL, especially for ``Device*`` algorithms in CUB. When
modifying any functionality that could impact these algorithms, contributors are encouraged to verify
that no performance regressions occur.

This verification is a two-step process:

#. Determine whether your changes affect the generated SASS code (details below).
#. If the generated SASS code changes, run the benchmarks (see :doc:`CUB Benchmarks
   </cub/benchmarking>`) to quantify potential performance implications.

Automatic SASS diffing
--------------------------

The CI will automatically compare the SASS of CUB's benchmarks including the PR's changes with the ``main`` branch.
The analysis is posted as a GitHub comment.
If there are SASS changes, a diff will be provided
and a benchmark is usually necessary for each architecture and benchmark impacted by SASS changes.
If the SASS changes are trivial, a follow-up benchmark may be waived.

Manual SASS diffing
--------------------------

.. TODO(bgruber): I believe we have a script for this by now, we should refer to it in these instructions

#. Identify the ``Device*`` algorithm(s) that may be affected by the change. This isn't always
   straightforward, and you will need to confirm whether any of the CUB algorithms depend on components
   modified by your changes. If your changes affect only certain GPU architectures, make sure those
   architectures are included in the list of architectures used during compilation (for example, by
   specifying them with the ``-arch`` flag when using the build scripts, or with
   ``-DCMAKE_CUDA_ARCHITECTURES`` when building with CMake).
#. Navigate to the build directory, compile the benchmarks for the specific ``Device*`` algorithm(s)
   identified in step 1, and dump the SASS code. For example: ``ninja cub.bench.radix_sort.keys.base &&
   cuobjdump -sass ./bin/cub.bench.radix_sort.keys.base |c++filt > ./radix_sort.keys_after.sass``.
#. Check out the ``main`` branch to compare against the baseline SASS code:
   ``git checkout $(git merge-base HEAD upstream/main)``
#. Recompile and dump the SASS code emitted on the ``main`` branch. For example: ``ninja
   cub.bench.radix_sort.keys.base && cuobjdump -sass ./bin/cub.bench.radix_sort.keys.base |c++filt >
   ./radix_sort.keys_before.sass``.
#. Check whether there are differences in the generated SASS output: ``git diff --text --no-index
   --word-diff radix_sort.keys_before.sass radix_sort.keys_after.sass``

Benchmark
--------------------------

.. TODO(bgruber): We should add instructions on the CI-based benchmarking

If non-trivial SASS changes have been detected,
a performance comparison must be provided per impacted architecture and benchmark.
See :doc:`CUB Benchmarks </cub/benchmarking>` for more information on how to produce those.
