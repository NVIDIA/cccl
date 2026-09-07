.. _contributors-index:

Contributor Docs
================

.. toctree::
   :maxdepth: 1

   how_tos/index
   code_of_conduct
   license

Thank you for your interest in contributing to the CUDA Core Compute Libraries (CCCL)!
This section covers the branching, build, test, debug, pull request, and review workflows a CCCL contributor uses day to day.

Looking for ideas for your first contribution? Check out the
`good first issue <https://github.com/nvidia/cccl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_ label.


Getting Started
----------------

#. **Fork & Clone the Repository**:

   Fork the `CCCL GitHub Repository <https://github.com/nvidia/cccl>`_ and clone the fork. For more
   information, check `GitHub's documentation on forking
   <https://docs.github.com/en/github/getting-started-with-github/fork-a-repo>`_ and `cloning a
   repository <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`_.

#. **Set up Development Environment**:

   CCCL uses Development Containers to provide a consistent development environment for both local
   development and CI. Contributors are strongly encouraged to use these containers as they simplify
   environment setup. See the `Dev Containers guide
   <https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md>`_ for instructions on how to
   quickly get up and running using dev containers with or without VSCode.

Making Changes
----------------

#. **Create a New Branch**:

   .. code-block:: bash

      git checkout -b your-feature-branch

#. **Make Changes**.

#. **Build and Test**:

   Ensure changes don't break existing functionality by building and running tests.

   .. code-block:: bash

      ./ci/build_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>

      # test implies build
      ./ci/test_[thrust|cub|libcudacxx].sh  -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>

   For more details on building and testing, refer to the `Building and Testing`_ section below.

#. **Commit Changes**:

   .. code-block:: bash

      git commit -m "Brief description of the change"

   If you are a member of the NVIDIA GitHub enterprise, please sign your commits.

Developer Guides
~~~~~~~~~~~~~~~~~

For more information about architecture, design, and development practices, consult the following developer guides:

- :doc:`CCCL Coding Guidelines </cccl/development/coding_guidelines>` - Our coding guidelines, must be followed
- :doc:`CCCL Development Guide </cccl/development/index>` - Internal details and development process shared across CCCL libraries, mostly libcudacxx
- :doc:`Thrust Systems </thrust/developer/systems>` - Overview of Thrust's backend systems and execution policies
- :doc:`Thrust Developer CMake Options </thrust/developer/cmake_options>` - CMake options for Thrust development builds
- :doc:`CUB Developer Guide </cub/developer_overview>` - General overview of the design of CUB internals
- :doc:`CUB Tests </cub/developer/test_overview>` - Overview of how to write CUB unit tests
- :doc:`CUB Benchmarks </cub/benchmarking>` - Overview of CUB's performance benchmarks
- :doc:`CUB Tunings </cub/tuning>` - Overview of CUB's performance tuning infrastructure

Building and Testing
----------------------

CCCL components are header-only libraries. This means there isn't a traditional build process for the
library itself. However, before submitting contributions, it's a good idea to build and run tests.

There are multiple options for building and running our tests, and which one you should reach for
depends on your goal (fixing a single test vs. reproducing a full CI job) and whether you are using a
:ref:`Dev Container <infra-devcontainer-launching>` (highly recommended!). See
:ref:`infra-install-build-test` for the full breakdown of available tools.

Manual build scripts
~~~~~~~~~~~~~~~~~~~~~~

``ci/build_<project>.sh`` and ``ci/test_<project>.sh`` build or test a whole project (``thrust``,
``cub``, or ``libcudacxx``) for a given host compiler, C++ standard, and GPU architecture set. These are
the scripts our CI runs, so they reproduce a CI job exactly:

.. code-block:: bash

   ./ci/build_cub.sh -cxx g++ -std 17 -arch "70;75;80-virtual"
   ./ci/test_cub.sh  -cxx g++ -std 17 -arch "70;75;80-virtual"

Building tests does not require a GPU; running them does. See :ref:`infra-install-build-test` for the
full script reference, including the faster, target-scoped ``ci/util/build_and_test_targets.sh`` for
iterating on a single test, and :ref:`infra-cmake-architecture-flags` for the ``-arch`` value syntax.

Using CMake Presets
~~~~~~~~~~~~~~~~~~~~~

CCCL also ships `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_ for
configuring, building, and testing directly with ``cmake``/``ctest``. See
:ref:`infra-cmake-preset-reference` for the full preset reference, including how to list presets and how
preset build output is laid out on disk.

See :doc:`how_tos/vscode_cmake_tools` for the recommended, GUI-driven way to use CMake Presets from VS
Code when working inside a Dev Container.

Pre-commit hooks (code formatting, etc.)
----------------------------------------

CCCL uses `pre-commit <https://pre-commit.com/>`_ to execute all code linters and formatters. These
tools ensure a consistent coding style throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

The linters used by CCCL are listed in ``.pre-commit-config.yaml``. For example, C++ and CUDA code is
formatted with `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.

To enable the use of ``pre-commit``, install via ``conda`` or ``pip``:

.. code-block:: bash

   conda config --add channels conda-forge
   conda install pre-commit

.. code-block:: bash

   pip install pre-commit

Then run pre-commit hooks before committing code:

.. code-block:: bash

   pre-commit run

By default, pre-commit runs on staged files (only changes and additions that will be committed). To run
pre-commit checks on all files, execute:

.. code-block:: bash

   pre-commit run --all-files

It is recommended to set up the pre-commit hooks to run automatically when you make a git commit. This
can be done by running:

.. code-block:: bash

   pre-commit install

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with ``git commit --no-verify`` or with the short version ``git commit -n``.

Secret Scanning
~~~~~~~~~~~~~~~~~

The ``secret-scan-trufflehog`` pre-commit hook scans staged files and installs TruffleHog on first run
(use Git Bash on Windows). If it flags a secret, remove it before committing, or contact a maintainer if
it's a false positive. Secrets are also scanned server-side in CI on ``main``.

Creating a Pull Request
--------------------------

Push the local branch with your changes to your fork on GitHub:

.. code-block:: bash

   git push origin your-feature-branch

See :doc:`how_tos/creating_a_pull_request` for further instructions.

Review Process
-----------------

Once submitted, maintainers will be automatically assigned to review the pull request. They might
suggest changes or improvements. Constructive feedback is a part of the collaborative process, aimed at
ensuring the highest quality code.

For constructive feedback and effective communication during reviews, we recommend following
`Conventional Comments <https://conventionalcomments.org/>`_.

Further recommended reading for successful PR reviews:

- `How to Do Code Reviews Like a Human (Part One) <https://mtlynch.io/human-code-reviews-1/>`_
- `How to Do Code Reviews Like a Human (Part Two) <https://mtlynch.io/human-code-reviews-2/>`_

Thank You
-----------

Your contributions enhance CCCL for the entire community. We appreciate your effort and collaboration!
