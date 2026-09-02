.. _contributors-index:

Contributor Docs
================

Thank you for your interest in contributing to the CUDA Core Compute Libraries (CCCL)!
This section covers the branching, build, test, debug, pull request, and review workflows a CCCL contributor uses day to day.

Looking for ideas for your first contribution? Check out the
`good first issue <https://github.com/nvidia/cccl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_ label.


.. toctree::
   :maxdepth: 1

   code_of_conduct
   license
   how_tos/index

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

Developer Guides
~~~~~~~~~~~~~~~~~

For more information about architecture, design, and development practices, consult the following developer guides:

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
library itself. However, before submitting contributions, it's a good idea to `build and run tests`_.

There are multiple options for building and running our tests. Which option you choose depends on your
preferences and whether you are using `CCCL's DevContainers
<https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md>`_ (highly recommended!).

Using Manual Build Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building
^^^^^^^^^

Use the build scripts provided in the ``ci/`` directory to build tests for each component. Building
tests does not require a GPU.

.. code-block:: bash

   ci/build_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>

- **HOST_COMPILER**: The desired host compiler (e.g., ``g++``, ``clang++``).
- **CXX_STANDARD**: The C++ standard version (e.g., ``17``, ``20``).
- **GPU_ARCHS**: A semicolon-separated list of CUDA GPU architectures (e.g., ``"70;85;90"``). This
  uses the same syntax as CMake's `CUDA_ARCHITECTURES
  <https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES>`_:

  - ``70`` - both PTX and SASS
  - ``70-real`` - SASS only
  - ``70-virtual`` - PTX only

**Example:**

.. code-block:: bash

   ./ci/build_cub.sh -cxx g++ -std 17 -arch "70;75;80-virtual"

Testing
^^^^^^^^

Use the test scripts provided in the ``ci/`` directory to run tests for each component. These take the
same arguments as the build scripts and will automatically build the tests if they haven't already been
built. Running tests requires a GPU.

.. code-block:: bash

   ci/test_[thrust|cub|libcudacxx].sh -cxx <HOST_COMPILER> -std <CXX_STANDARD> -arch <GPU_ARCHS>

**Example:**

.. code-block:: bash

   ./ci/test_cub.sh -cxx g++ -std 17 -arch "70;75;80-virtual"

Using CMake Presets
~~~~~~~~~~~~~~~~~~~~~

`CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_ are a set of
configurations defined in a JSON file that specify project-wide build details for CMake. They provide a
standardized and sharable way to configure, build, and test projects across different platforms and
development environments. Presets are available from CMake versions 3.19 and later.

There are three kinds of Presets

- Configure Presets: specify options for the ``cmake`` command,

- Build Presets: specify options for the ``cmake --build`` command,

- Test Presets: specify options for the ``ctest`` command.

In CCCL we provide many presets to be used out of the box. You can find the complete list in our
corresponding `CMakePresets.json <https://github.com/NVIDIA/cccl/blob/main/CMakePresets.json>`_ file.

These commands can be used to get lists of the configure, build, and test presets.

.. code-block:: bash

   cmake --list-presets # Configure presets
   cmake --build --list-presets # Build presets
   ctest --list-presets # Test presets

While there is a lot of overlap, there may be differences between the configure, build, and test
presets to support various testing workflows.

The ``dev`` presets are intended as a base for general development while the others are useful for
replicating CI failures.

Using CMake Presets via Command Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMake automatically generates the preset build directories. You can configure, build and test for a
specific preset (e.g. ``thrust-cpp17``) via cmake from the root directory by appending
``--preset=thrust-cpp17`` to the corresponding commands. For example:

.. code-block:: bash

   cmake --preset=thrust-cpp17
   cmake --build --preset=thrust-cpp17
   ctest --preset=thrust-cpp17

That will create ``build/<optional devcontainer name>/thrust-cpp17/`` and build everything in there.
The devcontainer name is inserted automatically on devcontainer builds to keep build artifacts separate
for the different toolchains.

It's also worth mentioning that additional cmake options can still be passed in and will override the
preset settings.

As a common example, the presets are currently always ``60;70;80`` for ``CMAKE_CUDA_ARCHITECTURES``,
but this can be overridden at configure time with something like:

.. code-block:: bash

   cmake --preset=thrust-cpp20 "-DCMAKE_CUDA_ARCHITECTURES=89"

.. note::

   Either using the ``cmake`` command from within the root directory or from within the build directory
   works, but will behave in slightly different ways. Building and running tests from the build
   directory will compile every target and run all of the tests configured in the configure step. Doing
   so from the root directory using the ``--preset=<test_preset>`` option will build and run a subset of
   configured targets and tests.

Using CMake Presets via VS Code GUI extension (Recommended when using DevContainers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended way to use CMake Presets is via the VS Code extension `CMake Tools
<https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools>`_, already included in
`CCCL's DevContainers <https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md>`_. As soon as
you install the extension you would be able to see the sidebar menu below.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_sidebar.png
   :alt: cmaketools sidebar

You can specify the desired CMake Preset by clicking the "Select Configure Preset" button under the
"Configure" node (see image below).

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_presets.png
   :alt: cmaketools presets

After that you can select the default build target from the "Build" node. As soon as you expand it, a
list will appear with all the available targets that are included within the preset you selected. For
example if you had selected the ``all-dev`` preset VS Code will display all the available targets we
have in cccl.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_targets.png
   :alt: cmaketools targets

You can build the selected target by pressing the gear button
(|gear|) at the bottom of the VS Code window.

Alternatively you can select the desired target from either the "Debug" or "Launch" drop down menu
(for debugging or running correspondingly). In that case after you select the target and either press
"Run" (|run|) or "Debug" (|debug|) the target will build on its own before running without the user
having to build it explicitly from the gear button.

.. |gear| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/build_button.png
.. |run| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/run.png
.. |debug| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/debug.png

----

We encourage users who want to debug device code to install the `Nsight Visual Studio Code Edition
extension <https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition>`_ that
enables the VS Code frontend for ``cuda-gdb``. To use it you should launch from the sidebar menu instead
of pressing the "Debug" button from the bottom menu.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/nsight.png
   :alt: nsight

Creating a Pull Request
--------------------------

#. Push changes to your fork
#. Create a pull request targeting the ``main`` branch of the original CCCL repository. Refer to
   `GitHub's documentation
   <https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
   for more information on creating a pull request.
#. Describe the purpose and context of the changes in the pull request description.

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

#. Determine whether your changes affect the generated SASS code (details on how to do this are
   provided below).
#. If the generated SASS code changes, run the benchmarks (see :doc:`CUB Benchmarks
   </cub/benchmarking>`) to quantify potential performance implications.

**Steps to check whether your changes generate different SASS code:**

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
#. Dump the SASS code emitted on the ``main`` branch. For example: ``ninja
   cub.bench.radix_sort.keys.base && cuobjdump -sass ./bin/cub.bench.radix_sort.keys.base |c++filt >
   ./radix_sort.keys_before.sass``.
#. Check whether there are differences in the generated SASS output: ``git diff --text --no-index
   --word-diff radix_sort.keys_before.sass radix_sort.keys_after.sass``

Code Formatting (pre-commit hooks)
-------------------------------------

CCCL uses `pre-commit <https://pre-commit.com/>`_ to execute all code linters and formatters. These
tools ensure a consistent coding style throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

The linters used by CCCL are listed in ``.pre-commit-config.yaml``. For example, C++ and CUDA code is
formatted with `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.

To use ``pre-commit``, install via ``conda`` or ``pip``:

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

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This
can be done by running:

.. code-block:: bash

   pre-commit install

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with ``git commit --no-verify`` or with the short version ``git commit -n``.

Secret Scanning
------------------

The ``secret-scan-trufflehog`` pre-commit hook scans staged files and installs TruffleHog on first run
(use Git Bash on Windows). If it flags a secret, remove it before committing, or contact a maintainer if
it's a false positive. Secrets are also scanned server-side in CI on ``main``.

Continuous Integration (CI)
-------------------------------

CCCL's CI pipeline tests across various CUDA versions, compilers, and GPU architectures. For external
contributors, the CI pipeline will not begin until a maintainer leaves an ``/ok to test`` comment. For
members of the NVIDIA GitHub enterprise, the CI pipeline will begin immediately. For a detailed overview
of CCCL's CI, see :doc:`CI overview </infrastructure/ci/references/ci_overview>`.

There is a CI check for pre-commit, called `pre-commit.ci <https://pre-commit.ci/>`_. This enforces
that all linters (such as ``clang-format``) pass. If pre-commit.ci is failing, you can comment
``pre-commit.ci autofix`` on a pull request to trigger the auto-fixer. The auto-fixer will push a commit
to your pull request that applies changes made by pre-commit hooks.

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
