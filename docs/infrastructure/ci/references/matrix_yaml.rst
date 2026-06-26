.. _infra-ci-matrix-yaml:

matrix.yaml reference
=====================

``ci/matrix.yaml`` is the authoritative definition of CCCL's CI job matrix. It declares
the workflows, the toolchain and hardware vocabulary jobs draw from, and the per-workflow
job entries that expand into individual GitHub Actions jobs.

Top-level keys
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Purpose
   * - ``workflows``
     - Map of workflow name to a list of job entries. Holds ``override``, ``pull_request``,
       ``pull_request_lite``, ``nightly``, ``weekly``, etc.
   * - ``devcontainer_version``
     - Image version tag for ``rapidsai/devcontainers``.
   * - ``cuda99_gcc_version``, ``cuda99_clang_version``
     - Compiler versions used for internal cuda99.X builds.
   * - ``all_stds``
     - Every C++ standard CCCL supports (used for ``std: 'all'`` et al).
   * - ``ctk_versions``
     - Map of supported CUDA Toolkit versions to supported standards and aliases.
   * - ``device_compilers``
     - Device compiler definitions (``nvcc``, ``clang``). Selected by the ``cudacxx`` tag.
   * - ``host_compilers``
     - Host compiler definitions (``gcc``, ``clang``, ``msvc``, ``nvhpc``) and their per-version
       standards. Selected by the ``cxx`` tag.
   * - ``jobs``
     - Job type definitions: GPU requirement, dependencies, and script invocation.
   * - ``projects``
     - Project definitions: supported standards, display name, and ``job_map`` expansions.
   * - ``gpus``
     - GPU runner pools and their ``sm`` value.
   * - ``tags``
     - The fields a job entry accepts, with ``required`` flags and defaults.

``exclude`` is nested under ``workflows``. Entries matching an ``exclude`` rule are removed
from every workflow's generated matrix.

Workflow types
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Workflow
     - Purpose
   * - ``override``
     - Overrides the CI jobs run for the active PR when defined. Blocks merge while set.
   * - ``pull_request``
     - Default per-PR matrix.
   * - ``pull_request_lite``
     - Reduced matrix run when only an upstream dependency changed. (See :ref:`infra-ci-change-detection`.)
   * - ``nightly``
     - Extended scheduled matrix.
   * - ``weekly``
     - Broadest scheduled matrix, including ``sm: 'all-cccl'`` and compute-sanitizer coverage.
   * - ``python-wheels``
     - Python wheel build and test matrix.
   * - ``devcontainers``
     - Image-generation matrix. Entries catalog currently available devcontainer configs and map to no real jobs.

Job entry format
----------------

A job entry is a YAML mapping. Array-valued fields expand to the cross-product of their
elements; ``exclude`` rules are applied after expansion. The fields a user specifies are the
tags in the ``tags`` section.

Current defaults for all tags are defined in the ``tags`` section of ``ci/matrix.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Meaning
   * - ``jobs``
     - Job types to run (e.g. ``build``, ``test``, ``nvrtc``, ``verify_codegen``). Expanded by
       the project's ``job_map``. Dependencies added automatically. Required.
   * - ``project``
     - Project key from the ``projects`` section.
   * - ``ctk``
     - CUDA Toolkit version or alias from ``ctk_versions``.
   * - ``cxx``
     - Host compiler. A bare name resolves to the latest version (see below).
   * - ``cudacxx``
     - Device compiler from ``device_compilers``.
     - Rarely needed; default is ``nvcc``.
   * - ``std``
     - C++ standard. Accepts an integer or the ``all`` / ``min`` / ``max`` / ``minmax`` shortcuts.
   * - ``cpu``
     - CPU architecture (``amd64``, ``arm64``).
   * - ``gpu``
     - GPU runner type from ``gpus``.
   * - ``sm``
     - GPU architectures, ``CMAKE_CUDA_ARCHITECTURES`` syntax. ``gpu`` targets the ``gpu`` tag's SM.
       Omitting ``sm`` defers architecture selection to defaults in build scripts and ``CMakePresets.json``.
   * - ``py_version``
     - Python version for Python jobs.
   * - ``args``
     - Arguments appended to the generated command. Forwards options to
       ``ci/util/build_and_test_targets.sh`` for the ``target`` project, but works for any job.
   * - ``cmake_options``
     - Extra CMake defines, passed as ``-cmake_options "<value>"``.
   * - ``environment``
     - Environment variables injected into the job.

Computed internally, not user-specified:

- ``needs`` — defined in the ``jobs`` section. A ``test`` entry auto-generates its ``build`` producer.
- ``gpu`` requirement, ``cuda_ext``, ``name``, and ``invoke`` script details - also taken from the ``jobs`` section.
- ``force_producer_ctk`` — set in the ``jobs`` section to pin a producer build's CTK independent
  of the consumer's ``ctk`` tag. Used mainly for python packaging special cases.

Annotated example
~~~~~~~~~~~~~~~~~

::

    - {jobs: ['test'],          # build (auto-added) then test
       project: 'thrust',       # only thrust
       std: 'max',              # highest std supported by thrust, ctk, and compiler
       cxx: ['gcc', 'clang'],   # expands to two jobs, latest gcc and latest clang
       gpu: 'h100',             # test job runs on an h100 runner
       sm: 'gpu',               # build for the SM of the h100 runner (sm_90)
       cmake_options: '-DCMAKE_CUDA_FLAGS="-lineinfo"'} # customize cmake config

This expands to two ``build`` jobs — one per compiler. Thrust's ``job_map`` expands ``test``
to ``test_cpu`` and ``test_gpu``, so the two compilers produce four combined test jobs. The test
jobs request h100 runners; the auto-generated ``build`` producers run on CPU runners using
``-lineinfo`` for CUDA targets and only produce device code for the h100's SM90 arch.

In total this spawns six jobs:

- build gcc
- build clang
- test_cpu gcc
- test_cpu clang
- test_gpu gcc
- test_gpu clang

C++ standard resolution
----------------------------------------------

``std`` accepts integers or four keywords resolved against the intersection of the standards
supported by the selected CTK, host compiler, device compiler, and project:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Resolves to
   * - ``all``
     - One job per supported standard.
   * - ``min``
     - The lowest supported standard.
   * - ``max``
     - The highest supported standard.
   * - ``minmax``
     - The lowest and highest, two jobs.

Compiler version resolution
---------------------------

A bare compiler name in ``cxx`` resolves to the latest version listed for that compiler in
``host_compilers``. ``cxx: 'gcc'`` selects the highest ``gcc`` version under
``host_compilers.gcc.versions``. Pin a version by naming it: ``cxx: 'gcc13'``.

Version aliases resolve the same way. ``cxx: 'msvc2022'`` maps to the ``msvc`` version whose
``alias`` is ``2022``. CTK aliases follow the ``ctk_versions`` map: ``ctk: '13.X'`` selects the
newest CTK 13 entry, ``ctk: 'nvhpc'`` selects the CTK shipped in the current NVHPC.

Each version entry declares its own supported standards. ``std`` resolution and the
``exclude`` rules drop combinations a compiler version does not support.
