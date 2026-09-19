.. _infra-cmake-helper-modules:

CMake helper modules
====================

CCCL's ``cmake/`` directory holds the helper modules, script templates, and the vendored
CPM package manager that drive the build. Library ``CMakeLists.txt`` files include these
modules to add executables, generate header tests, expand architecture lists, wire install
rules, and fetch dependencies. Each entry covers the module's purpose and the functions or
macros it provides.

Helper modules
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - File
     - Purpose
     - Key functions / macros
   * - ``CCCLAddExecutable.cmake``
     - Add an executable with standard CCCL configuration, optional CTest registration, metatargets, and clang-tidy.
     - ``cccl_add_executable()``
   * - ``CCCLAddSubdir.cmake``
     - Pull the CCCL libraries into the build as in-tree subprojects via ``cccl_add_subdir_helper``, honoring ``CCCL_REQUIRED_COMPONENTS`` / ``CCCL_OPTIONAL_COMPONENTS``.
     - (calls ``cccl_add_subdir_helper()``)
   * - ``CCCLAddSubdirHelper.cmake``
     - Standardizes project/subproject behavior when included with ``add_subdirectory()`` by a consumer/CPM.
     - ``cccl_add_subdir_helper()``
   * - ``CCCLAddTidyTarget.cmake``
     - clang-tidy integration: global, per-subproject, and per-source analysis targets.
     - ``cccl_tidy_init()``, ``cccl_tidy_make_subproject_target()``, ``cccl_tidy_add_target()``
   * - ``CCCLBuildCompilerTargets.cmake``
     - Build the ``cccl.compiler_interface`` target carrying warning, RTTI, exception, and ptxas flags.
     - ``cccl_build_compiler_targets()``
   * - ``CCCLCheckCudaArchitectures.cmake``
     - Expand the special ``all-cccl`` and ``all-major-cccl`` values for ``CMAKE_CUDA_ARCHITECTURES`` against the current NVCC, filtered to the minimum CCCL-supported arch.
     - ``cccl_check_cuda_architectures()``
   * - ``CCCLClangdCompileInfo.cmake``
     - Enable ``CMAKE_EXPORT_COMPILE_COMMANDS`` and symlink ``compile_commands.json`` into the source tree for clangd.
     - (script; no public functions)
   * - ``CCCLConfigureTarget.cmake``
     - Apply common target properties: disable extensions, set and require the C++/CUDA standard, propagate dialect compile features, set output directories.
     - ``cccl_configure_target()``
   * - ``CCCLDevBuildChecks.cmake``
     - Enforce supported developer-build configuration: require matching ``CMAKE_CXX_STANDARD`` and ``CMAKE_CUDA_STANDARD``, default both to 17.
     - ``cccl_dev_build_checks()``
   * - ``CCCLEnsureMetaTargets.cmake``
     - Create the dot-path metatarget hierarchy so ``ninja cub.test`` builds all descendants of ``cub.test``.
     - ``cccl_ensure_metatargets()``
   * - ``CCCLGenerateHeaderTests.cmake``
     - Generate per-header compilation tests from a template to verify headers are self-contained, plus a link-check executable that catches missing ``inline`` markup.
     - ``cccl_generate_header_tests()``
   * - ``CCCLGetDependencies.cmake``
     - Fetch external and in-tree dependencies via ``find_package`` or CPM. NVBench SHA is pinned in ``CCCL_NVBENCH_SHA``.
     - ``cccl_get_<dependency>()``
   * - ``CCCLHideThirdPartyOptions.cmake``
     - Mark Catch2, CPM, FetchContent, and LLVM cache variables advanced to keep them out of the default cache view.
     - (script; ``mark_as_advanced`` only)
   * - ``CCCLInstallRules.cmake``
     - Generate header and CMake-config install rules per project, gated by a ``<project>_ENABLE_INSTALL_RULES`` cache option.
     - ``cccl_generate_install_rules()``
   * - ``CCCLTestParams.cmake``
     - Parse ``%PARAM%`` comments in test sources into the cartesian product of variant labels and preprocessor definitions. See :doc:`/cccl/development/testing` for usage.
     - ``cccl_parse_variant_params()``
   * - ``CCCLUtilities.cmake``
     - Shared utilities: non-fatal process execution, CPM-consumption compile tests, and expected-failure compile tests.
     - ``cccl_execute_non_fatal_process()``, ``cccl_add_compile_test()``, ``cccl_add_xfail_compile_target_test()``
   * - ``AppendOptionIfAvailable.cmake``
     - Append a compiler flag to a list only if a ``check_cxx_compiler_flag`` probe accepts it.
     - ``append_option_if_available()``
   * - ``CPM.cmake``
     - Vendored CPM.cmake package manager. Used by ``CCCLGetDependencies.cmake`` and by downstream consumers fetching CCCL.
     - ``CPMAddPackage()`` (third-party)

Install rule files
------------------

``cmake/install/`` holds one file per installable project. Each calls
``cccl_generate_install_rules()`` with that project's header subdirectories and packaging
options.

Adding an executable
--------------------

``cccl_add_executable()`` is the entry point most test and example ``CMakeLists.txt`` files
use. It calls ``cccl_configure_target()`` for standard properties, registers metatargets via
``cccl_ensure_metatargets()``, and adds a clang-tidy target via ``cccl_tidy_add_target()``.

::

   cccl_add_executable(cub.test.device_reduce
     SOURCES test_device_reduce.cu
     ADD_CTEST
   )

``ADD_CTEST`` registers a CTest that runs the executable with no arguments. ``NO_METATARGETS``
and ``NO_CLANG_TIDY`` opt out of those integrations. ``METATARGET_PATH`` overrides the dot-path
(default: the target name). ``DIALECT`` forces a C++ standard for this target.

Architecture-flag expansion is covered in :ref:`infra-cmake-architecture-flags`.
