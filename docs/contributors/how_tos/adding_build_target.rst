.. _infra-cmake-adding-build-target:

Adding a build target
=====================

In most cases, new C++ test and example sources are detected automatically.
Their executables should appear after re-configuring / re-building.
When in doubt, check the CMakeLists.txt in the source directory to discover the conventions.

This document is intended for the rare case where a developer needs to build project infrastructure from scratch.

``cccl_add_executable`` wraps ``add_executable`` with CCCL's standard target
configuration: dialect handling, output directories, metatarget registration,
and clang-tidy integration. Use it for every test, example, benchmark, and tool
in the tree. The function lives in ``cmake/CCCLAddExecutable.cmake``.

Signature
---------

::

    cccl_add_executable(target_name
      SOURCES <source1> [source2 ...]
      [ADD_CTEST]
      [NO_METATARGETS]
      [NO_CLANG_TIDY]
      [METATARGET_PATH <path>]
      [DIALECT <standard>]
    )

The first positional argument is the target name. The remaining arguments are
keyword options:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Kind
     - Effect
   * - ``SOURCES``
     - Required
     - Source files for the executable. The function hard-fails with a fatal
       error if absent.
   * - ``ADD_CTEST``
     - Flag
     - Registers a CTest with the same name as the target, running the
       executable with no arguments.
   * - ``NO_METATARGETS``
     - Flag
     - Skips metatarget registration. The target builds only by its own name.
   * - ``NO_CLANG_TIDY``
     - Flag
     - Skips clang-tidy integration for these sources.
   * - ``METATARGET_PATH``
     - One value
     - Dotted path placing the target in the metatarget hierarchy. Defaults to
       ``target_name``.
   * - ``DIALECT``
     - One value
     - C++ standard override for this target (for example ``17`` or ``20``).

Add a test target
-----------------

**Step 1. Call the function with SOURCES and ADD_CTEST.** Name the target with
its dotted hierarchy path so the metatarget system groups it correctly::

    cccl_add_executable(cub.test.device_reduce
      SOURCES test_device_reduce.cu
      ADD_CTEST
    )

This creates the executable ``cub.test.device_reduce``, registers a CTest of the
same name, and adds it to the ``cub`` and ``cub.test`` metatargets.

**Step 2. Link the target's dependencies.** ``cccl_add_executable`` configures
the target but does not link libraries. Add them after the call::

    target_link_libraries(cub.test.device_reduce
      PRIVATE
        cub.compiler_interface
        cccl.c2h
    )

Place the target in the hierarchy
----------------------------------

By default the metatarget path equals the target name. A target named
``foo.bar.baz`` builds via metatargets ``foo`` and ``foo.bar``. Running
``ninja foo`` builds every descendant; ``ninja foo.bar`` builds that subtree.

Use ``METATARGET_PATH`` to decouple the target name from its hierarchy
position. Rare; advanced use only.

Exclude a target from the hierarchy
------------------------------------

Pass ``NO_METATARGETS`` for targets that should not appear in the test
hierarchy: benchmarks, standalone tools, and anything outside the
build-everything-and-test workflow. Rare; advanced use only.
