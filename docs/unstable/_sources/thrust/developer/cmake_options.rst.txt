.. _cmake-options:

Developer CMake Options
=======================

.. important::

    This document details the CMake options used for **developer builds**
    of the Thrust tests and examples included in the CCCL repository.
    The options presented here are not available when using Thrust as an
    end-user via our installed CMake packages, or when using
    ``add_subdirectory`` / CPM to add CCCL/Thrust to your project's build system.
    For details and examples of using Thrust as an end-user, please refer
    to the `Thrust + CMake example <https://github.com/NVIDIA/cccl/tree/main/examples/thrust_flexible_device_system>`_


A Thrust build is configured using CMake options. These may be passed to
CMake using

.. code:: bash

   cmake -D<option_name>=<value> /path/to/thrust/sources

or configured interactively with the ``ccmake`` or ``cmake-gui``
interfaces.

Thrust supports two build modes. By default, a single configuration is
built that targets a specific host system, device system, and C++
dialect. When ``THRUST_ENABLE_MULTICONFIG`` is ``ON``, multiple
configurations targeting a variety of systems and dialects are
generated.

The CMake options are divided into these categories:

1. `Generic CMake Options <#generic-cmake-options>`__: Options
   applicable to all Thrust builds.
2. `Single Config CMake Options <#single-config-cmake-options>`__
   Options applicable only when ``THRUST_ENABLE_MULTICONFIG`` is
   disabled.
3. `Multi Config CMake Options <#multi-config-cmake-options>`__ Options
   applicable only when ``THRUST_ENABLE_MULTICONFIG`` is enabled.
4. `CUDA Specific CMake Options <#cuda-specific-cmake-options>`__
   Options that control CUDA compilation. Only available when one or
   more configurations targets the CUDA system.
5. `TBB Specific CMake Options <#tbb-specific-cmake-options>`__
   Options that control TBB compilation. Only available when one or
   more configurations targets the TBB system.

Generic CMake Options
---------------------

-  ``CMAKE_BUILD_TYPE={Release, Debug, RelWithDebInfo, MinSizeRel}``

   -  Standard CMake build option. Default: ``RelWithDebInfo``

-  ``THRUST_ENABLE_HEADER_TESTING={ON, OFF}``

   -  Whether to test compile public headers. Default is ``ON``.

-  ``THRUST_ENABLE_TESTING={ON, OFF}``

   -  Whether to build unit tests. Default is ``ON``.

-  ``THRUST_ENABLE_EXAMPLES={ON, OFF}``

   -  Whether to build examples. Default is ``ON``.

-  ``THRUST_ENABLE_MULTICONFIG={ON, OFF}``

   -  Toggles single-config and multi-config modes. Default is ``OFF``
      (single config).

-  ``THRUST_ENABLE_EXAMPLE_FILECHECK={ON, OFF}``

   -  Enable validation of example outputs using the LLVM FileCheck
      utility. Default is ``OFF``.

-  ``THRUST_ENABLE_INSTALL_RULES={ON, OFF}``

   -  If true, installation rules will be generated for thrust. Default
      is ``ON``.

-  ``THRUST_DISPATCH_TYPE={Dynamic, Force32bit, Force64bit}``

   -  Allows the user to force Thrust to use a specific size for the offset type. Default
      is ``Dynamic``.

      -  ``Dynamic`` lets Thrust choose the index type based on input size, allowing
         large inputs and optimal performance at the cost of increased compile time and binary size,
         as Thrust will compile each kernel twice, once for 32 bit and once for 64 bit.
      -  ``Force32bit`` forces Thrust to use a 32 bit offset type. This improves compile time and
         binary size but limits the input size.
      -  ``Force64bit`` forces Thrust to use a 64 bit offset type. This improves compile time and
         binary size and allows large input sizes. However, it might degrade runtime performance.

Single Config CMake Options
---------------------------

-  ``THRUST_HOST_SYSTEM={CPP, TBB, OMP}``

   -  Selects the host system. Default: ``CPP``

-  ``THRUST_DEVICE_SYSTEM={CUDA, TBB, OMP, CPP}``

   -  Selects the device system. Default: ``CUDA``

-  ``THRUST_CPP_DIALECT={17, 20}``

   -  Selects the C++ standard dialect to use. Default is ``14``
      (C++14).

.. _cmake-multi-config-options:

Multi Config CMake Options
--------------------------

-  ``THRUST_MULTICONFIG_ENABLE_DIALECT_CPPXX={ON, OFF}``

   -  Toggle whether a specific C++ dialect will be targeted.
   -  Possible values of ``XX`` are ``{17, 20}``.
   -  By default, only C++14 is enabled.

-  ``THRUST_MULTICONFIG_ENABLE_SYSTEM_XXXX={ON, OFF}``

   -  Toggle whether a specific system will be targeted.
   -  Possible values of ``XXXX`` are ``{CPP, CUDA, TBB, OMP}``
   -  By default, only ``CPP`` and ``CUDA`` are enabled.

-  ``THRUST_MULTICONFIG_WORKLOAD={SMALL, MEDIUM, LARGE, FULL}``

   -  Restricts the host/device combinations that will be targeted.
   -  By default, the ``SMALL`` workload is used.
   -  The full cross product of ``host x device`` systems results in 12
      configurations, some of which are more important than others. This
      option can be used to prune some of the less important ones.
   -  ``SMALL``: (3 configs) Minimal coverage and validation of each
      device system against the ``CPP`` host.
   -  ``MEDIUM``: (6 configs) Cheap extended coverage.
   -  ``LARGE``: (8 configs) Expensive extended coverage. Includes all
      useful build configurations.
   -  ``FULL``: (12 configs) The complete cross product of all possible
      build configurations.

======== =========== ========== ========= ============================
Config   Workloads   Value      Expense   Note
======== =========== ========== ========= ============================
CPP/CUDA ``F L M S`` Essential  Expensive Validates CUDA against CPP
CPP/OMP  ``F L M S`` Essential  Cheap     Validates OMP against CPP
CPP/TBB  ``F L M S`` Essential  Cheap     Validates TBB against CPP
CPP/CPP  ``F L M``   Important  Cheap     Tests CPP as device
OMP/OMP  ``F L M``   Important  Cheap     Tests OMP as host
TBB/TBB  ``F L M``   Important  Cheap     Tests TBB as host
TBB/CUDA ``F L``     Important  Expensive Validates TBB/CUDA interop
OMP/CUDA ``F L``     Important  Expensive Validates OMP/CUDA interop
TBB/OMP  ``F``       Not useful Cheap     Mixes CPU-parallel systems
OMP/TBB  ``F``       Not useful Cheap     Mixes CPU-parallel systems
TBB/CPP  ``F``       Not Useful Cheap     Parallel host, serial device
OMP/CPP  ``F``       Not Useful Cheap     Parallel host, serial device
======== =========== ========== ========= ============================

CUDA Specific CMake Options
---------------------------

-  ``THRUST_ENABLE_RDC_TESTS={ON, OFF}``

   -  Enable tests that require separable compilation.
   -  Default is ``ON``.

-  ``THRUST_FORCE_RDC={ON, OFF}``

   -  Enable separable compilation on all targets that are agnostic of
      RDC.
   -  Targets that explicitly require RDC to be enabled or disabled will
      ignore this setting.
   -  Default is ``OFF``.

TBB Specific CMake Options
--------------------------

When using TBB as a host or device system, Thrust will automatically find and
link against an installed Intel TBB. The TBB installation is discovered through
CMake's standard ``find_package(TBB)`` mechanism. No additional Thrust-specific
options are provided for TBB configuration.
