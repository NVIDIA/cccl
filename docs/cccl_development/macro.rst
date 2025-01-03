.. _cccl-development-module-macros:

CCCL Internal Macros
====================

The document describes the main *internal* macros used by CCCL. They are not intended to be used by end users, but for development of CCCL features only.

----

Compiler Macros
---------------

**Host compiler macros**:

+---------------------------+-------------------------+
| ``_CCCL_COMPILER(MSVC)``  | Microsoft Visual Studio |
+---------------------------+-------------------------+
| ``_CCCL_COMPILER(CLANG)`` | Clang                   |
+---------------------------+-------------------------+
| ``_CCCL_COMPILER(GCC)``   | GCC                     |
+---------------------------+-------------------------+
| ``_CCCL_COMPILER(NVHPC)`` | Nvidia HPC compiler     |
+---------------------------+-------------------------+

The ``_CCCL_COMPILER`` function-like macro can also be used to check the version of a compiler.

.. code:: cpp

   _CCCL_COMPILER(MSVC, <, 19, 24)
   _CCCL_COMPILER(GCC, >=, 9)

*Pitfalls*: ``_CCCL_COMPILER(GCC, >, 9)`` internally expands ``_CCCL_COMPILER(GCC, >, 9, 0)`` to matches any GCC 9.x. Avoid using ``>`` and rather use ``>=``

**CUDA compiler macros**:

+--------------------------------+-------------------------+
| ``_CCCL_CUDA_COMPILER(NVCC)``  | Nvidia compiler         |
+--------------------------------+-------------------------+
| ``_CCCL_CUDA_COMPILER(NVHPC)`` | Nvidia HPC compiler     |
+--------------------------------+-------------------------+
| ``_CCCL_CUDA_COMPILER(NVRTC)`` | Nvidia Runtime Compiler |
+--------------------------------+-------------------------+
| ``_CCCL_CUDA_COMPILER(CLANG)`` | Clang                   |
+--------------------------------+-------------------------+

The ``_CCCL_CUDA_COMPILER`` function-like macro can also be used to check the version of a compiler.

.. code:: cpp

   _CCCL_CUDA_COMPILER(NVCC, <, 12, 3)
   _CCCL_CUDA_COMPILER(CLANG, >=, 14)

**CUDA identification/version macros**:

+----------------------------------+-----------------------------+
| ``_CCCL_HAS_CUDA_COMPILER``      | CUDA compiler is available  |
+----------------------------------+-----------------------------+
| ``_CCCL_CUDACC_BELOW(12, 7)``    | CUDA version below 12.7     |
+----------------------------------+-----------------------------+
| ``_CCCL_CUDACC_AT_LEAST(12, 7)`` | CUDA version at least 12.7  |
+----------------------------------+-----------------------------+

----

Architecture Macros
-------------------

The following macros are used to check the target architecture. They comply with the compiler supported by the CUDA toolkit. Compilers outside the CUDA toolkit may define such macros in a different way.

+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(ARM64)``   |  ARM 64-bit                         |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(X86)``     |  X86 both 32 and 64 bit             |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(X86_64)``  |  X86 64-bit                         |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(X86_32)``  |  X86 32-bit                         |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(64BIT)``   |  Any 64-bit OS                      |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(32BIT)``   |  Any 32-bit OS                      |
+-------------------------+-------------------------------------+

----

OS Macros
---------

+-----------------------+---------+
| ``_CCCL_OS(WINDOWS)`` | Windows |
+-----------------------+---------+
| ``_CCCL_OS(LINUX)``   | Linux   |
+-----------------------+---------+
| ``_CCCL_OS(ANDROID)`` | Android |
+-----------------------+---------+
| ``_CCCL_OS(QNX)``     | QNX     |
+-----------------------+---------+

----

CUDA Extension Macros
---------------------

**Execution space**:

+-----------------------+-----------------------+
| ``_CCCL_HOST``        | Host function         |
+-----------------------+-----------------------+
| ``_CCCL_DEVICE``      | Device function       |
+-----------------------+-----------------------+
| ``_CCCL_HOST_DEVICE`` | Host/Device function  |
+-----------------------+-----------------------+

**Other CUDA attributes**:

+------------------------------+----------------------------------------------------------+
| ``_CCCL_GRID_CONSTANT``      | Grid constant kernel parameter                           |
+------------------------------+----------------------------------------------------------+
| ``_CCCL_GLOBAL_CONSTANT``    | Host/device global scope constant (``inline constexpr``) |
+------------------------------+----------------------------------------------------------+
| ``_CCCL_EXEC_CHECK_DISABLE`` | Disable execution space check for the NVHPC compiler     |
+------------------------------+----------------------------------------------------------+

----

C++ Language Macros
-------------------

The following macros are required only if the target C++ version does not support the corresponding attribute

+-----------------------------+----------------------------------------------------------+
| ``_CCCL_STD_VER``           | C++ standard version, e.g. ``#if _CCCL_STD_VER >= 2017`` |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_IF_CONSTEXPR``      | Portable ``if constexpr`` (before C++17)                 |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX14``   | Enable ``constexpr`` for C++14 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX17``   | Enable ``constexpr`` for C++17 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX20``   | Enable ``constexpr`` for C++20 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX23``   | Enable ``constexpr`` for C++23 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_INLINE_VAR``        | Portable ``inline constexpr`` variable (before C++17)    |
+-----------------------------+----------------------------------------------------------+

**Portable feature testing**:

+--------------------------+--------------------------------------------------+
| ``_CCCL_HAS_BUILTIN(X)`` |  Portable ``__has_builtin(X)``                   |
+--------------------------+--------------------------------------------------+
| ``_CCCL_HAS_FEATURE(X)`` |  Portable ``__has_feature(X)``                   |
+--------------------------+--------------------------------------------------+
| ``_CCCL_HAS_INCLUDE(X)`` |  Portable ``__has_include(X)`` (before C++17)    |
+--------------------------+--------------------------------------------------+

**Portable attributes**:

+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_FALLTHROUGH()``          | Portable ``[[fallthrough]]`` attribute (before C++17)                        |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_NO_UNIQUE_ADDRESS``      | Portable ``[[no_unique_address]]`` attribute                                 |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_NODISCARD``              | Portable ``[[nodiscard]]`` attribute (before C++17)                          |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_NODISCARD_FRIEND``       | Portable ``[[nodiscard]]`` attribute for ``friend`` functions (before C++17) |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_NORETURN``               | Portable ``[[noreturn]]`` attribute (before C++11)                           |
+----------------------------------+------------------------------------------------------------------------------+
| ``CCCL_DEPRECATED``              | Portable ``[[deprecated]]`` attribute (before C++14)                         |
+----------------------------------+------------------------------------------------------------------------------+
| ``CCCL_DEPRECATED_BECAUSE(MSG)`` | Portable ``[[deprecated]]`` attribute with custom message (before C++14)     |
+----------------------------------+------------------------------------------------------------------------------+

----

Portable Compiler Builtin and Keyword Macros
--------------------------------------------

+-----------------------------+-------------------------------------+
| ``_CCCL_UNREACHABLE()``     | Portable ``__builtin_unreachable``  |
+-----------------------------+-------------------------------------+
| ``_CCCL_RESTRICT``          | Portable ``restrict`` keyword       |
+-----------------------------+-------------------------------------+
| ``_CCCL_BUILTIN_ASSUME(X)`` | Portable ``__builtin_assume(X)``    |
+-----------------------------+-------------------------------------+

----

Visibility Macros
-----------------

+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_HIDE_FROM_ABI``       | Hidden visibility (i.e. ``inline``, not exported, not instantiated)                                 |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_LIBCUDACXX_HIDE_FROM_ABI`` | Host/device function with hidden visibility. Most libcu++ functions are hidden with this attribute  |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
