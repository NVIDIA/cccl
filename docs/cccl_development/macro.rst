.. _cccl-development-module-macros:

CCCL Internal Macros
====================

The document describes the main *internal* macros used by CCCL. They are not intended to be used by end users, but for development of CCCL features only. We reserve the right to change them at any time without warning.

----

Compiler Macros
---------------

**Host compiler macros**:

+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(CLANG)``    | Clang                          |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(GCC)``      | GCC                            |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(NVHPC)``    | Nvidia HPC compiler            |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(MSVC)``     | Microsoft Visual Studio        |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(MSVC2017)`` | Microsoft Visual Studio 2017   |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(MSVC2019)`` | Microsoft Visual Studio 2019   |
+------------------------------+--------------------------------+
| ``_CCCL_COMPILER(MSVC2022)`` | Microsoft Visual Studio 2022   |
+------------------------------+--------------------------------+

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

**PTX macros**:

+-------------------------+-------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_PTX_ARCH``      | Alias of ``__CUDA_ARCH__`` with value equal to 0 if cuda compiler is not available                                |
+-------------------------+-------------------------------------------------------------------------------------------------------------------+
| ``__cccl_ptx_isa``      | PTX ISA version available with the current CUDA compiler, e.g. PTX ISA 8.4 (``840``) is available from CUDA 12.4  |
+-------------------------+-------------------------------------------------------------------------------------------------------------------+

----

Architecture Macros
-------------------

The following macros are used to check the target architecture. They comply with the compiler supported by the CUDA toolkit. Compilers outside the CUDA toolkit may define such macros in a different way.

+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(ARM64)``   |  ARM 64-bit                         |
+-------------------------+-------------------------------------+
| ``_CCCL_ARCH(X86_64)``  |  X86 64-bit                         |
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

Execution Space
---------------

**Functions**

+-----------------------+-----------------------+
| ``_CCCL_HOST``        | Host function         |
+-----------------------+-----------------------+
| ``_CCCL_DEVICE``      | Device function       |
+-----------------------+-----------------------+
| ``_CCCL_HOST_DEVICE`` | Host/Device function  |
+-----------------------+-----------------------+

In addition, ``_CCCL_EXEC_CHECK_DISABLE`` disables the execution space check for the NVHPC compiler

**Target Macros**

+---------------------------------------------------------------------------------+--------------------------------------------------------------------------+
| ``NV_IF_TARGET(TARGET, (CODE))``                                                | Enable ``CODE`` only if ``TARGET`` is satisfied.                         |
+---------------------------------------------------------------------------------+--------------------------------------------------------------------------+
| ``NV_IF_ELSE_TARGET(TARGET, (IF_CODE), (ELSE_CODE))``                           | Enable ``CODE_IF`` if ``TARGET`` is satisfied, ``CODE_ELSE`` otherwise.  |
+---------------------------------------------------------------------------------+--------------------------------------------------------------------------+
| ``NV_DISPATCH_TARGET(TARGET1, (TARGET1_CODE), ..., TARGET_N, (TARGET_N_CODE))`` | Enable a single code block if any of ``TARGET_i`` is satisfied.          |
+---------------------------------------------------------------------------------+--------------------------------------------------------------------------+

Possible ``TARGET`` values:

+---------------------------+-------------------------------------------------------------------+
| ``NV_IS_HOST``            | Host-code target                                                  |
+---------------------------+-------------------------------------------------------------------+
| ``NV_IS_DEVICE``          | Device-code target                                                |
+---------------------------+-------------------------------------------------------------------+
| ``NV_PROVIDES_SM_<VER>``  | SM architecture is at least ``VER``, e.g. ``NV_PROVIDES_SM_80``   |
+---------------------------+-------------------------------------------------------------------+
| ``NV_IS_EXACTLY_SM_<NN>`` | SM architecture is exactly ``VER``, e.g. ``NV_IS_EXACTLY_SM_80``  |
+---------------------------+-------------------------------------------------------------------+

Usage example:

.. code-block:: c++

    NV_IF_TARGET(NV_IS_DEVICE,    (auto x = threadIdx.x; return x;));
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return 0;), (auto x = threadIdx.x; return x;));
    NV_DISPATCH_TARGET(NV_PROVIDES_SM_90,   (return "Hopper+";),
                       NV_IS_EXACTLY_SM_75, (return "Turing";),
                       NV_IS_HOST,          (return "Host";))

*Pitfalls*:

* All target macros generate the code in a local scope, i.e. ``{ code }``.
* ``NV_DISPATCH_TARGET`` is *NOT* a switch statement. It enables the code associated with the first condition satisfied.
* The target macros take ``code`` as an argument, so it is *not* possible to use any conditional compilation, .e.g ``#if _CCCL_STD_VER >= 20`` within a target macro

----

CUDA Extension Macros
---------------------

**CUDA attributes**:

+------------------------------+----------------------------------------------------------+
| ``_CCCL_GRID_CONSTANT``      | Grid constant kernel parameter                           |
+------------------------------+----------------------------------------------------------+
| ``_CCCL_GLOBAL_CONSTANT``    | Host/device global scope constant (``inline constexpr``) |
+------------------------------+----------------------------------------------------------+

**Extended floating-point types**:

+------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_NVFP16``         | `__half/__half2` data types are supported and enabled. Prefer over ``__CUDA_FP16_TYPES_EXIST__``                |
+------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_NVBF16``         | `__nv_bfloat16/__nv_bfloat162` data types are supported and enabled.  Prefer over ``__CUDA_BF16_TYPES_EXIST__`` |
+------------------------------+-----------------------------------------------------------------------------------------------------------------+

+------------------------------+----------------------------------------------------------------+
| ``_LIBCUDACXX_HAS_NVFP16``   | `__half/__half2` host/device support  (CUDA 12.2)              |
+------------------------------+----------------------------------------------------------------+
| ``_LIBCUDACXX_HAS_NVBF16``   | `__nv_bfloat16/__nv_bfloat162` host/device support (CUDA 12.2) |
+------------------------------+----------------------------------------------------------------+

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

**Concept-like Macros**:

+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_TEMPLATE(X)``  | ``template`` clause                                                                        |
+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_REQUIRES(X)``  | ``requires`` clause                                                                        |
+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_TRAIT(X)``     | Selects variable template ``is_meow_v<T>`` instead of ``is_meow<T>::value`` when available |
+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_AND``          | Traits conjunction only used with ``_CCCL_REQUIRES``                                       |
+------------------------+--------------------------------------------------------------------------------------------+

Usage example:

.. code-block:: c++

    _CCCL_TEMPLATE(typename T)
    _CCCL_REQUIRES(_CCCL_TRAIT(is_integral, T) _CCCL_AND(sizeof(T) > 1))

.. code-block:: c++

    _CCCL_TEMPLATE(typename T)
    _CCCL_REQUIRES(_CCCL_TRAIT(is_arithmetic, T) _CCCL_AND (!_CCCL_TRAIT(is_integral, T)))


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
| ``_CCCL_FORCEINLINE``            | Portable "always inline" attribute                                           |
+----------------------------------+------------------------------------------------------------------------------+

**Portable Builtin Macros**:

+-----------------------------+--------------------------------------------+
| ``_CCCL_UNREACHABLE()``     | Portable ``__builtin_unreachable()``       |
+-----------------------------+--------------------------------------------+
| ``_CCCL_BUILTIN_ASSUME(X)`` | Portable ``__builtin_assume(X)``           |
+-----------------------------+--------------------------------------------+
| ``_CCCL_BUILTIN_EXPECT(X)`` | Portable ``__builtin_expected(X)``         |
+-----------------------------+--------------------------------------------+

**Portable Keyword Macros**

+-----------------------------+--------------------------------------------+
| ``_CCCL_RESTRICT``          | Portable ``restrict`` keyword              |
+-----------------------------+--------------------------------------------+
| ``_CCCL_ALIGNAS(X)``        | Portable ``alignas(X)`` keyword (variable) |
+-----------------------------+--------------------------------------------+
| ``_CCCL_ALIGNAS_TYPE(X)``   | Portable ``alignas(X)`` keyword (type)     |
+-----------------------------+--------------------------------------------+
| ``_CCCL_PRAGMA(X)``         | Portable ``_Pragma(X)`` keyword            |
+-----------------------------+--------------------------------------------+

----

Visibility Macros
-----------------

+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_VISIBILITY_HIDDEN``   | Hidden visibility attribute (e.g. ``__attribute__((visibility("hidden")))``)                        |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_HIDE_FROM_ABI``       | Hidden visibility (i.e. ``inline``, not exported, not instantiated)                                 |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_LIBCUDACXX_HIDE_FROM_ABI`` | Host/device function with hidden visibility. Most libcu++ functions are hidden with this attribute  |
+-------------------------------+-----------------------------------------------------------------------------------------------------+

----

Other Common Macros
-------------------

+-----------------------------+--------------------------------------------+
| ``_CUDA_VSTD``              | ``cuda::std`` namespace. To use in libcu++ |
+-----------------------------+--------------------------------------------+
| ``_CCCL_TO_STRING(X)``      | ``X`` to literal string                    |
+-----------------------------+--------------------------------------------+
| ``_CCCL_DOXYGEN_INVOKED``   | Defined during Doxygen parsing             |
+-----------------------------+--------------------------------------------+

----

Debugging Macros
----------------

+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``_CCCL_ASSERT(COND, MSG)``       | Portable CCCL assert macro. Requires (``CCCL_ENABLE_HOST_ASSERTIONS`` or ``CCCL_ENABLE_DEVICE_ASSERTIONS``) |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``_CCCL_VERIFY(COND, MSG)``       | Portable ``alignas(X)`` keyword (variable)                                                                  |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``_CCCL_ENABLE_ASSERTIONS``       | Enable assertions                                                                                           |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``CCCL_ENABLE_HOST_ASSERTIONS``   | Enable host-side assertions                                                                                 |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``CCCL_ENABLE_DEVICE_ASSERTIONS`` | Enable device-side assertions                                                                               |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``_CCCL_ENABLE_DEBUG_MODE``       | Enable debug mode (and assertions)                                                                          |
+-----------------------------------+-------------------------------------------------------------------------------------------------------------+

----

Warning Suppression Macros
--------------------------

+-----------------------------+--------------------------------------------+
| ``_CCCL_DIAG_PUSH``         | Portable ``#pragma push``                  |
+-----------------------------+--------------------------------------------+
| ``_CCCL_DIAG_POP``          | Portable ``#pragma pop``                   |
+-----------------------------+--------------------------------------------+
| ``_CCCL_PUSH_MACROS``       | Push common msvc warning suppressions      |
+-----------------------------+--------------------------------------------+
| ``_CCCL_POP_MACROS``        | Pop common msvc warning suppressions       |
+-----------------------------+--------------------------------------------+

**Compiler-specific Suppression Macros**:

+-----------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_CLANG(X)``  | Suppress clang warning, e.g. ``"-Wattributes"``             |
+-----------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_GCC(X)``    | Suppress gcc warning, e.g. ``"-Wattributes"``               |
+-----------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_NVHPC(X)``  | Suppress nvhpc warning, e.g. ``expr_has_no_effect``         |
+-----------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_MSVC(X)``   | Suppress msvc warning, e.g. ``4127``                        |
+-----------------------------------+-------------------------------------------------------------+
| ``_CCCL_NV_DIAG_SUPPRESS(X)``     | Suppress nvcc warning, e.g. ``177``                         |
+-----------------------------------+-------------------------------------------------------------+

Usage example:

.. code-block:: c++

    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
    // code ..
    _CCCL_DIAG_POP
