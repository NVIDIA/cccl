.. _cccl-development-module-macros:

CCCL Internal Macros
====================

The document describes the main *internal* macros used by CCCL. They are not intended to be used by end users, but for development of CCCL features only. We reserve the right to change them at any time without warning.

----

Compiler Macros
---------------

**Host compiler macros**:

+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(CLANG)``    | Clang                                       |
+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(GCC)``      | GCC                                         |
+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(NVHPC)``    | Nvidia HPC compiler                         |
+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(MSVC)``     | Microsoft Visual Studio                     |
+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(MSVC2019)`` | Microsoft Visual Studio 2019                |
+------------------------------+---------------------------------------------+
| ``_CCCL_COMPILER(MSVC2022)`` | Microsoft Visual Studio 2022                |
+------------------------------+---------------------------------------------+

The ``_CCCL_COMPILER`` function-like macro can also be used to check the version of a compiler.

.. code:: cpp

   _CCCL_COMPILER(MSVC, <, 19, 24)
   _CCCL_COMPILER(GCC, >=, 9)

*Note*: When used without specifying a minor version number, the macro will only test against
the compiler's major version number. For example, when the compiler is ``gcc-9.1``, the macro
``_CCCL_COMPILER(GCC, >, 9)`` will be ``false`` even though ``9.1`` is greater than ``9``.

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

The ``_CCCL_CUDA_COMPILER`` function-like macro can also be used to check the version of a CUDA compiler.

.. code:: cpp

   _CCCL_CUDA_COMPILER(NVCC, <, 12, 3)
   _CCCL_CUDA_COMPILER(CLANG, >=, 14)

*Note*: ``_CCCL_CUDA_COMPILER(...)`` check may result in a ``true`` value even during the compilation of a C++ source
file. Use ``_CCCL_CUDA_COMPILATION()`` to check for the compilation of a CUDA source file.

**CUDA identification/version macros**:

+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_CUDA_COMPILER()``    | CUDA compiler is available                                                                     |
+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_CUDA_COMPILATION()``     | CUDA code is being compiled                                                                    |
+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_HOST_COMPILATION()``     | Compiling host code, ``true`` when executing the CUDA host pass or compiling a C++ source file |
+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_DEVICE_COMPILATION()``   | Compiling device code, ``true`` when executing the CUDA device pass                            |
+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_CUDACC_BELOW(12, 7)``    | CUDA version below 12.7 when compiling a CUDA source file                                      |
+----------------------------------+------------------------------------------------------------------------------------------------+
| ``_CCCL_CUDACC_AT_LEAST(12, 7)`` | CUDA version at least 12.7 when compiling a CUDA source file                                   |
+----------------------------------+------------------------------------------------------------------------------------------------+

*Note*: When compiling CUDA code with ``nvc++`` both ``_CCCL_HOST_COMPILATION()`` and ``_CCCL_DEVICE_COMPILATION()`` result in a ``true`` value.

**PTX macros**:

+----------------------+-------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_PTX_ARCH()`` | Alias of ``__CUDA_ARCH__`` with value equal to 0 if a CUDA compiler is not available                              |
+----------------------+-------------------------------------------------------------------------------------------------------------------+
| ``__cccl_ptx_isa``   | PTX ISA version available with the current CUDA compiler, e.g. PTX ISA 8.4 (``840``) is available from CUDA 12.4  |
+----------------------+-------------------------------------------------------------------------------------------------------------------+

*Note*: When compiling CUDA code with ``nvc++`` the ``_CCCL_PTX_ARCH()`` macro expands to ``0``.

----

Architecture Macros
-------------------

The following macros are used to check the target architecture. They comply with the compiler supported by the CUDA toolkit. Compilers outside the CUDA toolkit may define such macros in a different way.

+-------------------------+---------------------------------------------------+
| ``_CCCL_ARCH(ARM64)``   |  ARM 64-bit, including MSVC emulation             |
+-------------------------+---------------------------------------------------+
| ``_CCCL_ARCH(X86_64)``  |  X86 64-bit. False on ARM 64-bit MSVC emulation   |
+-------------------------+---------------------------------------------------+

----

OS Macros
---------

+-----------------------+---------------------------------+
| ``_CCCL_OS(WINDOWS)`` | Windows, including NVRTC LLP64  |
+-----------------------+---------------------------------+
| ``_CCCL_OS(LINUX)``   | Linux, including NVRTC LP64     |
+-----------------------+---------------------------------+
| ``_CCCL_OS(ANDROID)`` | Android                         |
+-----------------------+---------------------------------+
| ``_CCCL_OS(QNX)``     | QNX                             |
+-----------------------+---------------------------------+

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
| ``NV_ANY_TARGET``         | Any target                                                        |
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

CUDA attributes
---------------

+------------------------------+----------------------------------------------------------+
| ``_CCCL_GRID_CONSTANT``      | Grid constant kernel parameter                           |
+------------------------------+----------------------------------------------------------+
| ``_CCCL_GLOBAL_CONSTANT``    | Host/device global scope constant (``inline constexpr``) |
+------------------------------+----------------------------------------------------------+

----

CUDA Toolkit macros
-------------------

+-------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_CTK()``           | CUDA toolkit is available if ``_CCCL_CUDA_COMPILER()`` evaluates to a ``true`` value or if ``cuda_runtime_api.h`` was found |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_CTK_BELOW``           | CUDA toolkit version below 12.7                                                                                             |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_CTK_AT_LEAST(12, 7)`` | CUDA toolkit version at least 12.7                                                                                          |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------------+

Non-standard Types Support
--------------------------

+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_INT128()``       | ``__int128`` and ``__uint128_t`` for 128-bit integer are supported and enabled                                                |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_NVFP8()``        | ``__nv_fp8_e5m2/__nv_fp8_e4m3/__nv_fp8_e8m0`` data types are supported and enabled.  Prefer over ``__CUDA_FP8_TYPES_EXIST__`` |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_NVFP16()``       | ``__half/__half2`` data types are supported and enabled. Prefer over ``__CUDA_FP16_TYPES_EXIST__``                            |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_NVBF16()``       | ``__nv_bfloat16/__nv_bfloat162`` data types are supported and enabled.  Prefer over ``__CUDA_BF16_TYPES_EXIST__``             |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ``_CCCL_HAS_FLOAT128()``     | ``__float128`` for 128-bit floating-point are supported and enabled                                                           |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

+-----------------------------------+-------------------------------------------------------------------------+
| ``CCCL_DISABLE_INT128_SUPPORT``   | Disable ``__int128/__uint128_t`` support                                |
+-----------------------------------+-------------------------------------------------------------------------+
| ``CCCL_DISABLE_NVFP8_SUPPORT``    | Disable ``__nv_fp8_e5m2/__nv_fp8_e4m3/__nv_fp8_e8m0`` support           |
+-----------------------------------+-------------------------------------------------------------------------+
| ``CCCL_DISABLE_NVFP16_SUPPORT``   | Disable ``__half/__half2`` support                                      |
+-----------------------------------+-------------------------------------------------------------------------+
| ``CCCL_DISABLE_NVBF16_SUPPORT``   | Disable ``__nv_bfloat16/__nv_bfloat162`` support                        |
+-----------------------------------+-------------------------------------------------------------------------+
| ``CCCL_DISABLE_FLOAT128_SUPPORT`` | Disable ``__float128`` support                                          |
+-----------------------------------+-------------------------------------------------------------------------+

+-----------------------------------+-------------------------------------------------------------------------+
| ``_LIBCUDACXX_HAS_NVFP16()``      | ``__half/__half2`` host/device are supported  (CUDA 12.2+)              |
+-----------------------------------+-------------------------------------------------------------------------+
| ``_LIBCUDACXX_HAS_NVBF16()``      | ``__nv_bfloat16/__nv_bfloat162`` host/device are supported (CUDA 12.2+) |
+-----------------------------------+-------------------------------------------------------------------------+

----

C++ Language Macros
-------------------

The following macros are required only if the target C++ version does not support the corresponding attribute

+-----------------------------+----------------------------------------------------------+
| ``_CCCL_STD_VER``           | C++ standard version, e.g. ``#if _CCCL_STD_VER >= 2017`` |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX20``   | Enable ``constexpr`` for C++20 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_CONSTEXPR_CXX23``   | Enable ``constexpr`` for C++23 or newer                  |
+-----------------------------+----------------------------------------------------------+
| ``_CCCL_HAS_EXCEPTIONS()``  | Features can use exceptions, e.g ``bad_optional_access`` |
+-----------------------------+----------------------------------------------------------+

**Concept-like Macros**:

+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_TEMPLATE(X)``  | ``template`` clause                                                                        |
+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_REQUIRES(X)``  | ``requires`` clause                                                                        |
+------------------------+--------------------------------------------------------------------------------------------+
| ``_CCCL_AND``          | Traits conjunction only used with ``_CCCL_REQUIRES``                                       |
+------------------------+--------------------------------------------------------------------------------------------+

Usage example:

.. code-block:: c++

    _CCCL_TEMPLATE(typename T)
    _CCCL_REQUIRES(is_integral_v<T> _CCCL_AND(sizeof(T) > 1))

.. code-block:: c++

    _CCCL_TEMPLATE(typename T)
    _CCCL_REQUIRES(is_arithmetic_v<T> _CCCL_AND (!is_integral_v<T>))


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
| ``_CCCL_ASSUME(EXPR)``           | Portable ``[[assume]]`` attribute (before C++23)                             |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_NO_UNIQUE_ADDRESS``      | Portable ``[[no_unique_address]]`` attribute                                 |
+----------------------------------+------------------------------------------------------------------------------+
| ``CCCL_DEPRECATED``              | Portable ``[[deprecated]]`` attribute (before C++14)                         |
+----------------------------------+------------------------------------------------------------------------------+
| ``CCCL_DEPRECATED_BECAUSE(MSG)`` | Portable ``[[deprecated]]`` attribute with custom message (before C++14)     |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_FORCEINLINE``            | Portable "always inline" attribute                                           |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_PURE``                   | Portable "pure" function attribute                                           |
+----------------------------------+------------------------------------------------------------------------------+
| ``_CCCL_CONST``                  | Portable "constant" function attribute                                       |
+----------------------------------+------------------------------------------------------------------------------+


**Portable Builtin Macros**:

+---------------------------------------+--------------------------------------------+
| ``_CCCL_UNREACHABLE()``               | Portable ``__builtin_unreachable()``       |
+---------------------------------------+--------------------------------------------+
| ``_CCCL_BUILTIN_EXPECT(X)``           | Portable ``__builtin_expected(X)``         |
+---------------------------------------+--------------------------------------------+
| ``_CCCL_BUILTIN_PREFETCH(X[, Y, Z])`` | Portable ``__builtin_prefetch(X, Y, Z)``   |
+---------------------------------------+--------------------------------------------+

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

**Portable Pragma Macros**

+--------------------------------+-------------------------------------------+
| ``_CCCL_PRAGMA_UNROLL(N)``     | Portable ``#pragma unroll N`` pragma      |
+--------------------------------+-------------------------------------------+
| ``_CCCL_PRAGMA_UNROLL_FULL()`` | Portable ``#pragma unroll`` pragma        |
+--------------------------------+-------------------------------------------+
| ``_CCCL_PRAGMA_NOUNROLL()``    | Portable ``#pragma nounroll`` pragma      |
+--------------------------------+-------------------------------------------+

**Exception Macros**

CUDA doesn't support exceptions in device code, however, sometimes we need to write host/device functions that use exceptions on host and ``__trap()`` on device. CCCL provides a set of macros that should be used in place of the standard C++ keywords to make the code compile in both, host and device code.

+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_TRY``              | Replacement for the ``try`` keyword                               |
+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_CATCH (X)``        | Replacement for the ``catch (/*X*/)`` statement                   |
+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_CATCH_ALL``        | Replacement for the ``catch (...)`` statement                     |
+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_CATCH_FALLTHOUGH`` | End of ``try``/``catch`` block if ``_CCCL_CATCH_ALL`` is not used |
+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_THROW``            | Replacement for the ``throw /*arg*/`` expression                  |
+----------------------------+-------------------------------------------------------------------+
| ``_CCCL_RETHROW``          | Replacement for the plain ``throw`` expression                    |
+----------------------------+-------------------------------------------------------------------+

*Note*: The ``_CCCL_CATCH`` clause must always introduce a named variable, like: ``_CCCL_CATCH(const exception_type& var)``.

Example:

.. code-block:: c++

    __host__ __device__ void* alloc(cuda::std::size_t nbytes)
    {
        if (void* ptr = cuda::std::malloc(nbytes))
        {
            return ptr;
        }
        _CCCL_THROW std::bad_alloc{}; // on device calls cuda::std::terminate()
    }

    __host__ __device__ void do_something(int* buff)
    {
        _CCCL_THROW std::runtime_error{"Something went wrong"}; // on device calls cuda::std::terminate()
    }

    __host__ __device__ void fn(cuda::std::size_t n)
    {
        int* buff{};

        _CCCL_TRY
        {
            buff = reinterpret_cast<int*>(alloc(n * sizeof(int)));

            do_something(buff);
        }
        _CCCL_CATCH ([[maybe_unused]] const std::bad_alloc& e) // must be always named
        {
            std::fprintf(stderr, "Failed to allocate memory\n"); // We can directly call host-only functions
            cuda::std::terminate();
        }
        _CCCL_CATCH_ALL // or _CCCL_CATCH_FALLTHOUGH
        {
            cuda::std::free(buff);
            _CCCL_RETHROW;
        }
    }

    __global__ void kernel()
    {
        fn(10);
    }

    int main()
    {
        fn(10);
        return 0;
    }

----

Visibility Macros
-----------------

+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_VISIBILITY_HIDDEN``   | Hidden visibility attribute (e.g. ``__attribute__((visibility("hidden")))``)                        |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_HIDE_FROM_ABI``       | Hidden visibility (i.e. ``inline``, not exported, not instantiated)                                 |
+-------------------------------+-----------------------------------------------------------------------------------------------------+
| ``_CCCL_API``                 | Host/device function with hidden visibility. Most CCCL functions are hidden with this attribute     |
+-------------------------------+-----------------------------------------------------------------------------------------------------+

----

Other Common Macros
-------------------

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

**Compiler-specific Suppression Macros**:

+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_CLANG(X)``    | Suppress clang warning, e.g. ``"-Wattributes"``             |
+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_GCC(X)``      | Suppress gcc warning, e.g. ``"-Wattributes"``               |
+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_NVHPC(X)``    | Suppress nvhpc warning, e.g. ``expr_has_no_effect``         |
+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_DIAG_SUPPRESS_MSVC(X)``     | Suppress msvc warning, e.g. ``4127``                        |
+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_BEGIN_NV_DIAG_SUPPRESS(X)`` | Start to suppress nvcc warning, e.g. ``177``                |
+-------------------------------------+-------------------------------------------------------------+
| ``_CCCL_END_NV_DIAG_SUPPRESS()``    | End to suppress nvcc warning                                |
+-------------------------------------+-------------------------------------------------------------+

Usage example:

.. code-block:: c++

    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
    // code ..
    _CCCL_DIAG_POP
