.. _cccl-macro:

CCCL Internal Macros
====================

The document describes the main *internal* macros used by CCCL. They are not intended to be used by end users.

Compiler Macros
---------------

**Host compiler macros**:

- ``_CCCL_COMPILER(MSVC)``:  Microsoft Visual Studio
- ``_CCCL_COMPILER(CLANG)``: Clang
- ``_CCCL_COMPILER(GCC)``:   GCC
- ``_CCCL_COMPILER(NVHPC)``: Nvidia HPC compiler

The ``_CCCL_COMPILER`` function-like macro can also be used to check the version of a compiler.

.. code:: cpp

   _CCCL_COMPILER(MSVC, <, 19, 24)
   _CCCL_COMPILER(GCC, >=, 9)

*Pitfalls*: ``_CCCL_COMPILER(GCC, >, 9)`` internally expands ``_CCCL_COMPILER(GCC, >, 9, 0)`` to matches any GCC 9.x. Avoid using `>` and rather use `>=`

**CUDA compiler macros**:

- ``_CCCL_CUDA_COMPILER(NVCC)``:  Nvidia compiler
- ``_CCCL_CUDA_COMPILER(NVHPC)``: Nvidia HPC compiler
- ``_CCCL_CUDA_COMPILER(NVRTC)``: Nvidia Runtime Compiler
- ``_CCCL_CUDA_COMPILER(CLANG)``: Clang

**CUDA identification/version macros**:

- ``_CCCL_HAS_CUDA_COMPILER``:      CUDA compiler is available
- ``_CCCL_CUDACC_BELOW(12, 7)``:    CUDA version below 12.7
- ``_CCCL_CUDACC_AT_LEAST(12, 7)``: CUDA version at least 12.7

Platform Macros
---------------

- ``_CCCL_ARCH(ARM64)``:  ARM 64-bit
- ``_CCCL_ARCH(X86)``:    X86 both 32 and 64 bit
- ``_CCCL_ARCH(X86_64)``: X86 64-bit
- ``_CCCL_ARCH(X86_32)``: X86 64-bit
- ``_CCCL_ARCH(64BIT)``:  Any 64-bit OS (supported by CUDA)
- ``_CCCL_ARCH(32BIT)``:  Any 32-bit OS (supported by CUDA)

OS Macros
---------

- ``_CCCL_OS(WINDOWS)``:  Windows
- ``_CCCL_OS(LINUX)``:    Linux
- ``_CCCL_OS(ANDROID)``:  Android
- ``_CCCL_OS(QNX)``:      QNX

Attribute Macros
----------------

- ``_CCCL_FALLTHROUGH()``:          Portable `[[fallthrough]]` attribute
- ``_CCCL_NO_UNIQUE_ADDRESS``:      Portable `[[no_unique_address]]` attribute
- ``_CCCL_NODISCARD``:              Portable `[[nodiscard]]` attribute
- ``_CCCL_NODISCARD_FRIEND``:       Portable `[[nodiscard]]` attribute for `friend` functions
- ``_CCCL_NORETURN``:               Portable `[[noreturn]]` attribute
- ``_CCCL_RESTRICT``:               Portable `restrict` keyword
- ``CCCL_DEPRECATED``:              Portable `[[deprecated]]` attribute
- ``CCCL_DEPRECATED_BECAUSE(MSG)``: Portable `[[deprecated]]` attribute with custom message

CUDA Extension Macros
---------------------

**Execution space**:

- ``_CCCL_HOST``:               Host function
- ``_CCCL_DEVICE``:             Device function
- ``_CCCL_HOST_DEVICE``:        Host/Device function

**Other CUDA attributes**:

- ``_CCCL_GRID_CONSTANT``:      Grid constant kernel parameter
- ``_CCCL_GLOBAL_CONSTANT``:    Host/device global scope constant (inline constexpr)
- ``_CCCL_EXEC_CHECK_DISABLE``: Disable execution space check for the NVHPC compiler

C++ Language Macros
-------------------

- ``_CCCL_STD_VER``:         C++ standard version, e.g.`#if _CCCL_STD_VER >= 2017`
- ``_CCCL_IF_CONSTEXPR``:    Portable `if constexpr`
- ``_CCCL_CONSTEXPR_CXX14``: Enable `constexpr` for C++14
- ``_CCCL_CONSTEXPR_CXX17``: Enable `constexpr` for C++17
- ``_CCCL_CONSTEXPR_CXX20``: Enable `constexpr` for C++20
- ``_CCCL_CONSTEXPR_CXX23``: Enable `constexpr` for C++23
- ``_CCCL_INLINE_VAR``:      Portable `inline constexpr` variable
- ``_CCCL_HAS_BUILTIN``:     Portable `__has_builtin`
- ``_CCCL_HAS_FEATURE``:     Portable `__has_feature`

Compiler Specific Macros
------------------------

- ``_CCCL_UNREACHABLE()``:   Portable `__builtin_unreachable`

Libcu++ Specific Macros
-----------------------

- ``_LIBCUDACXX_HIDE_FROM_ABI`` Host/device function with hidden visibility
