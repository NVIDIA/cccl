//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_HOST_COMPILER_H
#define __CCCL_HOST_COMPILER_H

#include <cuda/std/__cccl/compiler.h>

#define _CCCL_HOST_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) _CCCL_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#define _CCCL_HOST_COMPILER_NVHPC()                      _CCCL_COMPILER_NVHPC()
#define _CCCL_HOST_COMPILER_CLANG()                      _CCCL_COMPILER_CLANG()
#define _CCCL_HOST_COMPILER_GCC()                        _CCCL_COMPILER_GCC()
#define _CCCL_HOST_COMPILER_MSVC()                       _CCCL_COMPILER_MSVC()
#define _CCCL_HOST_COMPILER_MSVC2019()                   _CCCL_COMPILER_MSVC2019()
#define _CCCL_HOST_COMPILER_MSVC2022()                   _CCCL_COMPILER_MSVC2022()
#define _CCCL_HOST_COMPILER_MSVC2026()                   _CCCL_COMPILER_MSVC2026()

//! @def CCCL_HOST_COMPILER(...) /* implementation defined */
//!
//! @brief Detect the current host compiler and optionally compare its version.
//!
//! The macro supports the following forms:
//!
//! - ``CCCL_HOST_COMPILER(COMPILER)``: Detect whether ``COMPILER`` is the current host compiler.
//! - ``CCCL_HOST_COMPILER(COMPILER, OP, MAJOR)``: Compare the compiler's major version.
//! - ``CCCL_HOST_COMPILER(COMPILER, OP, MAJOR, MINOR)``: Compare the compiler's major and minor version.
//!
//! @warning When used without specifying a minor version, the macro compares only the compiler's
//! major version. For example, when the compiler is GCC 9.1, ``CCCL_HOST_COMPILER(GCC, >, 9)``
//! is ``false`` even though version 9.1 is greater than 9.
//!
//! @warning Passing any other value will result in an undefined expansion, which may or may not be
//! diagnosed by the compiler.
//! <br>
//! In addition, the macro is intended to support all host compilers that are supported by CUDA Toolkit.
//! See the Host Compiler Support Policy for
//! [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy) and
//! [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements) for
//! more details.
//!
//! @note This macro is made available when including any libcu++ header. Users that wish to
//! include the smallest possible header for this macro should include ``<cuda/std/version>``.
//!
//! For supported host compilers, the macro expands to an implementation-defined ``true`` value if the
//! current host compiler and optional version comparison match, or ``false`` otherwise. These values
//! may be used in boolean expressions (preprocessor or otherwise), but no other guarantees are made.
//!
//! Available values for ``COMPILER`` include:
//!
//! - ``NVHPC``: NVIDIA HPC C++ compiler.
//! - ``CLANG``: Clang.
//! - ``GCC``: GCC.
//! - ``MSVC``: Microsoft Visual C++.
//! - ``MSVC2019``: Microsoft Visual C++ 2019.
//! - ``MSVC2022``: Microsoft Visual C++ 2022.
//! - ``MSVC2026``: Microsoft Visual C++ 2026.
//!
//! @par Example
//! @code
//! #if CCCL_HOST_COMPILER(GCC)
//!   // GCC-only code
//! #endif
//!
//! #if CCCL_HOST_COMPILER(MSVC, >=, 19, 35)
//!   // MSVC 2019 or newer code
//! #endif
//! @endcode
//!
//! @return ``true`` if the specified host compiler and optional version comparison match, ``false`` otherwise.
#ifdef _CCCL_DOXYGEN_INVOKED
#  define CCCL_HOST_COMPILER(...) /* implementation defined */
#else
#  define CCCL_HOST_COMPILER(...) _CCCL_VERSION_COMPARE(_CCCL_HOST_COMPILER_, _CCCL_HOST_COMPILER_##__VA_ARGS__)
#endif

// The implementation is duplicated to guard against the compiler targets being accidentally
// defined by the user and to exclude NVRTC.

#endif // __CCCL_HOST_COMPILER_H
