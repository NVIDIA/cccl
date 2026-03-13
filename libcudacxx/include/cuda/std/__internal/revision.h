//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_REVISION_H
#define _CUDA_STD___INTERNAL_REVISION_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// This module allows us to create revisioned symbols in the current ABI namespace. To create a revisioned symbol, the
// symbol must be forward declared as:
//
// _CCCL_DECLARE_ABI_REV(template (class T) class my_class, N);
//
// first, where N is the revision number. Then when the symbol is being defined, wrap the synbol name by _CCCL_REV macro
// as:
//
// template <class T>
// class _CCCL_REV(my_class) { ... };
//
// _CCCL_REV automatically selects the right revision namespace to define the symbol in. _CCCL_DECLARE_ABI_REV imports
// the symbol to the current namespace, too.

#define _CCCL_REV_NS(_NAME) __##_NAME##_rev_ns
#define _CCCL_REV(_NAME)    _CCCL_REV_NS(_NAME)::_NAME

#define _CCCL_DECLARE_ABI_REV_UNPACK_class         class
#define _CCCL_DECLARE_ABI_REV_UNPACK_struct        struct
#define _CCCL_DECLARE_ABI_REV_UNPACK_template(...) template <__VA_ARGS__>

#define _CCCL_DECLARE_ABI_REV_EXTRACT_NAME_class
#define _CCCL_DECLARE_ABI_REV_EXTRACT_NAME_struct
#define _CCCL_DECLARE_ABI_REV_EXTRACT_NAME_template(...) _CCCL_DECLARE_ABI_REV_EXTRACT_NAME_,

#define _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL4(_X)      _CCCL_REV_NS(_X)
#define _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL3(_X)      _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL4(_X)
#define _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL2(_X, ...) _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL3(_X##__VA_ARGS__)
#define _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL1(_X)      _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL2(_X)
#define _CCCL_DECLARE_ABI_REV_MAKE_NS(_X)            _CCCL_DECLARE_ABI_REV_MAKE_NS_IMPL1(_CCCL_DECLARE_ABI_REV_EXTRACT_NAME_##_X)

#define _CCCL_DECLARE_ABI_REV(_TYPE, _REV) \
  inline namespace __r##_REV               \
  {                                        \
    _CCCL_DECLARE_ABI_REV_UNPACK_##_TYPE;  \
  }                                        \
  namespace _CCCL_DECLARE_ABI_REV_MAKE_NS(_TYPE) = __r##_REV;

#endif // _CUDA_STD___INTERNAL_REVISION_H
