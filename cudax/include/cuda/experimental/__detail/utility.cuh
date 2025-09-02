//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_UTILITY_H
#define __CUDAX_DETAIL_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__detail/type_traits.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// NOLINTBEGIN(misc-unused-using-decls)
using ::cuda::std::declval;
// NOLINTEND(misc-unused-using-decls)

struct _CCCL_TYPE_VISIBILITY_DEFAULT no_init_t
{
  _CCCL_HIDE_FROM_ABI explicit no_init_t() = default;
};

_CCCL_GLOBAL_CONSTANT no_init_t no_init{};

using uninit_t CCCL_DEPRECATED_BECAUSE("Use cuda::experimental::no_init_t instead") = no_init_t;

// TODO: CCCL_DEPRECATED_BECAUSE("Use cuda::experimental::no_init instead")
_CCCL_GLOBAL_CONSTANT no_init_t uninit{};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_UTILITY_H
