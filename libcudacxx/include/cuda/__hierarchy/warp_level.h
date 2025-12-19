//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_WARP_LEVEL_H
#define _CUDA___HIERARCHY_WARP_LEVEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/block_level.h>
#  include <cuda/__hierarchy/hierarchy_query_result.h>
#  include <cuda/__hierarchy/native_hierarchy_level_base.h>
#  include <cuda/__hierarchy/thread_level.h>
#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct warp_level : __native_hierarchy_level_base<warp_level>
{
  using __next_native_level = block_level;

  using __base_type = __native_hierarchy_level_base<warp_level>;
  using __base_type::extents_as;

#  if _CCCL_CUDA_COMPILATION()
  using __base_type::index_as;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<1, _Tp> extents_as(const block_level&) noexcept
  {
    return ::cuda::std::dims<1, _Tp>{static_cast<_Tp>((gpu_thread.count(block) + 31) / 32)};
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<_Tp> index_as(const block_level&) noexcept
  {
    return {static_cast<_Tp>(gpu_thread.rank(block) / 32), 0, 0};
  }
#  endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_GLOBAL_CONSTANT warp_level warp;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_WARP_LEVEL_H
