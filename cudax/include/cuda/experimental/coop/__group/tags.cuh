//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_TAGS_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_TAGS_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::coop
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT by_group_rank_t
{
  _CCCL_HIDE_FROM_ABI explicit by_group_rank_t() = default;
};

_CCCL_GLOBAL_CONSTANT by_group_rank_t by_group_rank{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT by_unit_count_t
{
  _CCCL_HIDE_FROM_ABI explicit by_unit_count_t() = default;
};

_CCCL_GLOBAL_CONSTANT by_unit_count_t by_unit_count{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT non_exhaustive_t
{
  _CCCL_HIDE_FROM_ABI explicit non_exhaustive_t() = default;
};

_CCCL_DEVICE constexpr non_exhaustive_t non_exhaustive;

struct _CCCL_TYPE_VISIBILITY_DEFAULT parent_group_t
{
  _CCCL_HIDE_FROM_ABI explicit parent_group_t() = default;
};

_CCCL_GLOBAL_CONSTANT parent_group_t parent_group{};
} // namespace cuda::experimental::coop

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_TAGS_CUH
