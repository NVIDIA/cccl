// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__cccl_config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/util_namespace.cuh>

CUB_NAMESPACE_BEGIN

template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT>
class CacheModifiedInputIterator;

namespace detail
{
template <typename Iterator>
inline constexpr bool is_CacheModifiedInputIterator = false;

template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT>
inline constexpr bool is_CacheModifiedInputIterator<CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT>> = true;
} // namespace detail

CUB_NAMESPACE_END
