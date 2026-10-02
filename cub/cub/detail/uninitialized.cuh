// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/unit_word.cuh>

#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
//! \brief A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions. Has the
//! same size as T.
template <typename T>
struct Uninitialized
{
  /// Largest memory-access word evenly dividing T and not increasing the alignment
  using DeviceWord = typename UnitWord<T>::DeviceWord;

  static constexpr ::cuda::std::size_t DATA_SIZE = sizeof(T);
  static constexpr ::cuda::std::size_t WORD_SIZE = sizeof(DeviceWord);
  static constexpr ::cuda::std::size_t WORDS     = DATA_SIZE / WORD_SIZE;
  static_assert(DATA_SIZE % WORDS == 0);

  /// Backing storage
  alignas(T) DeviceWord storage[WORDS];

  /// Alias
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T& Alias()
  {
    return reinterpret_cast<T&>(*this);
  }
};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
