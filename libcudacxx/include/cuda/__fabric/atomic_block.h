//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_ATOMIC_BLOCK_H
#define _CUDA___FABRIC_ATOMIC_BLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/cstring>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC
//! @brief A 16-byte-aligned, 16-byte staging block for one fabric atomic value.
//! @tparam _Tp A trivially copyable 4-, 8-, or 16-byte type supported by the atomic operation.
//! @note The value occupies the slot at endpoint_offset % 16. Result writes may overwrite the entire block.
template <class _Tp>
struct alignas(16) atomic_block
{
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>);
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8 || sizeof(_Tp) == 16);

  //! @brief Store an operand in the slot selected by the endpoint offset.
  //! @param[in] __offset Endpoint byte offset, naturally aligned for _Tp.
  //! @param[in] __value The operand value to store.
  _CCCL_DEVICE_API void store(::cuda::std::uint64_t __offset, _Tp __value) noexcept
  {
    const auto __slot = __offset % 16;
    _CCCL_ASSERT(__offset % sizeof(_Tp) == 0 && __slot + sizeof(_Tp) <= 16, "invalid atomic element offset");
    ::cuda::std::memcpy(__storage_ + __slot, &__value, sizeof(_Tp));
  }

  //! @brief Read the value in the slot selected by the endpoint offset.
  //! @param[in] __offset Endpoint byte offset, naturally aligned for _Tp.
  //! @return The value stored in that slot; an atomic result is valid only after completion.
  [[nodiscard]] _CCCL_DEVICE_API _Tp load(::cuda::std::uint64_t __offset) const noexcept
  {
    const auto __slot = __offset % 16;
    _CCCL_ASSERT(__offset % sizeof(_Tp) == 0 && __slot + sizeof(_Tp) <= 16, "invalid atomic element offset");
    _Tp __value;
    ::cuda::std::memcpy(&__value, __storage_ + __slot, sizeof(_Tp));
    return __value;
  }

private:
  unsigned char __storage_[16];
};

//! @brief A 32-byte-aligned CAS input containing adjacent compare and desired blocks.
//! @tparam _Tp A trivially copyable 4-, 8-, or 16-byte type.
//! @note Both values use the same endpoint-offset slot within their respective 16-byte blocks.
template <class _Tp>
struct alignas(32) compare_exchange_block
{
  atomic_block<_Tp> compare;
  atomic_block<_Tp> desired;
};
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_ATOMIC_BLOCK_H
