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

#  include <cuda/__fabric/common.h>
#  include <cuda/std/__floating_point/cuda_fp_types.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/cstring>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC

template <class _Tp>
struct __atomic_block_representation
{
  using type = _Tp;
};

#  if _CCCL_HAS_NVFP16()
template <>
struct __atomic_block_representation<::__half2>
{
  using type = ::__half2_raw;
};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct __atomic_block_representation<::__nv_bfloat162>
{
  using type = ::__nv_bfloat162_raw;
};
#  endif // _CCCL_HAS_NVBF16()

//! @brief A 16-byte-aligned, 16-byte staging block for one fabric atomic value.
//! @tparam _Tp A trivially copyable 4-, 8-, or 16-byte type, __half2, or __nv_bfloat162, supported by the atomic
//! operation.
//! @note The value occupies the slot at endpoint_offset % 16. Result writes may overwrite the entire block.
//! @note Exchange and compare-exchange also support 16-byte aggregate types; _Tp need not be arithmetic.
template <class _Tp>
struct alignas(__fabric_block_size) atomic_block
{
private:
  using _Representation = typename __atomic_block_representation<_Tp>::type;

public:
  static_assert(::cuda::std::is_trivially_copyable_v<_Representation>);
  static_assert(sizeof(_Representation) == sizeof(_Tp));
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8 || sizeof(_Tp) == 16);

  //! @brief Store an operand in the slot selected by the endpoint offset.
  //! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
  //! @param[in] __value The operand value to store.
  _CCCL_DEVICE_API void store(::cuda::std::uint64_t __offset, const _Tp& __value) noexcept
  {
    const auto __slot = __offset % __fabric_block_size;
    _CCCL_ASSERT(__offset % sizeof(_Tp) == 0 && __slot + sizeof(_Tp) <= __fabric_block_size,
                 "invalid atomic element offset");
    if constexpr (::cuda::std::is_same_v<_Tp, _Representation>)
    {
      ::cuda::std::memcpy(__storage_ + __slot, ::cuda::std::addressof(__value), sizeof(_Tp));
    }
    else
    {
      // CUDA packed floating-point types provide a trivially copyable raw representation.
      const _Representation __raw = __value;
      ::cuda::std::memcpy(__storage_ + __slot, ::cuda::std::addressof(__raw), sizeof(_Tp));
    }
  }

  //! @brief Read the value in the slot selected by the endpoint offset.
  //! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
  //! @return The value stored in that slot; an atomic result is valid only after completion.
  [[nodiscard]] _CCCL_DEVICE_API _Tp load(::cuda::std::uint64_t __offset) const noexcept
  {
    const auto __slot = __offset % __fabric_block_size;
    _CCCL_ASSERT(__offset % sizeof(_Tp) == 0 && __slot + sizeof(_Tp) <= __fabric_block_size,
                 "invalid atomic element offset");
    _Representation __value;
    ::cuda::std::memcpy(::cuda::std::addressof(__value), __storage_ + __slot, sizeof(_Tp));
    return static_cast<_Tp>(__value);
  }

private:
  unsigned char __storage_[__fabric_block_size];
};

//! @brief A 32-byte-aligned CAS input containing adjacent compare and desired blocks.
//! @tparam _Tp A 4-, 8-, or 16-byte type supported by atomic_block.
//! @note Both values use the same endpoint-offset slot within their respective 16-byte blocks.
template <class _Tp>
struct alignas(2 * __fabric_block_size) compare_exchange_block
{
  atomic_block<_Tp> compare;
  atomic_block<_Tp> desired;
};
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_ATOMIC_BLOCK_H
