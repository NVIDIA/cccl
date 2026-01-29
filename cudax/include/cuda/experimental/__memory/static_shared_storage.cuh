//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_STORAGE_H
#define _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__charconv/to_chars.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/experimental/__memory/shared_memory_ptr.cuh>

namespace cuda::experimental
{
[[nodiscard]] _CCCL_CONSTEVAL _CCCL_DEVICE_API ::cuda::std::size_t __size_t_to_str_size(::cuda::std::size_t __n) noexcept
{
  ::cuda::std::size_t __ret = 0;
  while (__n > 0)
  {
    __n /= 10;
    ++__ret;
  }
  return __ret;
}

template <::cuda::std::size_t _StrSize>
[[nodiscard]] _CCCL_CONSTEVAL _CCCL_DEVICE_API auto __size_t_to_str(::cuda::std::size_t __n) noexcept
{
  ::cuda::std::array<char, _StrSize> __ret{};
  ::cuda::std::to_chars(__ret.data(), __ret.data() + _StrSize, __n);
  return __ret;
}

template <char... _Cs>
struct __static_cstr
{
  static constexpr char __value[]{_Cs..., '\0'};
};

template <::cuda::std::size_t _Size,
          ::cuda::std::size_t _Align,
          ::cuda::std::size_t... _SizeIdx,
          ::cuda::std::size_t... _AlignIdx>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned __cccl_alloc_static_shared_impl(
  ::cuda::std::integer_sequence<::cuda::std::size_t, _SizeIdx...>,
  ::cuda::std::integer_sequence<::cuda::std::size_t, _AlignIdx...>) noexcept
{
  constexpr auto __size_str  = __size_t_to_str<sizeof...(_SizeIdx)>(_Size);
  constexpr auto __align_str = __size_t_to_str<sizeof...(_AlignIdx)>(_Align);

  using _SizeCStr  = __static_cstr<__size_str[_SizeIdx]...>;
  using _AlignCStr = __static_cstr<__align_str[_AlignIdx]...>;

  unsigned __ret;
  asm(R"({
  .shared .align %2 .b8 _cccl_static_shared_storage[%1];
  mov.b32 %0, _cccl_static_shared_storage;
  })"
      : "=r"(__ret)
      : "C"(_SizeCStr::__value), "C"(_AlignCStr::__value));
  return __ret;
}

template <::cuda::std::size_t _Size, ::cuda::std::size_t _Align>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned __cccl_alloc_static_shared() noexcept
{
  return __cccl_alloc_static_shared_impl<_Size, _Align>(
    ::cuda::std::make_index_sequence<__size_t_to_str_size(_Size)>{},
    ::cuda::std::make_index_sequence<__size_t_to_str_size(_Align)>{});
}

//! @brief Allocates static shared memory with the given size and alignment.
//!
//! @tparam _Size  Size of the storage in bytes.
//! @tparam _Align  Alignment of the storage in bytes.
template <::cuda::std::size_t _Size, ::cuda::std::size_t _Align>
class [[nodiscard]] static_shared_storage
{
  static_assert(::cuda::is_power_of_two(_Align), "_Align must be power of two");

  unsigned __smem_addr_; //!< Shared memory address of the storage.

public:
  static constexpr ::cuda::std::size_t size      = _Size; //!< Size of the storage.
  static constexpr ::cuda::std::size_t alignment = _Align; //!< Alignment of the storage.

  //! @brief Allocates the static shared memory and constructs the handle.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static_shared_storage() noexcept
      : __smem_addr_{__cccl_alloc_static_shared<_Size, _Align>()}
  {}

  static_shared_storage(const static_shared_storage&) = delete;

  static_shared_storage(static_shared_storage&&) = delete;

  static_shared_storage& operator=(const static_shared_storage&) = delete;

  static_shared_storage& operator=(static_shared_storage&&) = delete;

  //! @brief Obtains the address of the storage.
  //!
  //! @return The address of the storage.
  [[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<void> get() const noexcept
  {
    return shared_memory_ptr<void>{__smem_addr_t{__smem_addr_}};
  }

  //! @brief Obtains the address of the storage.
  //!
  //! @return The address of the storage.
  [[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<void> operator&() const noexcept
  {
    return get();
  }
};
} // namespace cuda::experimental

#endif // _CUDA_EXPERIMENTAL___MEMORY_STATIC_SHARED_STORAGE_H
