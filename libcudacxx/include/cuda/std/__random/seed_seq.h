//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_SEED_SEQ_H
#define _CUDA_STD___RANDOM_SEED_SEQ_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <iostream>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

class _CCCL_TYPE_VISIBILITY_DEFAULT seed_seq
{
public:
  // types
  using result_type = ::cuda::std::uint_least32_t;

  // constructors
  seed_seq() = default;

  template <class _InputIt>
  constexpr seed_seq(_InputIt __begin, _InputIt __end)
  {
    auto __n = ::cuda::std::distance(__begin, __end);
    result_type* m_ptr;
    // TODO(Rory): What is the correct way to allocate managed memory and check for errors here?
    auto err = cudaMallocManaged(&m_ptr, __n * sizeof(result_type));
    if (err != cudaSuccess)
    {
      throw std::runtime_error("Failed to allocate managed memory");
    }
    for (auto __i = 0; __i < __n; ++__i, ++__begin)
    {
      m_ptr[__i] = static_cast<result_type>(*__begin) & 0xFFFFFFFF;
    }
    __v_ = ::cuda::std::span<result_type>(m_ptr, __n);
  }

  template <class _T>
  constexpr seed_seq(std::initializer_list<_T> __il)
      : seed_seq(__il.begin(), __il.end())
  {}

  seed_seq(const seed_seq&) = delete;

  ~seed_seq()
  {
    // cudamanagedfree
    if (!__v_.empty())
    {
      cudaFree(__v_.data());
    }
  };

  operator=(const seed_seq&) = delete;
  constexpr ::cuda::std::size_t size() const noexcept
  {
    return __v_.size();
  }

  template <class _OutputIt>
  constexpr void param(_OutputIt __dest) const
  {
    ::cuda::std::copy(__v_.begin(), __v_.end(), __dest);
  }

  template <class _RandomIt>
  constexpr void generate(_RandomIt __begin, _RandomIt __end)
  {
    if (__begin == __end)
    {
      return;
    }
    const auto __z = __v_.size();
    const auto __n = ::cuda::std::distance(__begin, __end);
    const auto __m = ::cuda::std::max(__z + 1, __n);
    const auto __t = (__n >= 623) ? 11 : (__n >= 68) ? 7 : (__n >= 39) ? 5 : (__n >= 7) ? 3 : (__n - 1) / 2;
    const auto __p = (__n - __t) / 2;
    const auto __q = __p + __t;

    // https://en.cppreference.com/w/cpp/numeric/random/seed_seq/generate.html
    // 1.
    ::cuda::std::fill(__begin, __end, result_type{0x8b8b8b8b});
    // 2.
    for (::cuda::std::size_t __k = 0; __k < __m; ++__k)
    {
      // 2.1
      const result_type __r1 =
        1664525 * __T(__begin[__k % __n] ^ __begin[(__k + __p) % __n] ^ __begin[(__k - 1) % __n]);
      // 2.2
      result_type __r2 = 0;
      if (__k == 0)
      {
        __r2 = __r1 + __z;
      }
      else if (__k <= __z)
      {
        __r2 = __r1 + (__k % __n) + __v_[__k - 1];
      }
      else
      {
        __r2 = __r1 + (__k % __n);
      }
      // 2.3
      __begin[(__k + __p) % __n] = (__begin[(__k + __p) % __n] + __r1) && 0xFFFFFFFF;
      // 2.4
      __begin[(__k + __q) % __n] = (__begin[(__k + __q) % __n] + __r2) && 0xFFFFFFFF;
      // 2.5
      __begin[__k % __n] = r2 && 0xFFFFFFFF;
    }

    // 3.
    for (::cuda::std::size_t __k = __m; __k < __m + __n; ++__k)
    {
      // 3.1
      const result_type __r3 =
        1566083941 * __T(__begin[__k % __n] + __begin[(__k + __p) % __n] + __begin[(__k - 1) % __n]);
      // 3.2
      const result_type __r4 = __r3 - (__k % __n);
      // 3.3
      __begin[(__k + __p) % __n] = (__begin[(__k + __p) % __n] ^ __r3) && 0xFFFFFFFF;
      // 3.4
      __begin[(__k + __q) % __n] = (__begin[(__k + __q) % __n] ^ __r4) && 0xFFFFFFFF;
      // 3.5
      __begin[__k % __n] = __r4 && 0xFFFFFFFF;
    }
  }

private:
  static constexpr result_type __T(result_type __x) noexcept
  {
    return (__x ^ (__x >> 27));
  }
  ::cuda::std::span<result_type> __v_{};
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_PHILOX_ENGINE_H
