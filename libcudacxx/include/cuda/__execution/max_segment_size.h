//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_MAX_SEG_SIZE_H
#define __CUDA___EXECUTION_MAX_SEG_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__execution/guarantee.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_one_of.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

struct __get_max_segment_size_t;

_CCCL_GLOBAL_CONSTANT auto dynamic_max_segment_size = static_cast<size_t>(-1);

//! A class template that can be used to specify the maximum segment size
//! for segmented algorithms.
//! \tparam _Size The maximum segment size.
template <size_t _Size = dynamic_max_segment_size>
struct max_segment_size : __guarantee
{
  using value_type             = size_t;
  static constexpr size_t size = _Size;

  constexpr max_segment_size() = default;

  _CCCL_API constexpr max_segment_size(size_t) noexcept {}

  _CCCL_API constexpr operator value_type() const noexcept
  {
    return _Size;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_max_segment_size_t&) const noexcept
  {
    return *this;
  }
};

template <>
struct max_segment_size<dynamic_max_segment_size> : __guarantee
{
  using value_type = size_t;

  static constexpr size_t size = dynamic_max_segment_size;

  _CCCL_API constexpr max_segment_size(size_t __s)
      : __val(__s)
  {}

  _CCCL_API constexpr operator value_type() const noexcept
  {
    return __val;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_max_segment_size_t&) const noexcept
  {
    return *this;
  }

private:
  size_t __val;
};

struct __get_max_segment_size_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_max_segment_size_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_max_segment_size = __get_max_segment_size_t{};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_MAX_SEG_SIZE_H
