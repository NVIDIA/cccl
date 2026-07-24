//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_SCRATCH_CUH
#define _CUDA_EXPERIMENTAL___COOP_SCRATCH_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
struct __empty_smem_scratch
{};

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::reference_wrapper<__empty_smem_scratch>
__make_empty_smem_scratch() noexcept
{
  __shared__ __empty_smem_scratch __smem_scratch;
  return {__smem_scratch};
}

template <class _Tp, class... _Args>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::reference_wrapper<_Tp> __make_smem_scratch(const _Args&...) noexcept
{
  if constexpr (::cuda::std::is_same_v<_Tp, __empty_smem_scratch>)
  {
    return ::cuda::experimental::coop::__make_empty_smem_scratch();
  }
  else
  {
    __shared__ _Tp __smem_scratch;
    return {__smem_scratch};
  }
}

struct __empty_gmem_scratch
{};

template <class _Tp, class... _Args>
_CCCL_DEVICE inline _Tp __gmem_scratch{};

template <class _Tp, class... _Args>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::reference_wrapper<_Tp> __make_gmem_scratch(const _Args&...) noexcept
{
  if constexpr (::cuda::std::is_same_v<_Tp, __empty_gmem_scratch>)
  {
    return {__gmem_scratch<__empty_gmem_scratch>};
  }
  else
  {
    return {__gmem_scratch<_Tp, _Args...>};
  }
}

template <class _Smem, class _Gmem>
struct __scratch_reqs
{
  using shared_memory_type = _Smem;
  using global_memory_type = _Gmem;

  static constexpr bool needs_shared_memory = !::cuda::std::is_same_v<_Smem, __empty_smem_scratch>;
  static constexpr bool needs_global_memory = !::cuda::std::is_same_v<_Gmem, __empty_gmem_scratch>;

  static constexpr ::cuda::std::size_t shared_memory_size = (needs_shared_memory) ? sizeof(_Smem) : 0;
  static constexpr ::cuda::std::size_t global_memory_size = (needs_global_memory) ? sizeof(_Gmem) : 0;

  static constexpr ::cuda::std::size_t shared_memory_alignment = (needs_shared_memory) ? alignof(_Smem) : 0;
  static constexpr ::cuda::std::size_t global_memory_alignment = (needs_global_memory) ? alignof(_Gmem) : 0;
};

template <class _Alg, class... _Args>
[[nodiscard]] _CCCL_DEVICE _CCCL_CONSTEVAL auto get_scratch_requirements(const _Alg& __alg, _Args&&... __args) noexcept
  -> decltype(_Alg::__get_scratch_requirements(::cuda::std::forward<_Args>(__args)...))
{
  static_assert(::cuda::std::is_invocable_v<_Alg, _Args...>, "_Alg must be invocable with _Args");
  static_assert(::cuda::std::__always_false_v<_Alg>,
                "This function should only be used inside decltype(...) specifier");
}

struct __get_smem_scratch_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_smem_scratch_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr bool query(::cuda::std::execution::forwarding_query_t) noexcept
  {
    return true;
  }
};

struct __get_gmem_scratch_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_gmem_scratch_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr bool query(::cuda::std::execution::forwarding_query_t) noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT __get_smem_scratch_t __get_smem_scratch;
_CCCL_GLOBAL_CONSTANT __get_gmem_scratch_t __get_gmem_scratch;

template <class _SmemScratch>
[[nodiscard]] _CCCL_NODEBUG_API auto shared_memory_scratch(_SmemScratch& __smem_scratch) noexcept
{
  static_assert(!::cuda::std::is_const_v<_SmemScratch>, "_SmemScratch must be non-const");
  return ::cuda::std::execution::prop{__get_smem_scratch, ::cuda::std::reference_wrapper{__smem_scratch}};
}

template <class _GmemScratch>
[[nodiscard]] _CCCL_NODEBUG_API auto global_memory_scratch(_GmemScratch& __gmem_scratch) noexcept
{
  static_assert(!::cuda::std::is_const_v<_GmemScratch>, "_GmemScratch must be non-const");
  return ::cuda::std::execution::prop{__get_gmem_scratch, ::cuda::std::reference_wrapper{__gmem_scratch}};
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_SCRATCH_CUH
