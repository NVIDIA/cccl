//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_BULK
#define __CUDAX_EXECUTION_STREAM_BULK

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__execution/bulk.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
struct __bulk_chunked_t : execution::__bulk_t<__bulk_chunked_t>
{
  template <class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t : __bulk_t::__rcvr_base_t<_Shape, _Fn, _Rcvr>
  {
    // We permit this `set_value` function to be called multiple times, once for each
    // thread in the block.
    template <class... _Values>
    _CCCL_DEVICE_API void set_value(_Values&&... __values) noexcept
    {
      const _Shape __tid = threadIdx.x + blockIdx.x * blockDim.x;

      if (__tid < this->__state_->__shape_)
      {
        if constexpr (::cuda::__is_specialization_of_v<_Fn, bulk_t::__bulk_chunked_fn>)
        {
          // If the chunked function was adapted from an unchunked function, we can call
          // the unchunked functions directly.
          this->__state_->__fn_.__fn_(_Shape(__tid), __values...);
        }
        else
        {
          // Otherwise, we call the function with the half-open range [__tid, __tid + 1)
          // to process a single element.
          this->__state_->__fn_(_Shape(__tid), _Shape(__tid + 1), __values...);
        }
      }

      __syncthreads();

      // Only call the downstream receiver once, after all threads have processed their
      // elements.
      if (__tid == 0)
      {
        execution::set_value(static_cast<_Rcvr&&>(this->__state_->__rcvr_), static_cast<_Values&&>(__values)...);
      }
    }
  };

  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  // This function is called when the `bulk_chunked` CPO calls `transform_sender` with a
  // domain argument of stream_domain. It adapts a `bulk_chunked` sender to the stream
  // domain.
  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const
  {
    // Decompose the bulk sender into its components:
    auto& [__tag, __state, __child] = __sndr;
    auto& [__policy, __shape, __fn] = __state;

    using __policy_t  = decltype(__policy);
    using __shape_t   = decltype(__shape);
    using __fn_t      = decltype(__fn);
    using __sndr_t    = __bulk_chunked_t::__sndr_t<decltype(__child), __policy_t, __shape_t, __fn_t>;
    using __closure_t = __bulk_t::__closure_base_t<__policy_t, __shape_t, __fn_t>;

    auto __closure  = __closure_t{__policy, __shape, ::cuda::std::forward_like<_Sndr>(__fn)};
    auto __new_sndr = __sndr_t{{{}, static_cast<__closure_t&&>(__closure), ::cuda::std::forward_like<_Sndr>(__child)}};
    return __stream::__adapt(static_cast<__sndr_t&&>(__new_sndr));
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return true;
  }
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT __bulk_unchunked_t : execution::__bulk_t<__bulk_unchunked_t>
{
  template <class _Shape, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t : __bulk_t::__rcvr_base_t<_Shape, _Fn, _Rcvr>
  {
    // We permit this `set_value` function to be called multiple times, once for each
    // thread in the block.
    template <class... _Values>
    _CCCL_DEVICE_API void set_value(_Values&&... __values) noexcept
    {
      const _Shape __tid = threadIdx.x + blockIdx.x * blockDim.x;

      if (__tid < this->__shape_)
      {
        this->__fn_(_Shape(__tid), __values...);
      }

      __syncthreads();

      // Only call the downstream receiver once, after all threads have processed their
      // elements.
      if (__tid == 0)
      {
        execution::set_value(static_cast<_Rcvr&&>(this->__rcvr_), static_cast<_Values&&>(__values)...);
      }
    }
  };

  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  // This function is called when the `bulk_unchunked` CPO calls `transform_sender` with a
  // domain argument of stream_domain. It adapts a `bulk_unchunked` sender to the stream
  // domain.
  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const
  {
    // Decompose the bulk sender into its components:
    auto& [__tag, __state, __child] = __sndr;
    auto& [__policy, __shape, __fn] = __state;

    using __policy_t  = decltype(__policy);
    using __shape_t   = decltype(__shape);
    using __fn_t      = decltype(__fn);
    using __sndr_t    = __bulk_unchunked_t::__sndr_t<decltype(__child), __policy_t, __shape_t, __fn_t>;
    using __closure_t = __bulk_t::__closure_base_t<__policy_t, __shape_t, __fn_t>;

    auto __closure  = __closure_t{__policy, __shape, ::cuda::std::forward_like<_Sndr>(__fn)};
    auto __new_sndr = __sndr_t{{{}, static_cast<__closure_t&&>(__closure), ::cuda::std::forward_like<_Sndr>(__child)}};
    return __stream::__adapt(static_cast<__sndr_t&&>(__new_sndr));
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

struct __bulk_t : execution::__bulk_t<__bulk_t>
{
  template <class _Sndr, class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __bulk_t::__sndr_base_t<_Sndr, _Policy, _Shape, _Fn>
  {};

  template <class _Policy, class _Shape, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t : __bulk_t::__closure_base_t<_Policy, _Shape, _Fn>
  {};

  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr, ::cuda::std::__ignore_t) const -> decltype(auto)
  {
    // This converts a bulk sender into a bulk_chunked sender, which will then be
    // further transformed by __bulk_chunked_t above.
    return bulk.transform_sender(static_cast<_Sndr&&>(__sndr), env{});
  }
};

} // namespace __stream

template <>
struct stream_domain::__apply_t<bulk_chunked_t> : __stream::__bulk_chunked_t
{};

template <>
struct stream_domain::__apply_t<bulk_unchunked_t> : __stream::__bulk_unchunked_t
{};

template <>
struct stream_domain::__apply_t<bulk_t> : __stream::__bulk_t
{};

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__stream::__bulk_chunked_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__stream::__bulk_unchunked_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> =
  3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t structured_binding_size<__stream::__bulk_t::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_BULK
