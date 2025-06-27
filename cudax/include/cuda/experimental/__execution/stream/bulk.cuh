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
template <>
struct stream_domain::__apply_t<bulk_chunked_t> : __bulk_t<__apply_t<bulk_chunked_t>>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t<__apply_t<bulk_chunked_t>>::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {}

    // We permit this `set_value` function to be called multiple times, once for each
    // thread in the block.
    template <class... _Values>
    _CCCL_DEVICE_API void set_value(_Values&&... __values) noexcept
    {
      const _Shape __tid = threadIdx.x + blockIdx.x * blockDim.x;

      if (__tid < this->__shape_)
      {
        if constexpr (__is_specialization_of_v<_Fn, bulk_t::__bulk_chunked_fn>)
        {
          // If the chunked function was adapted from an unchunked function, we can call
          // the unchunked functions directly.
          this->__fn_.__fn_(_Shape(__tid), __values...);
        }
        else
        {
          // Otherwise, we call the function with the half-open range [__tid, __tid + 1)
          // to process a single element.
          this->__fn_(_Shape(__tid), _Shape(__tid + 1), __values...);
        }
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

  // This function is called when the `bulk_chunked` CPO calls `transform_sender` with a
  // domain argument of stream_domain. It adapts a `bulk_chunked` sender to the stream
  // domain.
  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr) const
  {
    // Decompose the bulk sender into its components:
    auto& [__tag, __state, __child] = __sndr;
    auto& [__policy, __shape, __fn] = __state;

    using __sndr_t = __bulk_t::__sndr_t<decltype(__child), decltype(__policy), decltype(__shape), decltype(__fn)>;
    return __stream::__adapt(__sndr_t{
      {}, {__policy, __shape, _CUDA_VSTD::forward_like<_Sndr>(__fn)}, _CUDA_VSTD::forward_like<_Sndr>(__child)});
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return true;
  }
};

template <>
struct stream_domain::__apply_t<bulk_unchunked_t> : __bulk_t<__apply_t<bulk_unchunked_t>>
{
  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  using __base_opstate_t = __bulk_t<__apply_t<bulk_unchunked_t>>::__opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>;

  template <class _CvSndr, class _Rcvr, class _Shape, class _Fn>
  struct __opstate_t : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>
  {
    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, _Shape __shape, _Fn __fn)
        : __base_opstate_t<_CvSndr, _Rcvr, _Shape, _Fn>{
            static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr), __shape, static_cast<_Fn&&>(__fn)}
    {}

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

  // This function is called when the `bulk_unchunked` CPO calls `transform_sender` with a
  // domain argument of stream_domain. It adapts a `bulk_unchunked` sender to the stream
  // domain.
  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr) const
  {
    // Decompose the bulk sender into its components:
    auto& [__tag, __state, __child] = __sndr;
    auto& [__policy, __shape, __fn] = __state;

    using __sndr_t = __bulk_t::__sndr_t<decltype(__child), decltype(__policy), decltype(__shape), decltype(__fn)>;
    return __stream::__adapt(__sndr_t{
      {}, {__policy, __shape, _CUDA_VSTD::forward_like<_Sndr>(__fn)}, _CUDA_VSTD::forward_like<_Sndr>(__child)});
  }

  _CCCL_API static constexpr bool __is_chunked() noexcept
  {
    return false;
  }
};

template <>
struct stream_domain::__apply_t<bulk_t>
{
  template <class _Sndr>
  _CCCL_API constexpr auto operator()(_Sndr&& __sndr) const -> decltype(auto)
  {
    // This converts a bulk sender into a bulk_chunked sender, which will then be
    // further transformed by the __apply_t<bulk_chunked_t> specialization above.
    return bulk.transform_sender(static_cast<_Sndr&&>(__sndr), env{});
  }
};

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t
  structured_binding_size<__bulk_t<stream_domain::__apply_t<bulk_chunked_t>>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> =
    3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t
  structured_binding_size<__bulk_t<stream_domain::__apply_t<bulk_unchunked_t>>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> =
    3;

template <class _Sndr, class _Policy, class _Shape, class _Fn>
inline constexpr size_t
  structured_binding_size<__bulk_t<stream_domain::__apply_t<bulk_t>>::__sndr_t<_Sndr, _Policy, _Shape, _Fn>> = 3;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_BULK
