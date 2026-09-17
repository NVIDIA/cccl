// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__functional/always_true_false.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__mdspan/__copy/mdspan_d2d.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__mdspan/layout_right.h>

CUB_NAMESPACE_BEGIN

namespace detail::copy_mdspan
{
template <typename MdspanIn, typename MdspanOut>
struct copy_mdspan_t
{
  MdspanIn mdspan_in;
  MdspanOut mdspan_out;

  _CCCL_HOST_DEVICE_API copy_mdspan_t(MdspanIn mdspan_in, MdspanOut mdspan_out)
      : mdspan_in{mdspan_in}
      , mdspan_out{mdspan_out}
  {}

  template <typename Idx, typename... Indices>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void operator()(Idx, Indices... indices)
  {
    mdspan_out(indices...) = mdspan_in(indices...);
  }
};

template <class _MDSpanIn, class _MDSpanOut, class _Env>
[[nodiscard]] CUB_RUNTIME_FUNCTION ::cudaError_t
__transform_copy(_MDSpanIn&& __mdspan_in, _MDSpanOut&& __mdspan_out, const _Env& __env)
{
  return cub::DeviceTransform::__transform_internal(
    ::cuda::std::make_tuple(__mdspan_in.data_handle()),
    __mdspan_out.data_handle(),
    __mdspan_in.size(),
    ::cuda::always_true{},
    ::cuda::std::identity{},
    __env);
}

template <typename T_In,
          typename E_In,
          typename L_In,
          typename A_In,
          typename T_Out,
          typename E_Out,
          typename L_Out,
          typename A_Out,
          typename EnvT = ::cuda::std::execution::env<>>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
copy(::cuda::std::mdspan<T_In, E_In, L_In, A_In> mdspan_in,
     ::cuda::std::mdspan<T_Out, E_Out, L_Out, A_Out> mdspan_out,
     const EnvT& env = {})
{
  // In a similar way of Thrust assign_value(), get_value(), iter_swap(), we need to ensure that  __global__ template
  // functions are instantiated from __host__ __device__ functions regardless of whether __CUDA_ARCH__ is defined (in
  // CDP). For this reason, we keep cuda::copy path outside NV_IF_ELSE_TARGET to keep the host path visible. See NVBug
  // 881631.
  struct copy_on_host_t
  {
    ::cuda::std::mdspan<T_In, E_In, L_In, A_In> input;
    ::cuda::std::mdspan<T_Out, E_Out, L_Out, A_Out> output;
    const EnvT& environment;

    _CCCL_HOST ::cudaError_t operator()() const
    {
      _CCCL_TRY
      {
        if (input.extents() != output.extents())
        {
          _CCCL_THROW(::std::invalid_argument, "mdspan extents must be equal");
        }
        if (input.size() == 0)
        {
          return ::cudaSuccess;
        }
        using mdspan_in_t    = ::cuda::device_mdspan<T_In, E_In, L_In, A_In>;
        using mdspan_out_t   = ::cuda::device_mdspan<T_Out, E_Out, L_Out, A_Out>;
        using accessor_in_t  = ::cuda::device_accessor<A_In>;
        using accessor_out_t = ::cuda::device_accessor<A_Out>;
        const mdspan_in_t mdspan_in{input.data_handle(), input.mapping(), accessor_in_t{input.accessor()}};
        const mdspan_out_t mdspan_out{output.data_handle(), output.mapping(), accessor_out_t{output.accessor()}};
        const auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{::cudaStream_t{}}, environment);
        if (stream.get() == nullptr)
        {
          ::cuda::copy(mdspan_in, mdspan_out, stream);
        }
        else
        {
          const ::cuda::__ensure_current_context ctx{stream};
          ::cuda::copy(mdspan_in, mdspan_out, stream);
        }
      }
#if _CCCL_HOSTED()
      _CCCL_CATCH (const ::cuda::cuda_error& error)
      {
        return error.status();
      }
      _CCCL_CATCH (const ::std::invalid_argument& error)
      {
        return ::cudaErrorInvalidValue;
      }
#endif // _CCCL_HOSTED
      _CCCL_CATCH_ALL
      {
        return ::cudaErrorUnknown;
      }
      return ::cudaSuccess;
    }
  };
  const copy_on_host_t copy_on_host{mdspan_in, mdspan_out, env};

  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (return copy_on_host();), //
    ({
      _CCCL_ASSERT(mdspan_in.extents() == mdspan_out.extents(), "mdspan extents must be equal");
      _CCCL_ASSERT((mdspan_in.data_handle() != nullptr && mdspan_out.data_handle() != nullptr) || mdspan_in.size() == 0,
                   "mdspan data handle must not be nullptr if the size is not 0");

      if (mdspan_in.size() != 0)
      {
        auto in_start  = mdspan_in.data_handle();
        auto in_end    = in_start + mdspan_in.mapping().required_span_size();
        auto out_start = mdspan_out.data_handle();
        auto out_end   = out_start + mdspan_out.mapping().required_span_size();
        _CCCL_ASSERT(!(in_end >= out_start && out_end >= in_start), "mdspan memory ranges must not overlap");
      }

      if (mdspan_in.is_exhaustive() && mdspan_out.is_exhaustive()
          && cub::detail::have_same_strides(mdspan_in.mapping(), mdspan_out.mapping()))
      {
        return cub::detail::copy_mdspan::__transform_copy(mdspan_in, mdspan_out, env);
      }
      // we use row-major order for the iteration
      const ::cuda::std::layout_right::mapping<E_In> mapping{mdspan_in.extents()};
      return cub::DeviceFor::__for_each_in_extents(mapping, copy_mdspan_t{mdspan_in, mdspan_out}, env);
    }));
}
} // namespace detail::copy_mdspan
CUB_NAMESPACE_END
