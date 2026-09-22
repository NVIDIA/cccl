// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
#pragma once

#include <cub/config.cuh>

#include <cuda/std/__type_traits/is_same.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/util_debug.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__functional/always_true_false.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/mdspan>

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

template <class MDSpanIn, class MDSpanOut>
[[nodiscard]] _CCCL_HOST_API ::cudaError_t
__copy_mdspan_bytes(::cuda::stream_ref __stream, MDSpanIn&& __mdspan_in, MDSpanOut&& __mdspan_out)
{
  _CCCL_TRY
  {
    ::cuda::copy_bytes(__stream, __mdspan_in, __mdspan_out);
  }
#if _CCCL_HOSTED()
  _CCCL_CATCH (const ::cuda::cuda_error& __e)
  {
    return __e.status();
  }
  _CCCL_CATCH (const ::std::invalid_argument& __e)
  {
    static_cast<void>(__e);
    return ::cudaErrorInvalidValue;
  }
#endif // _CCCL_HOSTED
  _CCCL_CATCH_ALL
  {
    return ::cudaErrorUnknown;
  }

  return ::cudaSuccess;
}

template <class MDSpanIn, class MDSpanOut, class Env>
[[nodiscard]] CUB_RUNTIME_FUNCTION ::cudaError_t
__transform_copy(MDSpanIn&& __mdspan_in, MDSpanOut&& __mdspan_out, const Env& __env)
{
  return CUB_NS_QUALIFIER::DeviceTransform::__transform_internal(
    ::cuda::std::make_tuple(__mdspan_in.data_handle()),
    __mdspan_out.data_handle(),
    __mdspan_in.size(),
    ::cuda::always_true{},
    ::cuda::std::identity{},
    __env);
}

template <typename TIn,
          typename EIn,
          typename LIn,
          typename AIn,
          typename TOut,
          typename EOut,
          typename LOut,
          typename AOut,
          typename EnvT = ::cuda::std::execution::env<>>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
copy(::cuda::std::mdspan<TIn, EIn, LIn, AIn> mdspan_in,
     ::cuda::std::mdspan<TOut, EOut, LOut, AOut> mdspan_out,
     const EnvT& env = {})
{
  if (mdspan_in.is_exhaustive() && mdspan_out.is_exhaustive()
      && detail::have_same_strides(mdspan_in.mapping(), mdspan_out.mapping()))
  {
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr (::cuda::std::same_as<TIn, TOut>
                  && ::cuda::__detail::__can_mdspan_copy_bytes<TIn, EIn, LIn, TOut, EOut, LOut>
                  && ::cuda::std::__is_callable_v<::cuda::get_stream_t, const EnvT&>)
    {
      NV_IF_TARGET(
        NV_IS_HOST,
        ({
          auto __stream = ::cuda::get_stream(env);

          // cuda::copy_bytes() builds an __ensure_current_context(stream_ref), which calls
          // cuStreamGetCtx(). That driver call rejects the NULL stream with
          // CUDA_ERROR_INVALID_VALUE. Use the transform path, which goes through the runtime
          // API and accepts the NULL stream.
          //
          // Likewise, we cannot retrieve the context for a stream that is capturings so we
          // need to call the kernel.
          if (__stream.get() == nullptr
              || (::cuda::__driver::__streamIsCapturing(__stream.get()) == ::CU_STREAM_CAPTURE_STATUS_ACTIVE))
          {
            return CUB_NS_QUALIFIER::detail::copy_mdspan::__transform_copy(mdspan_in, mdspan_out, env);
          }

          return CUB_NS_QUALIFIER::detail::copy_mdspan::__copy_mdspan_bytes(__stream, mdspan_in, mdspan_out);
        }),
        (return CUB_NS_QUALIFIER::detail::copy_mdspan::__transform_copy(mdspan_in, mdspan_out, env);))
    }
    else
    {
      return CUB_NS_QUALIFIER::detail::copy_mdspan::__transform_copy(mdspan_in, mdspan_out, env);
    }
    // NOLINTEND(bugprone-branch-clone)
  }
  // TODO (fbusato): add ForEachInLayout when mdspan_in and mdspan_out have compatible layouts
  // Compatible layouts could use more efficient iteration patterns
  return cub::DeviceFor::__for_each_in_extents(
    ::cuda::std::layout_right::mapping<EIn>{mdspan_in.extents()}, copy_mdspan_t{mdspan_in, mdspan_out}, env);
}
} // namespace detail::copy_mdspan

CUB_NAMESPACE_END
