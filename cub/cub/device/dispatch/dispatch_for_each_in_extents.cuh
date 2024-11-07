/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cub/detail/mdspan_utils.cuh> // size(extent)
#  include <cub/device/dispatch/kernels/for_each_in_extents_kernel.cuh>
#  include <cub/device/dispatch/tuning/tuning_for_each_in_extents.cuh> // policy_hub_t
#  include <cub/util_device.cuh>
#  include <cub/util_namespace.cuh> // CUB_NAMESPACE_BEGIN

#  include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#  include <cuda/std/__type_traits/integral_constant.h> // std::integral_constant
#  include <cuda/std/__utility/integer_sequence.h> // std::index_sequence
#  include <cuda/std/array> // std::array
#  include <cuda/std/cstddef> // std::size_t

#  define _CUB_RETURN_IF_ERROR(STATUS)       \
    {                                        \
      cudaError_t error_ = CubDebug(STATUS); \
      if (error_ != cudaSuccess)             \
      {                                      \
        return error_;                       \
      }                                      \
    }

#  define _CUB_RETURN_IF_STREAM_ERROR(STREAM)                         \
    {                                                                 \
      cudaError_t error_ = CubDebug(detail::DebugSyncStream(STREAM)); \
      if (error_ != cudaSuccess)                                      \
      {                                                               \
        CubDebug(error_ = SyncStream(STREAM));                        \
        return error_;                                                \
      }                                                               \
    }

CUB_NAMESPACE_BEGIN

namespace detail::for_each_in_extents
{

// The dispatch layer is in the detail namespace until we figure out tuning API
template <typename ExtentsType, typename OpType, typename PolicyHubT = policy_hub_t>
struct dispatch_t : PolicyHubT
{
  ExtentsType _ext;
  OpType _op;
  cudaStream_t _stream;
  ::cuda::std::size_t _size;

  dispatch_t() = delete;

  CUB_RUNTIME_FUNCTION dispatch_t(const ExtentsType& ext, const OpType& op, cudaStream_t stream)
      : _ext{ext}
      , _op{op}
      , _stream{stream}
      , _size{cub::detail::size(ext)}
  {}

  dispatch_t(const dispatch_t&) = delete;

  dispatch_t(dispatch_t&&) = delete;

  // InvokeVariadic is a workaround for an nvcc problem with variadic templates and index sequence
  template <::cuda::std::size_t... Ranks>
  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION
  _CCCL_FORCEINLINE cudaError_t InvokeVariadic(::cuda::std::index_sequence<Ranks...>) const
  {
    using max_policy_t   = typename dispatch_t::MaxPolicy;
    using ext_index_type = typename ExtentsType::index_type;
    using ArrayType      = ::cuda::std::array<fast_div_mod, sizeof...(Ranks)>; // workaround for nvcc 11.1 bug
    if (_size == 0)
    {
      return cudaSuccess;
    }
    int block_threads             = 256;
    cudaError_t status            = cudaSuccess;
    constexpr auto seq            = ::cuda::std::make_index_sequence<ExtentsType::rank()>{};
    ArrayType sub_sizes_div_array = cub::detail::sub_sizes_fast_div_mod(_ext, seq);
    ArrayType extents_div_array   = cub::detail::extents_fast_div_mod(_ext, seq);
    using FastDivModArrayType     = decltype(sub_sizes_div_array);

    [[maybe_unused]] auto kernel =
      detail::for_each_in_extents::dynamic_kernel<OpType, ext_index_type, decltype(sub_sizes_div_array), Ranks...>;

    NV_IF_TARGET(NV_IS_HOST,
                 (int _{}; //
                  status = cudaOccupancyMaxPotentialBlockSize(&_, &block_threads, kernel);));

    _CUB_RETURN_IF_ERROR(status)
    const auto num_cta = ::cuda::ceil_div(_size, block_threads);
#  ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking detail::for_each_in_extents::dynamic_kernel<<<%d, %d, 0, %p>>>()\n",
            static_cast<int>(num_cta),
            static_cast<int>(block_threads),
            _stream);
#  endif
    status =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        static_cast<unsigned>(num_cta), static_cast<unsigned>(block_threads), 0, _stream)
        .doit(
          detail::for_each_in_extents::dynamic_kernel<OpType, ext_index_type, decltype(sub_sizes_div_array), Ranks...>,
          _op,
          static_cast<ext_index_type>(_size),
          sub_sizes_div_array,
          extents_div_array);
    _CUB_RETURN_IF_ERROR(status)
    _CUB_RETURN_IF_STREAM_ERROR(_stream)
    return cudaSuccess;
  }

  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke() const
  {
    constexpr auto seq = ::cuda::std::make_index_sequence<ExtentsType::rank()>{};
    return InvokeVariadic(seq);
  }

  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  dispatch(const ExtentsType& _ext, const OpType& _op, cudaStream_t _stream)
  {
    return dispatch_t{_ext, _op, _stream}.Invoke();
  }
};

} // namespace detail::for_each_in_extents

CUB_NAMESPACE_END

#endif // _CCCL_STD_VER >= 2017
