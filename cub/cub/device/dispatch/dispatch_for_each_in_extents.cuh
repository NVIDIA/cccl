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

#include <limits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if __cccl_lib_mdspan

#  include <cub/detail/mdspan_utils.cuh> // size(extent)
#  include <cub/device/dispatch/kernels/for_each_in_extents_kernel.cuh>
#  include <cub/device/dispatch/tuning/tuning_for.cuh>
#  include <cub/util_device.cuh>
#  include <cub/util_namespace.cuh>

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

// The dispatch layer is in the detail namespace until we figure out the tuning API
template <typename ExtentsType, typename OpType, typename PolicyHubT = cub::detail::for_each::policy_hub_t>
class dispatch_t : PolicyHubT
{
  using IndexType        = typename ExtentsType::index_type;
  using UnsigndIndexType = ::cuda::std::make_unsigned_t<IndexType>;
  using max_policy_t     = typename dispatch_t::MaxPolicy;
  //  workaround for nvcc 11.1 bug related to deduction guides, vvv
  using ArrayType = ::cuda::std::array<fast_div_mod<IndexType>, ExtentsType::rank()>;

  static constexpr auto seq = ::cuda::std::make_index_sequence<ExtentsType::rank()>{};

  static constexpr size_t max_index = ::cuda::std::numeric_limits<IndexType>::max();

public:
  dispatch_t() = delete;

  CUB_RUNTIME_FUNCTION dispatch_t(const ExtentsType& ext, const OpType& op, cudaStream_t stream)
      : _ext{ext}
      , _op{op}
      , _stream{stream}
      , _size{cub::detail::size(ext)}
  {}

  dispatch_t(const dispatch_t&) = delete;

  dispatch_t(dispatch_t&&) = delete;

  // InvokeVariadic is a workaround for an nvcc 11.x/12.x problem with variadic template kernel and index sequence
  template <typename ActivePolicyT, ::cuda::std::size_t... Ranks>
  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION
  _CCCL_FORCEINLINE cudaError_t InvokeVariadic(::cuda::std::true_type, ::cuda::std::index_sequence<Ranks...>) const
  {
    ArrayType sub_sizes_div_array    = cub::detail::sub_sizes_fast_div_mod(_ext, seq);
    ArrayType extents_div_array      = cub::detail::extents_fast_div_mod(_ext, seq);
    constexpr unsigned block_threads = ::cuda::std::min(size_t{ActivePolicyT::for_policy_t::block_threads}, max_index);
    unsigned num_cta                 = ::cuda::ceil_div(_size, block_threads);

#  ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog(
      "Invoking detail::for_each_in_extents::static_kernel<<<%u, %u, 0, %p>>>()\n", num_cta, block_threads, _stream);
#  endif
    auto status =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_cta, block_threads, 0, _stream)
        .doit(detail::for_each_in_extents::
                static_kernel<max_policy_t, OpType, ExtentsType, decltype(sub_sizes_div_array), Ranks...>,
              _op,
              _ext,
              sub_sizes_div_array,
              extents_div_array);
    _CUB_RETURN_IF_ERROR(status)
    _CUB_RETURN_IF_STREAM_ERROR(_stream)
    return cudaSuccess;
  }

  template <typename ActivePolicyT, ::cuda::std::size_t... Ranks>
  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION
  _CCCL_FORCEINLINE cudaError_t InvokeVariadic(::cuda::std::false_type, ::cuda::std::index_sequence<Ranks...>) const
  {
    ArrayType sub_sizes_div_array = cub::detail::sub_sizes_fast_div_mod(_ext, seq);
    ArrayType extents_div_array   = cub::detail::extents_fast_div_mod(_ext, seq);
    auto kernel                   = detail::for_each_in_extents::
      dynamic_kernel<max_policy_t, IndexType, OpType, decltype(sub_sizes_div_array), UnsigndIndexType, Ranks...>;

    unsigned block_threads = 256;
    cudaError_t status     = cudaSuccess;
    NV_IF_TARGET(NV_IS_HOST,
                 (int _{}; //
                  status = cudaOccupancyMaxPotentialBlockSize(&_, &block_threads, kernel);));
    _CUB_RETURN_IF_ERROR(status)
    block_threads    = ::cuda::std::min(size_t{block_threads}, max_index);
    unsigned num_cta = ::cuda::ceil_div(_size, block_threads);

#  ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog(
      "Invoking detail::for_each_in_extents::dynamic_kernel<<<%u, %u, 0, %p>>>()\n", num_cta, block_threads, _stream);
#  endif
    status = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_cta, block_threads, 0, _stream)
               .doit(kernel, _op, _ext, sub_sizes_div_array, extents_div_array);
    _CUB_RETURN_IF_ERROR(status)
    _CUB_RETURN_IF_STREAM_ERROR(_stream)
    return cudaSuccess;
  }

  template <typename ActivePolicyT, bool StaticBlockSize>
  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(::cuda::std::bool_constant<StaticBlockSize> v) const
  {
    constexpr auto seq = ::cuda::std::make_index_sequence<ExtentsType::rank()>{};
    return InvokeVariadic<ActivePolicyT>(v, seq);
  }

  template <typename ActivePolicyT>
  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke() const
  {
    if (_size == 0)
    {
      return cudaSuccess;
    }
    constexpr bool static_block_size = (ActivePolicyT::for_policy_t::block_threads > 0);
    return dispatch_t::Invoke<ActivePolicyT>(::cuda::std::bool_constant<static_block_size>{});
  }

  _CCCL_NODISCARD CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  dispatch(const ExtentsType& ext, const OpType& op, cudaStream_t stream)
  {
    using max_policy_t = typename dispatch_t::MaxPolicy;
    int ptx_version    = 0;
    _CUB_RETURN_IF_ERROR(CubDebug(PtxVersion(ptx_version)))
    dispatch_t dispatch(ext, op, stream);
    return CubDebug(max_policy_t::Invoke(ptx_version, dispatch));
  }

private:
  ExtentsType _ext;
  OpType _op;
  cudaStream_t _stream;
  UnsigndIndexType _size;
};

} // namespace detail::for_each_in_extents

CUB_NAMESPACE_END

#endif // __cccl_lib_mdspan