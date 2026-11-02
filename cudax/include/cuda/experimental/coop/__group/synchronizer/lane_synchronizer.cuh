//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LANE_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LANE_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/hierarchy>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/group_by.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
class lane_synchronizer
{
public:
  struct __synchronizer_instance
  {
    using arrival_token _CCCL_NODEBUG = __dummy_arrival_token<__synchronizer_instance>;

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API void do_sync(const _MappingResult& __mapping_result, const _Hierarchy&) const noexcept
    {
      ::__syncwarp(__mapping_result.lane_mask().value());
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API void do_sync_aligned(const _MappingResult& __mapping_result, const _Hierarchy&) const noexcept
    {
      ::__syncwarp(__mapping_result.lane_mask().value());
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API arrival_token do_arrive(const _MappingResult&, const _Hierarchy&) const noexcept
    {
      return arrival_token{};
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _MappingResult&, const _Hierarchy&) const noexcept
    {
      return arrival_token{};
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API void do_wait(const _MappingResult& __mapping_result,
                                  const _Hierarchy& __hier,
                                  [[maybe_unused]] arrival_token&& __tok) const noexcept
    {
      _CCCL_ASSERT(__tok.__valid(), "invalid arrival token passed to do_wait function");
      do_sync(__mapping_result, __hier);
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API void do_wait_aligned(const _MappingResult& __mapping_result,
                                          const _Hierarchy& __hier,
                                          [[maybe_unused]] arrival_token&& __tok) const noexcept
    {
      _CCCL_ASSERT(__tok.__valid(), "invalid arrival token passed to do_wait_aligned function");
      do_sync_aligned(__mapping_result, __hier);
    }

    [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance view() const noexcept
    {
      return *this;
    }

    template <class _MappingResult, class _Hierarchy>
    _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&) noexcept
    {}
  };

  _CCCL_HIDE_FROM_ABI explicit lane_synchronizer() = default;

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance
  make_instance(const _Unit&, const _ParentGroup&, const _MappingResult& __mapping_result) const noexcept
  {
    static_assert(::cuda::std::is_same_v<_Unit, thread_level>, "_Unit must be cuda::thread_level");
    static_assert(__group_mapping_result<_MappingResult>);
    if (__mapping_result.is_valid())
    {
      _CCCL_ASSERT(::cuda::std::popcount(__mapping_result.lane_mask().value()) == __mapping_result.unit_count(),
                   "lane_synchronizer can only synchronize units within the same warp");
    }
    return {};
  }
};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LANE_SYNCHRONIZER_CUH
