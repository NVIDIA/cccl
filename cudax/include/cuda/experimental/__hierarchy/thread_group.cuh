//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___THREAD_GROUP_CUH
#define _CUDA_EXPERIMENTAL___THREAD_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/barrier>
#include <cuda/hierarchy>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include <cuda/experimental/__hierarchy/fwd.cuh>
#include <cuda/experimental/__hierarchy/group.cuh>
#include <cuda/experimental/__hierarchy/group_common.cuh>
#include <cuda/experimental/__hierarchy/implicit_hierarchy.cuh>

#if _CCCL_HAS_COOPERATIVE_GROUPS()
#  include <cooperative_groups.h>
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// this thread_group

template <class _Hierarchy>
class thread_group<__this_hierarchy_group_kind, _Hierarchy> : __this_hierarchy_group_base<thread_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<thread_level, _Hierarchy>;

public:
  using level_type = thread_level;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::hierarchy;
#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
  _CCCL_DEVICE_API thread_group(const ::cooperative_groups::thread_block_tile<1, void>&) noexcept
    : __base_type{::cuda::experimental::__implicit_hierarchy()}
  {}
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

  _CCCL_DEVICE_API void sync() noexcept {}

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    return true;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(const _HierarchyLike&)
  -> thread_group<__this_hierarchy_group_kind, __hierarchy_type_of<_HierarchyLike>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_HOST_DEVICE thread_group(const ::cooperative_groups::thread_block_tile<1, void>&)
  -> thread_group<__this_hierarchy_group_kind, decltype(::cuda::experimental::__implicit_hierarchy())>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_thread(const _HierarchyLike& __hier_like) noexcept
{
  return thread_group{__hier_like};
}

// thread_group in warp level

template <::cuda::std::size_t _Np, class _Hierarchy>
class thread_group<__group_by_hierarchy_group_kind<warp_level, _Np>, _Hierarchy>
{
  static_assert(_Np <= 32, "_Np must be less than or equal to 32");
  static_assert(::cuda::is_power_of_two(_Np), "_Np must be a power of 2");

  unsigned __lane_mask_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  template <class _HierarchyLike>
  _CCCL_DEVICE_API thread_group(const warp_level&, const group_by_t<_Np>&, const _HierarchyLike& __hier_like) noexcept
      : __lane_mask_{((1u << _Np) - 1) << ((gpu_thread.rank(warp) / static_cast<unsigned>(_Np)) * _Np)}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp(__lane_mask_);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    return true;
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(::cuda::std::size_t _Np, class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(const warp_level&, const group_by_t<_Np>&, const _HierarchyLike&)
  -> thread_group<__group_by_hierarchy_group_kind<warp_level, _Np>, __hierarchy_type_of<_HierarchyLike>>;

template <class _Hierarchy>
class thread_group<__group_as_hierarchy_group_kind<warp_level>, _Hierarchy>
{
  unsigned __ngs_;
  unsigned __grank_;
  unsigned __lane_mask_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  template <class _HierarchyLike>
  _CCCL_DEVICE_API thread_group(const warp_level&, group_as __mapping, const _HierarchyLike& __hier_like) noexcept
      : __ngs_{static_cast<unsigned>(__mapping.get().size())}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {
    const auto __trank = gpu_thread.rank_as<unsigned>(warp);
    unsigned __goffset = 0;
    for (unsigned __i = 0; __i < __ngs_; ++__i)
    {
      if (__trank < __goffset + __mapping.get()[__i])
      {
        __grank_     = __i;
        __lane_mask_ = ((1u << __mapping.get()[__i]) - 1) << __goffset;
        return;
      }
      __goffset += __mapping.get()[__i];
    }

    __grank_     = 0xffff'ffffu;
    __lane_mask_ = ::cuda::ptx::get_sreg_lanemask_eq();
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp(__lane_mask_);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    return __grank_ != 0xffff'ffffu;
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(const warp_level&, group_as, const _HierarchyLike&)
  -> thread_group<__group_as_hierarchy_group_kind<warp_level>, __hierarchy_type_of<_HierarchyLike>>;

template <bool _Opt, class _Hierarchy>
class thread_group<__generic_hierarchy_group_kind<warp_level, _Opt>, _Hierarchy>
{
  unsigned __ngs_;
  unsigned __grank_;
  unsigned __lane_mask_;
  unsigned __rank_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  _CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
  _CCCL_REQUIRES(
    ::cuda::std::is_invocable_v<_Mapping, unsigned> _CCCL_AND __is_or_has_hierarchy_member_v<_HierarchyLike>)
  _CCCL_DEVICE_API
  thread_group(const warp_level&, unsigned __ng, _Mapping&& __mapping, const _HierarchyLike& __hier_like) noexcept
      : __ngs_{__ng}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {
    using _MappingResult = ::cuda::std::invoke_result_t<_Mapping, unsigned>;

    if constexpr (::cuda::std::__is_cuda_std_optional_v<_MappingResult>)
    {
      using _Value                    = typename _MappingResult::value_type;
      const auto __opt_mapping_result = __mapping(gpu_thread.rank(warp));

      __grank_ = 0xffff'ffffu;
      if (__opt_mapping_result.has_value())
      {
        if constexpr (::cuda::std::__is_cuda_std_tuple_v<_Value>)
        {
          static_assert(::cuda::std::tuple_size_v<_Value> == 2, "invalid tuple size");
          __grank_ = ::cuda::std::get<0>(__opt_mapping_result.value());
        }
        else
        {
          __grank_ = __opt_mapping_result.value();
        }
      }

      __lane_mask_ = ::__match_any_sync(0xffff'ffffu, __grank_);

      if (__opt_mapping_result.has_value())
      {
        if constexpr (::cuda::std::__is_cuda_std_tuple_v<_Value>)
        {
          __rank_ = ::cuda::std::get<1>(__opt_mapping_result.value());
        }
        else
        {
          __rank_ = ::cuda::std::popcount(__lane_mask_ & ::cuda::ptx::get_sreg_lanemask_lt());
        }
      }
      else
      {
        __lane_mask_ = 0u;
        __rank_      = 0xffff'ffffu;
      }
    }
    else
    {
      if constexpr (::cuda::std::__is_cuda_std_tuple_v<_MappingResult>)
      {
        static_assert(::cuda::std::tuple_size_v<_MappingResult> == 2, "invalid tuple size");
        const auto [__grank, __rank] = __mapping(gpu_thread.rank(warp));
        __grank_                     = __grank;
        __lane_mask_                 = ::__match_any_sync(0xffff'ffffu, __grank_);
        __rank_                      = __rank;
      }
      else
      {
        __grank_     = __mapping(gpu_thread.rank(warp));
        __lane_mask_ = ::__match_any_sync(0xffff'ffffu, __grank_);
        __rank_      = ::cuda::std::popcount(__lane_mask_ & ::cuda::ptx::get_sreg_lanemask_lt());
      }
    }
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    if constexpr (_Opt)
    {
      if (!is_part_of(gpu_thread))
      {
        return;
      }
    }
    ::__syncwarp(__lane_mask_);
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    if constexpr (_Opt)
    {
      return (__lane_mask_ != 0u);
    }
    else
    {
      return true;
    }
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
_CCCL_REQUIRES(::cuda::std::is_invocable_v<_Mapping, unsigned> _CCCL_AND __is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(const warp_level&, unsigned, _Mapping&&, const _HierarchyLike&) -> thread_group<
  __generic_hierarchy_group_kind<warp_level,
                                 ::cuda::std::__is_cuda_std_optional_v<::cuda::std::invoke_result_t<_Mapping, unsigned>>>,
  __hierarchy_type_of<_HierarchyLike>>;

// thread_group in block level

template <::cuda::std::size_t _Np, class _Hierarchy>
class thread_group<__group_by_hierarchy_group_kind<block_level, _Np>, _Hierarchy>
{
  unsigned __ngs_;
  unsigned __grank_;
  unsigned __rank_;
  barrier<thread_scope_block>* __barrier_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  template <class _HierarchyLike>
  _CCCL_DEVICE_API thread_group(
    const block_level&,
    const group_by_t<_Np>&,
    ::cuda::std::span<barrier<thread_scope_block>> __barriers,
    const _HierarchyLike& __hier_like) noexcept
      : __ngs_{gpu_thread.count_as<unsigned>(block, __hier_like) / static_cast<unsigned>(_Np)}
      , __grank_{gpu_thread.rank_as<unsigned>(block, __hier_like) / static_cast<unsigned>(_Np)}
      , __rank_{gpu_thread.rank_as<unsigned>(block, __hier_like) % static_cast<unsigned>(_Np)}
      , __barrier_{__barriers.data() + __grank_}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {
    if (__rank_ == 0)
    {
      init(__barrier_, static_cast<::cuda::std::ptrdiff_t>(_Np));
    }
    ::__syncthreads();
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    // todo: optimize this by synchronizing the warp first
    __barrier_->arrive_and_wait();
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    return true;
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(::cuda::std::size_t _Np, class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(
  const block_level&, const group_by_t<_Np>&, ::cuda::std::span<barrier<thread_scope_block>>, const _HierarchyLike&)
  -> thread_group<__group_by_hierarchy_group_kind<block_level, _Np>, __hierarchy_type_of<_HierarchyLike>>;

template <class _Hierarchy>
class thread_group<__group_as_hierarchy_group_kind<block_level>, _Hierarchy>
{
  unsigned __ngs_;
  unsigned __grank_;
  unsigned __n_;
  unsigned __rank_;
  barrier<thread_scope_block>* __barrier_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  template <class _HierarchyLike>
  _CCCL_DEVICE_API thread_group(
    const block_level&,
    group_as __mapping,
    ::cuda::std::span<barrier<thread_scope_block>> __barriers,
    const _HierarchyLike& __hier_like) noexcept
      : __ngs_{static_cast<unsigned>(__mapping.get().size())}
      , __grank_{0xffff'ffffu}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {
    const auto __trank = gpu_thread.rank_as<unsigned>(block, __hier_);
    unsigned __goffset = 0;
    for (unsigned __i = 0; __i < __ngs_; ++__i)
    {
      if (__trank < __goffset + __mapping.get()[__i])
      {
        __grank_   = __i;
        __n_       = __mapping.get()[__i];
        __rank_    = __trank - __goffset;
        __barrier_ = __barriers.data() + __grank_;
        break;
      }
      __goffset += __mapping.get()[__i];
    }

    if (__grank_ != 0xffff'ffffu && __rank_ == 0)
    {
      init(__barrier_, static_cast<::cuda::std::ptrdiff_t>(__n_));
    }
    ::__syncthreads();
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    if (!is_part_of(gpu_thread))
    {
      return;
    }
    // todo: optimize this by synchronizing the warp first
    __barrier_->arrive_and_wait();
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    return __grank_ != 0xffff'ffffu;
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE
thread_group(const block_level&, group_as, ::cuda::std::span<barrier<thread_scope_block>>, const _HierarchyLike&)
  -> thread_group<__group_as_hierarchy_group_kind<block_level>, __hierarchy_type_of<_HierarchyLike>>;

template <bool _Opt, class _Hierarchy>
class thread_group<__generic_hierarchy_group_kind<block_level, _Opt>, _Hierarchy>
{
  unsigned __ngs_;
  unsigned __grank_;
  unsigned __n_;
  unsigned __rank_;
  barrier<thread_scope_block>* __barrier_;
  _Hierarchy __hier_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  _CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
  _CCCL_REQUIRES(
    ::cuda::std::is_invocable_v<_Mapping, unsigned> _CCCL_AND __is_or_has_hierarchy_member_v<_HierarchyLike>)
  _CCCL_DEVICE_API thread_group(
    const block_level&,
    unsigned __ng,
    _Mapping&& __mapping,
    ::cuda::std::span<barrier<thread_scope_block>> __barriers,
    const _HierarchyLike& __hier_like) noexcept
      : __ngs_{__ng}
      , __grank_{0xffff'ffffu}
      , __n_{0xffff'ffffu}
      , __rank_{0xffff'ffffu}
      , __barrier_{}
      , __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {
    using _MappingResult = ::cuda::std::invoke_result_t<_Mapping, unsigned>;

    const auto __opt_mapping_result = __mapping(gpu_thread.rank(block, __hier_));

    if constexpr (::cuda::std::__is_cuda_std_optional_v<_MappingResult>)
    {
      using _Value = typename _MappingResult::value_type;
      static_assert(::cuda::std::tuple_size_v<_Value> == 3, "invalid tuple size");

      if (__opt_mapping_result.has_value())
      {
        __grank_   = ::cuda::std::get<0>(__opt_mapping_result.value());
        __n_       = ::cuda::std::get<1>(__opt_mapping_result.value());
        __rank_    = ::cuda::std::get<2>(__opt_mapping_result.value());
        __barrier_ = __barriers.data() + __grank_;
      }
    }
    else
    {
      static_assert(::cuda::std::tuple_size_v<_MappingResult> == 3, "invalid tuple size");
      __grank_   = ::cuda::std::get<0>(__opt_mapping_result);
      __n_       = ::cuda::std::get<1>(__opt_mapping_result);
      __rank_    = ::cuda::std::get<2>(__opt_mapping_result);
      __barrier_ = __barriers.data() + __grank_;
    }

    if (__barrier_ != nullptr && __rank_ == 0)
    {
      init(__barrier_, static_cast<::cuda::std::ptrdiff_t>(__n_));
    }
    ::__syncthreads();
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    if constexpr (_Opt)
    {
      if (!is_part_of(gpu_thread))
      {
        return;
      }
    }
    __barrier_->arrive_and_wait();
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) noexcept
  {
    if constexpr (_Opt)
    {
      return (__grank_ != 0xffff'ffffu);
    }
    else
    {
      return true;
    }
  }
#endif // _CCCL_DEVICE_COMPILATION()
};

_CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
_CCCL_REQUIRES(::cuda::std::is_invocable_v<_Mapping, unsigned> _CCCL_AND __is_or_has_hierarchy_member_v<_HierarchyLike>)
_CCCL_HOST_DEVICE thread_group(
  const block_level&, unsigned, _Mapping&&, ::cuda::std::span<barrier<thread_scope_block>>, const _HierarchyLike&)
  -> thread_group<__generic_hierarchy_group_kind<
                    block_level,
                    ::cuda::std::__is_cuda_std_optional_v<::cuda::std::invoke_result_t<_Mapping, unsigned>>>,
                  __hierarchy_type_of<_HierarchyLike>>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_CUH
