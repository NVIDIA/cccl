//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_CONTINUES_ON
#define __CUDAX_EXECUTION_CONTINUES_ON

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

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t
{
  // When calling connect on a continues_on sender, first transform the sender into a
  // schedule_from sender.
  template <class _Sndr>
  _CCCL_NODEBUG_API static constexpr auto transform_sender(_Sndr&& __sndr, ::cuda::std::__ignore_t) noexcept
  {
    // _Sndr is a (possibly cvref-qualified) instance of continues_on_t::__sndr_t
    auto&& [__tag, __sch, __child] = static_cast<_Sndr&&>(__sndr);
    // By default, continues_on(sndr, sch) lowers to schedule_from(sch, sndr) in connect:
    return schedule_from(__sch, ::cuda::std::forward_like<_Sndr>(__child));
  }

  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __detail::__transfer_sndr_t<continues_on_t, _Sch, _Sndr>
  {};

  template <class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr) const
    {
      return continues_on_t{}(static_cast<_Sndr&&>(__sndr), __sch);
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
    {
      return continues_on_t{}(static_cast<_Sndr&&>(__sndr), static_cast<_Sch&&>(__self.__sch));
    }

    _Sch __sch;
  };

  template <class _Sndr, class _Sch>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Sch __sch) const
  {
    static_assert(__is_sender<_Sndr>);
    static_assert(__is_scheduler<_Sch>);
    // continues_on always dispatches based on the domain of the predecessor sender
    using __dom_t _CCCL_NODEBUG_ALIAS = __early_domain_of_t<_Sndr>;
    return execution::transform_sender(__dom_t{}, __sndr_t<_Sch, _Sndr>{{{}, __sch, static_cast<_Sndr&&>(__sndr)}});
  }

  template <class _Sch>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sch __sch) const noexcept -> __closure_t<_Sch>
  {
    return __closure_t<_Sch>{__sch};
  }
};

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<continues_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT continues_on_t continues_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_CONTINUES_ON
