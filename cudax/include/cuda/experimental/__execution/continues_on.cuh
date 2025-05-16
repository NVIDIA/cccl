//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONTINUES_ON
#define __CUDAX_ASYNC_DETAIL_CONTINUES_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/ignore.h>

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
  template <class _Sndr>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr, _CUDA_VSTD::__ignore_t) noexcept
  {
    // _Sndr is a (possibly cvref-qualified) instance of continues_on_t::__sndr_t
    auto&& [__tag, __sch, __child] = static_cast<_Sndr&&>(__sndr);
    // By default, continues_on(sndr, sch) lowers to schedule_from(sch, sndr) in connect:
    return schedule_from(__sch, static_cast<__copy_cvref_t<_Sndr&&, decltype(__child)>>(__child));
  }

  template <class _Sndr, class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;

  template <class _Sndr, class _Sch>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr __sndr, _Sch __sch) const;

  template <class _Sch>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sch __sch) const noexcept -> __closure_t<_Sch>;
};

template <class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__closure_t
{
  _Sch __sch;

  template <class _Sndr>
  _CCCL_TRIVIAL_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
  {
    return continues_on_t()(static_cast<_Sndr&&>(__sndr), static_cast<_Sch&&>(__self.__sch));
  }
};

template <class _Sndr, class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS continues_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    template <class _SetTag>
    _CCCL_API auto query(get_completion_scheduler_t<_SetTag>) const = delete;

    _CCCL_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> _Sch
    {
      return __sndr_->__sch_;
    }

    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<_Sndr, _Query>)
    [[nodiscard]] _CCCL_API auto query(_Query) const -> __query_result_t<_Query, env_of_t<_Sndr>>
    {
      return execution::get_env(__sndr_->__sndr_).query(_Query{});
    }

    const __sndr_t* __sndr_;
  };

  template <class _Self, class... _Env>
  _CCCL_API static constexpr auto get_completion_signatures()
  {
    using __sndr_t = __copy_cvref_t<_Self, schedule_from_t::__sndr_t<_Sndr, _Sch>>;
    return execution::get_completion_signatures<__sndr_t, _Env...>();
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{this};
  }
};

template <class _Sndr, class _Sch>
_CCCL_TRIVIAL_API constexpr auto continues_on_t::operator()(_Sndr __sndr, _Sch __sch) const
{
  static_assert(__is_sender<_Sndr>);
  static_assert(__is_scheduler<_Sch>);
  using __dom_t _CCCL_NODEBUG_ALIAS = domain_for_t<_Sndr>;
  return execution::transform_sender(__dom_t{}, __sndr_t<_Sndr, _Sch>{{}, __sch, static_cast<_Sndr&&>(__sndr)});
}

template <class _Sch>
_CCCL_TRIVIAL_API constexpr auto continues_on_t::operator()(_Sch __sch) const noexcept -> __closure_t<_Sch>
{
  return __closure_t<_Sch>{__sch};
}

template <class _Sndr, class _Sch>
inline constexpr size_t structured_binding_size<continues_on_t::__sndr_t<_Sndr, _Sch>> = 3;

_CCCL_GLOBAL_CONSTANT continues_on_t continues_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
