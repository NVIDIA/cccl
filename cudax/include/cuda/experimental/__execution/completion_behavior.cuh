//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_COMPLETION_BEHAVIOR
#define __CUDAX_EXECUTION_COMPLETION_BEHAVIOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__utility/rel_ops.h>
#include <cuda/std/initializer_list>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/fwd.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_behavior
namespace __completion_behavior
{
enum class _CCCL_TYPE_VISIBILITY_DEFAULT completion_behavior : int
{
  unknown, ///< The completion behavior is unknown.
  asynchronous, ///< The operation's completion will not happen on the calling thread before `start()`
                ///< returns.
  synchronous, ///< The operation's completion happens-before the return of `start()`.
  inline_completion ///< The operation completes synchronously within `start()` on the same thread that called
                    ///< `start()`.
};

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
[[nodiscard]] _CCCL_API constexpr auto operator<=>(completion_behavior __a, completion_behavior __b) noexcept
  -> ::cuda::std::strong_ordering
{
  return static_cast<int>(__a) <=> static_cast<int>(__b);
}
#else
[[nodiscard]] _CCCL_API constexpr auto operator<(completion_behavior __a, completion_behavior __b) noexcept -> bool
{
  return static_cast<int>(__a) < static_cast<int>(__b);
}
[[nodiscard]] _CCCL_API constexpr auto operator==(completion_behavior __a, completion_behavior __b) noexcept -> bool
{
  return static_cast<int>(__a) == static_cast<int>(__b);
}
using namespace ::cuda::std::rel_ops;
#endif

template <completion_behavior _CB>
using __constant_t = ::cuda::std::integral_constant<completion_behavior, _CB>;

using __unknown_t           = __constant_t<completion_behavior::unknown>;
using __asynchronous_t      = __constant_t<completion_behavior::asynchronous>;
using __synchronous_t       = __constant_t<completion_behavior::synchronous>;
using __inline_completion_t = __constant_t<completion_behavior::inline_completion>;
} // namespace __completion_behavior

struct _CCCL_TYPE_VISIBILITY_DEFAULT min_t;

struct completion_behavior
{
private:
  template <__completion_behavior::completion_behavior _CB>
  using __constant_t = ::cuda::std::integral_constant<__completion_behavior::completion_behavior, _CB>;

  friend struct min_t;

public:
  struct _CCCL_TYPE_VISIBILITY_DEFAULT unknown_t : __completion_behavior::__unknown_t
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT asynchronous_t : __completion_behavior::__asynchronous_t
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT synchronous_t : __completion_behavior::__synchronous_t
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT inline_completion_t : __completion_behavior::__inline_completion_t
  {};

  static constexpr unknown_t unknown{};
  static constexpr asynchronous_t asynchronous{};
  static constexpr synchronous_t synchronous{};
  static constexpr inline_completion_t inline_completion{};
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_behavior: A sender can define this attribute to describe the sender's
// completion behavior
struct get_completion_behavior_t
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::std::__ignore_t, ::cuda::std::__ignore_t = {}) const noexcept
  {
    return completion_behavior::unknown;
  }

  _CCCL_TEMPLATE(class _Attrs)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_behavior_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, ::cuda::std::__ignore_t = {}) const noexcept
  {
    static_assert(__nothrow_queryable_with<_Attrs, get_completion_behavior_t>,
                  "The get_completion_behavior query must be noexcept.");
    static_assert(::cuda::std::is_convertible_v<__query_result_t<_Attrs, get_completion_behavior_t>,
                                                __completion_behavior::completion_behavior>,
                  "The get_completion_behavior query must return one of the static member variables in "
                  "execution::completion_behavior.");
    return __attrs.query(*this);
  }

  _CCCL_TEMPLATE(class _Attrs, class _Env)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_behavior_t, const _Env&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
  {
    static_assert(__nothrow_queryable_with<_Attrs, get_completion_behavior_t, const _Env&>,
                  "The get_completion_behavior query must be noexcept.");
    static_assert(::cuda::std::is_convertible_v<__query_result_t<_Attrs, get_completion_behavior_t, const _Env&>,
                                                __completion_behavior::completion_behavior>,
                  "The get_completion_behavior query must return one of the static member variables in "
                  "execution::completion_behavior.");
    return __attrs.query(*this, __env);
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT min_t
{
  using __completion_behavior_t = __completion_behavior::completion_behavior;

  [[nodiscard]] _CCCL_API static constexpr auto
  __minimum(::cuda::std::initializer_list<__completion_behavior_t> __cbs) noexcept -> __completion_behavior_t
  {
    auto __result = __completion_behavior::completion_behavior::inline_completion;
    for (auto __cb : __cbs)
    {
      if (__cb < __result)
      {
        __result = __cb;
      }
    }
    return __result;
  }

  template <__completion_behavior_t... _CBs>
  [[nodiscard]] _CCCL_API constexpr auto operator()(completion_behavior::__constant_t<_CBs>...) const noexcept
  {
    constexpr auto __behavior = __minimum({_CBs...});

    if constexpr (__behavior == completion_behavior::unknown)
    {
      return completion_behavior::unknown;
    }
    else if constexpr (__behavior == completion_behavior::asynchronous)
    {
      return completion_behavior::asynchronous;
    }
    else if constexpr (__behavior == completion_behavior::synchronous)
    {
      return completion_behavior::synchronous;
    }
    else if constexpr (__behavior == completion_behavior::inline_completion)
    {
      return completion_behavior::inline_completion;
    }
    _CCCL_UNREACHABLE();
  }
};

_CCCL_GLOBAL_CONSTANT min_t min{};

template <class _Sndr, class... _Env>
[[nodiscard]] _CCCL_API constexpr auto get_completion_behavior() noexcept
{
  using __behavior_t = __call_result_t<get_completion_behavior_t, env_of_t<_Sndr>, const _Env&...>;
  return __behavior_t{};
}

template <class _Attrs, class... _Env>
_CCCL_CONCEPT __completes_inline =
  (__call_result_t<get_completion_behavior_t, const _Attrs&, const _Env&...>{}
   == completion_behavior::inline_completion);

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_COMPLETION_BEHAVIOR
