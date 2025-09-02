//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_VISIT
#define __CUDAX_EXECUTION_VISIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/is_aggregate.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

#define _CCCL_BIND_CHILD(_Ord) , _CCCL_PP_CAT(__child, _Ord)
#define _CCCL_FWD_CHILD(_Ord)  , _CCCL_FWD_LIKE(_Sndr, _CCCL_PP_CAT(__child, _Ord))
#define _CCCL_FWD_LIKE(_X, _Y) static_cast<::cuda::std::__copy_cvref_t<_X&&, decltype(_Y)>>(_Y)

namespace cuda::experimental::execution
{
#if _CCCL_HAS_BUILTIN(__builtin_structured_binding_size)

template <class _Sndr>
inline constexpr size_t structured_binding_size = __builtin_structured_binding_size(_Sndr);

#else // ^^^ _CCCL_HAS_BUILTIN(__builtin_structured_binding_size) ^^^ /
      // vvv !_CCCL_HAS_BUILTIN(__builtin_structured_binding_size) vvv

struct __any_t
{
  template <class _Ty>
  _CCCL_API operator _Ty&&();
};

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")

// use the "magic tuple" trick to get the arity of a structured binding
// see https://github.com/apolukhin/magic_get
template <class _Ty, bool = ::cuda::std::is_aggregate_v<_Ty>>
struct __arity_of_t
{
  template <class... _Ts, class _Uy = _Ty, class _Uy2 = decltype(_Uy{_Ts{}...}), class _Self = __arity_of_t>
  _CCCL_API auto operator()(_Ts... __ts) -> decltype(_Self{}(__ts..., __any_t{}));

  template <class... _Ts>
  _CCCL_API auto operator()(_Ts...) const -> char (*)[sizeof...(_Ts) + 1];
};

template <class _Ty>
struct __arity_of_t<_Ty, false>
{
  _CCCL_API auto operator()() const -> char*;
};

_CCCL_DIAG_POP

// Specialize this for each sender type that can be used to initialize a structured binding.
template <class _Sndr>
inline constexpr size_t structured_binding_size = sizeof(*__arity_of_t<_Sndr>{}()) - 2ul;

#endif // _CCCL_HAS_BUILTIN(__builtin_structured_binding_size)

template <class _Sndr>
inline constexpr size_t structured_binding_size<_Sndr&> = structured_binding_size<_Sndr>;

template <class _Sndr>
inline constexpr size_t structured_binding_size<_Sndr const&> = structured_binding_size<_Sndr>;

// If structured bindings can be used to introduce a pack, then `visit` has a very simple
// implementation. Otherwise, we need an `__unpack` function template specialized for
#if __cpp_structured_bindings >= 202411L

// C++26, structured binding can introduce a pack.
struct _CCCL_TYPE_VISIBILITY_DEFAULT visit_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Visitor, class _CvSndr, class _Context>
    requires(static_cast<int>(structured_binding_size<_CvSndr>) >= 2)
  _CCCL_NODEBUG_API constexpr auto operator()(_Visitor& __visitor, _CvSndr&& __sndr, _Context& __context) const
    -> decltype(auto)
  {
    auto&& [__tag, __data, ... __children] = static_cast<_CvSndr&&>(__sndr);
    return __visitor(__context, __tag, _CCCL_FWD_LIKE(_CvSndr, __data), _CCCL_FWD_LIKE(_CvSndr, __children)...);
  }
};

#else // ^^^ __cpp_structured_bindings >= 202411L / !__cpp_structured_bindings >= 202411L vvv

template <size_t _Arity>
struct __sender_type_cannot_be_used_to_initialize_a_structured_binding;

template <size_t _Arity>
struct __unpack
{
  // This is to generate a compile-time error if the sender type cannot be used to
  // initialize a structured binding.
  _CCCL_API void operator()(::cuda::std::__ignore_t,
                            __sender_type_cannot_be_used_to_initialize_a_structured_binding<_Arity>,
                            ::cuda::std::__ignore_t) const;
};

#  define _CCCL_UNPACK_SENDER(_Arity)                                                                               \
    template <>                                                                                                     \
    struct __unpack<2 + _Arity>                                                                                     \
    {                                                                                                               \
      _CCCL_EXEC_CHECK_DISABLE                                                                                      \
      template <class _Visitor, class _Sndr, class _Context>                                                        \
      _CCCL_API constexpr auto operator()(_Visitor& __visitor, _Sndr&& __sndr, _Context& __context) const           \
        -> decltype(auto)                                                                                           \
      {                                                                                                             \
        auto&& [__tag, __data _CCCL_PP_REPEAT(_Arity, _CCCL_BIND_CHILD)] = static_cast<_Sndr&&>(__sndr);            \
        return __visitor(__context, __tag, _CCCL_FWD_LIKE(_Sndr, __data) _CCCL_PP_REPEAT(_Arity, _CCCL_FWD_CHILD)); \
      }                                                                                                             \
    }

_CCCL_UNPACK_SENDER(0);
_CCCL_UNPACK_SENDER(1);
_CCCL_UNPACK_SENDER(2);
_CCCL_UNPACK_SENDER(3);
_CCCL_UNPACK_SENDER(4);
_CCCL_UNPACK_SENDER(5);
_CCCL_UNPACK_SENDER(6);
_CCCL_UNPACK_SENDER(7);

struct _CCCL_TYPE_VISIBILITY_DEFAULT visit_t
{
  _CCCL_TEMPLATE(class _Visitor, class _Sndr, class _Context)
  _CCCL_REQUIRES((static_cast<int>(structured_binding_size<_Sndr>) >= 2))
  _CCCL_NODEBUG_API constexpr auto operator()(_Visitor& __visitor, _Sndr&& __sndr, _Context& __context) const
    -> decltype(auto)
  {
    // This `if constexpr` shouldn't be needed given the `requires` clause above. It is
    // here because nvcc 12.0 has a bug where the full signature of the function template
    // -- including the return type -- is instantiated before the `requires` clause is
    // checked.
    if constexpr (static_cast<int>(structured_binding_size<_Sndr>) >= 2)
    {
      return __unpack<structured_binding_size<_Sndr>>{}(__visitor, static_cast<_Sndr&&>(__sndr), __context);
    }
  }
};

#endif // ^^^ __cpp_structured_bindings < 202411L

[[maybe_unused]]
_CCCL_GLOBAL_CONSTANT visit_t visit{};

template <class _Visitor, class _CvSndr, class _Context>
using __visit_result_t _CCCL_NODEBUG_ALIAS =
  decltype(execution::visit(declval<_Visitor&>(), declval<_CvSndr>(), declval<_Context&>()));

} // namespace cuda::experimental::execution

#undef _CCCL_FWD_LIKE
#undef _CCCL_FWD_CHILD
#undef _CCCL_BIND_CHILD

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_VISIT
