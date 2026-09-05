// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_REFERENCE_WRAPPER_H
#define _CUDA_STD___FUNCTIONAL_REFERENCE_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/weak_result_type.h>
#include <cuda/std/__fwd/reference_wrapper.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT reference_wrapper : public __weak_result_type<_Tp>
{
public:
  // types
  using type = _Tp;

private:
  type* __f_{};

  static _CCCL_API void __fun(_Tp&) noexcept;
  static void __fun(_Tp&&) = delete; // NOLINT(modernize-use-equals-delete)

public:
  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  template <class _Up,
            class = enable_if_t<!is_same_v<remove_cvref_t<_Up>, reference_wrapper>,
                                decltype(__fun(::cuda::std::declval<_Up>()))>>
  _CCCL_API constexpr reference_wrapper(_Up&& __u) noexcept(noexcept(__fun(::cuda::std::declval<_Up>())))
  {
    type& __f = static_cast<_Up&&>(__u);
    __f_      = ::cuda::std::addressof(__f);
  }
  // NOLINTEND(bugprone-forwarding-reference-overload)

  // access
  _CCCL_API constexpr operator type&() const noexcept
  {
    return *__f_;
  }
  [[nodiscard]] _CCCL_API constexpr type& get() const noexcept
  {
    return *__f_;
  }

  // invoke
  template <class... _ArgTypes>
  _CCCL_API constexpr invoke_result_t<type&, _ArgTypes...> operator()(_ArgTypes&&... __args) const
    noexcept(is_nothrow_invocable_v<_Tp&, _ArgTypes...>)
  {
    return ::cuda::std::invoke(get(), ::cuda::std::forward<_ArgTypes>(__args)...);
  }
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES reference_wrapper(_Tp&) -> reference_wrapper<_Tp>;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr reference_wrapper<_Tp> ref(_Tp& __t) noexcept
{
  return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr reference_wrapper<_Tp> ref(reference_wrapper<_Tp> __t) noexcept
{
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr reference_wrapper<const _Tp> cref(const _Tp& __t) noexcept
{
  return reference_wrapper<const _Tp>(__t);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr reference_wrapper<const _Tp> cref(reference_wrapper<_Tp> __t) noexcept
{
  return __t;
}

template <class _Tp>
void ref(const _Tp&&) = delete;
template <class _Tp>
void cref(const _Tp&&) = delete;

// [refwrap.common.ref]
template <class _Rp, class _Tp, class _RpQual, class _TpQual>
_CCCL_CONCEPT __ref_wrap_common_reference_exists_with = _CCCL_REQUIRES_EXPR((_Rp, _Tp, _RpQual, _TpQual), )(
  requires(__is_cuda_std_reference_wrapper_v<_Rp> || __is_std_reference_wrapper_v<_Rp>),
  typename(common_reference_t<typename _Rp::type&, _TpQual>),
  requires(convertible_to<_RpQual, common_reference_t<typename _Rp::type&, _TpQual>>));

template <class _Rp, class _Tp, template <class> class _RpQual, template <class> class _TpQual>
struct basic_common_reference<
  _Rp,
  _Tp,
  _RpQual,
  _TpQual,
  enable_if_t<__ref_wrap_common_reference_exists_with<_Rp, _Tp, _RpQual<_Rp>, _TpQual<_Tp>>
              && !__ref_wrap_common_reference_exists_with<_Tp, _Rp, _TpQual<_Tp>, _RpQual<_Rp>>>>
{
  using type _CCCL_NODEBUG_ALIAS = common_reference_t<typename _Rp::type&, _TpQual<_Tp>>;
};

template <class _Tp, class _Rp, template <class> class _TpQual, template <class> class _RpQual>
struct basic_common_reference<
  _Tp,
  _Rp,
  _TpQual,
  _RpQual,
  enable_if_t<__ref_wrap_common_reference_exists_with<_Rp, _Tp, _RpQual<_Rp>, _TpQual<_Tp>>
              && !__ref_wrap_common_reference_exists_with<_Tp, _Rp, _TpQual<_Tp>, _RpQual<_Rp>>>>
{
  using type _CCCL_NODEBUG_ALIAS = common_reference_t<typename _Rp::type&, _TpQual<_Tp>>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_REFERENCE_WRAPPER_H
