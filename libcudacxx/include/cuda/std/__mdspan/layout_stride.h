/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_HPP
#define _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__mdspan/compressed_pair.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/macros.h>
#ifdef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#  include <cuda/std/__mdspan/no_unique_address.h>
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
#include <cuda/std/__mdspan/static_array.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/numeric>
#if __MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20
#  include <cuda/std/concepts>
#endif // __MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20
#include <cuda/std/span>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct layout_left
{
  template <class _Extents>
  class mapping;
};
struct layout_right
{
  template <class _Extents>
  class mapping;
};

namespace __detail
{
template <class _Layout, class _Mapping>
_CCCL_INLINE_VAR constexpr bool __is_mapping_of =
  is_same<typename _Layout::template mapping<typename _Mapping::extents_type>, _Mapping>::value;

#if __MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20
template <class _Mp>
concept __layout_mapping_alike = requires {
  requires __is_extents<typename _Mp::extents_type>::value;
  { _Mp::is_always_strided() } -> same_as<bool>;
  { _Mp::is_always_exhaustive() } -> same_as<bool>;
  { _Mp::is_always_unique() } -> same_as<bool>;
  bool_constant<_Mp::is_always_strided()>::value;
  bool_constant<_Mp::is_always_exhaustive()>::value;
  bool_constant<_Mp::is_always_unique()>::value;
};
#endif // __MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20
} // namespace __detail

struct layout_stride
{
  template <class _Extents>
  class mapping
#ifdef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
      : private __detail::__no_unique_address_emulation<
          __detail::__compressed_pair<_Extents, _CUDA_VSTD::array<typename _Extents::index_type, _Extents::rank()>>>
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
  {
  public:
    using extents_type = _Extents;
    using index_type   = typename extents_type::index_type;
    using size_type    = typename extents_type::size_type;
    using rank_type    = typename extents_type::rank_type;
    using layout_type  = layout_stride;

    // This could be a `requires`, but I think it's better and clearer as a `static_assert`.
    static_assert(__detail::__is_extents_v<_Extents>,
                  "layout_stride::mapping must be instantiated with a specialization of _CUDA_VSTD::extents.");

  private:
    //----------------------------------------------------------------------------

    using __strides_storage_t = _CUDA_VSTD::array<index_type, extents_type::rank()>; //_CUDA_VSTD::dextents<index_type,
                                                                                     // extents_type::rank()>;
    using __member_pair_t = __detail::__compressed_pair<extents_type, __strides_storage_t>;

#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    _CCCL_NO_UNIQUE_ADDRESS __member_pair_t __members;
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
    using __base_t = __detail::__no_unique_address_emulation<__member_pair_t>;
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS

    __MDSPAN_FORCE_INLINE_FUNCTION constexpr __strides_storage_t const& __strides_storage() const noexcept
    {
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
      return __members.__second();
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
      return this->__base_t::__ref().__second();
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    }
    __MDSPAN_FORCE_INLINE_FUNCTION constexpr __strides_storage_t& __strides_storage() noexcept
    {
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
      return __members.__second();
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
      return this->__base_t::__ref().__second();
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    }

    template <class _SizeType, size_t... _Ep, size_t... _Idx>
    _CCCL_HOST_DEVICE constexpr index_type
    __get_size(_CUDA_VSTD::extents<_SizeType, _Ep...>, _CUDA_VSTD::integer_sequence<size_t, _Idx...>) const
    {
      return __MDSPAN_FOLD_TIMES_RIGHT(static_cast<index_type>(extents().extent(_Idx)), 1);
    }

    //----------------------------------------------------------------------------

    template <class>
    friend class mapping;

    //----------------------------------------------------------------------------

    // Workaround for non-deducibility of the index sequence template parameter if it's given at the top level
    template <class>
    struct __deduction_workaround;

    template <size_t... _Idxs>
    struct __deduction_workaround<_CUDA_VSTD::index_sequence<_Idxs...>>
    {
      template <class _OtherExtents>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
      _eq_impl(mapping const& __self, mapping<_OtherExtents> const& __other) noexcept
      {
        return __MDSPAN_FOLD_AND((__self.stride(_Idxs) == __other.stride(_Idxs)) /* && ... */)
            && __MDSPAN_FOLD_AND((__self.extents().extent(_Idxs) == __other.extents().extent(_Idxs)) /* || ... */);
      }
      template <class _OtherExtents>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
      _not_eq_impl(mapping const& __self, mapping<_OtherExtents> const& __other) noexcept
      {
        return __MDSPAN_FOLD_OR((__self.stride(_Idxs) != __other.stride(_Idxs)) /* || ... */)
            || __MDSPAN_FOLD_OR((__self.extents().extent(_Idxs) != __other.extents().extent(_Idxs)) /* || ... */);
      }

      template <class... _Integral>
      __MDSPAN_FORCE_INLINE_FUNCTION static constexpr size_t
      _call_op_impl(mapping const& __self, _Integral... __idxs) noexcept
      {
        return __MDSPAN_FOLD_PLUS_RIGHT((__idxs * __self.stride(_Idxs)), /* + ... + */ 0);
      }

      _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t _req_span_size_impl(mapping const& __self) noexcept
      {
        // assumes no negative strides; not sure if I'm allowed to assume that or not
        return __impl::_call_op_impl(__self, (__self.extents().template __extent<_Idxs>() - 1)...) + 1;
      }

      template <class _OtherMapping>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr const __strides_storage_t fill_strides(const _OtherMapping& __map)
      {
        return __strides_storage_t{static_cast<index_type>(__map.stride(_Idxs))...};
      }

      _LIBCUDACXX_HIDE_FROM_ABI static constexpr const __strides_storage_t& fill_strides(const __strides_storage_t& __s)
      {
        return __s;
      }

      template <class _IntegralType>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr const __strides_storage_t
      fill_strides(const _CUDA_VSTD::array<_IntegralType, extents_type::rank()>& __s)
      {
        return __strides_storage_t{static_cast<index_type>(__s[_Idxs])...};
      }

      template <class _IntegralType>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr const __strides_storage_t
      fill_strides(const _CUDA_VSTD::span<_IntegralType, extents_type::rank()>& __s)
      {
        return __strides_storage_t{static_cast<index_type>(__s[_Idxs])...};
      }

      _LIBCUDACXX_HIDE_FROM_ABI static constexpr const __strides_storage_t fill_strides(
        __detail::__extents_to_partially_static_sizes_t<_CUDA_VSTD::dextents<index_type, extents_type::rank()>>&& __s)
      {
        return __strides_storage_t{static_cast<index_type>(__s.template __get_n<_Idxs>())...};
      }

      template <size_t K>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __return_zero()
      {
        return 0;
      }

      template <class _Mapping>
      _LIBCUDACXX_HIDE_FROM_ABI static constexpr typename _Mapping::index_type __OFFSET(const _Mapping& m)
      {
        return m(__return_zero<_Idxs>()...);
      }
    };

    // Can't use defaulted parameter in the __deduction_workaround template because of a bug in MSVC warning C4348.
    using __impl = __deduction_workaround<_CUDA_VSTD::make_index_sequence<_Extents::rank()>>;

    //----------------------------------------------------------------------------

#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit mapping(__member_pair_t&& __m)
        : __members(_CUDA_VSTD::move(__m))
    {}
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit mapping(__base_t&& __b)
        : __base_t(_CUDA_VSTD::move(__b))
    {}
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS

  public: // but not really
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr mapping __make_mapping(
      __detail::__extents_to_partially_static_sizes_t<_Extents>&& __exts,
      __detail::__extents_to_partially_static_sizes_t<_CUDA_VSTD::dextents<index_type, _Extents::rank()>>&&
        __strs) noexcept
    {
      // call the private constructor we created for this purpose
      return mapping(
#ifdef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
        __base_t{
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
          __member_pair_t(extents_type::__make_extents_impl(_CUDA_VSTD::move(__exts)),
                          __strides_storage_t{__impl::fill_strides(_CUDA_VSTD::move(__strs))})
#ifdef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
        }
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
      );
    }
    //----------------------------------------------------------------------------

  public:
    //--------------------------------------------------------------------------------

    _CCCL_HIDE_FROM_ABI constexpr mapping() noexcept               = default;
    _CCCL_HIDE_FROM_ABI constexpr mapping(mapping const&) noexcept = default;

    // nvcc cannot deduce this constructor when using _CCCL_REQUIRES
    template <
      class _IntegralTypes,
      enable_if_t<_CCCL_TRAIT(is_convertible, const remove_const_t<_IntegralTypes>&, index_type), int>           = 0,
      enable_if_t<_CCCL_TRAIT(is_nothrow_constructible, index_type, const remove_const_t<_IntegralTypes>&), int> = 0>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
      extents_type const& __e, _CUDA_VSTD::array<_IntegralTypes, extents_type::rank()> const& __s) noexcept
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
        : __members{
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
        : __base_t(__base_t{__member_pair_t(
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
            __e, __strides_storage_t(__impl::fill_strides(__s))
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
          }
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
              )})
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    {
      /*
       * TODO: check preconditions
       * - __s[i] > 0 is true for all i in the range [0, rank_ ).
       * - REQUIRED-SPAN-SIZE(__e, __s) is a representable value of type index_type ([basic.fundamental]).
       * - If rank_ is greater than 0, then there exists a permutation P of the integers in the
       *   range [0, rank_), such that __s[ pi ] >= __s[ pi − 1 ] * __e.extent( pi − 1 ) is true for
       *   all i in the range [1, rank_ ), where pi is the ith element of P.
       */
    }

    // nvcc cannot deduce this constructor when using _CCCL_REQUIRES
    template <
      class _IntegralTypes,
      enable_if_t<_CCCL_TRAIT(is_convertible, const remove_const_t<_IntegralTypes>&, index_type), int>           = 0,
      enable_if_t<_CCCL_TRAIT(is_nothrow_constructible, index_type, const remove_const_t<_IntegralTypes>&), int> = 0>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
      extents_type const& __e, _CUDA_VSTD::span<_IntegralTypes, extents_type::rank()> const& __s) noexcept
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
        : __members{
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
        : __base_t(__base_t{__member_pair_t(
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
            __e, __strides_storage_t(__impl::fill_strides(__s))
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
          }
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
              )})
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    {
      /*
       * TODO: check preconditions
       * - __s[i] > 0 is true for all i in the range [0, rank_ ).
       * - REQUIRED-SPAN-SIZE(__e, __s) is a representable value of type index_type ([basic.fundamental]).
       * - If rank_ is greater than 0, then there exists a permutation P of the integers in the
       *   range [0, rank_), such that __s[ pi ] >= __s[ pi − 1 ] * __e.extent( pi − 1 ) is true for
       *   all i in the range [1, rank_ ), where pi is the ith element of P.
       */
    }

#if !(__MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20)
    _CCCL_TEMPLATE(class _StridedLayoutMapping)
    _CCCL_REQUIRES(
      _CCCL_TRAIT(_CUDA_VSTD::is_constructible, extents_type, typename _StridedLayoutMapping::extents_type)
        _CCCL_AND __detail::__is_mapping_of<typename _StridedLayoutMapping::layout_type, _StridedLayoutMapping>
          _CCCL_AND(_StridedLayoutMapping::is_always_unique()) _CCCL_AND(_StridedLayoutMapping::is_always_strided()))
#else // ^^^ !__MDSPAN_USE_CONCEPTS ^^^^/ vvv __MDSPAN_USE_CONCEPTS vvv
    template <class _StridedLayoutMapping>
      requires(__detail::__layout_mapping_alike<_StridedLayoutMapping>
               && _CCCL_TRAIT(is_constructible, extents_type, typename _StridedLayoutMapping::extents_type)
               && _StridedLayoutMapping::is_always_unique() && _StridedLayoutMapping::is_always_strided())
#endif // __MDSPAN_USE_CONCEPTS
    __MDSPAN_CONDITIONAL_EXPLICIT(
      (!is_convertible<typename _StridedLayoutMapping::extents_type, extents_type>::value)
      && (__detail::__is_mapping_of<layout_left, _StridedLayoutMapping>
          || __detail::__is_mapping_of<layout_right, _StridedLayoutMapping>
          || __detail::__is_mapping_of<layout_stride, _StridedLayoutMapping>) ) // needs two () due to comma
    _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
      _StridedLayoutMapping const& __other) noexcept // NOLINT(google-explicit-constructor)
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
        : __members{
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
        : __base_t(__base_t{__member_pair_t(
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
            __other.extents(), __strides_storage_t(__impl::fill_strides(__other))
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
          }
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
              )})
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    {
      /*
       * TODO: check preconditions
       * - __other.stride(i) > 0 is true for all i in the range [0, rank_ ).
       * - __other.required_span_size() is a representable value of type index_type ([basic.fundamental]).
       * - OFFSET(__other) == 0
       */
    }

    //--------------------------------------------------------------------------------

    _CCCL_HIDE_FROM_ABI __MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
    {
#ifndef _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
      return __members.__first();
#else // ^^^ !_CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS ^^^ / vvv _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS vvv
      return this->__base_t::__ref().__first();
#endif // _CCCL_HAS_NO_ATTRIBUTE_NO_UNIQUE_ADDRESS
    };

    _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::array<index_type, extents_type::rank()> strides() const noexcept
    {
      return __strides_storage();
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type required_span_size() const noexcept
    {
      index_type __span_size = 1;
      for (rank_type __r = 0; __r != extents_type::rank(); __r++)
      {
        // Return early if any of the extents are zero
        if (extents().extent(__r) == 0)
        {
          return 0;
        }
        __span_size += (static_cast<index_type>(extents().extent(__r) - 1) * __strides_storage()[__r]);
      }
      return __span_size;
    }

    _CCCL_TEMPLATE(class... _Indices)
    _CCCL_REQUIRES((sizeof...(_Indices) == _Extents::rank())
                     _CCCL_AND __fold_and_v<_CCCL_TRAIT(is_convertible, _Indices, index_type)...> //
                       _CCCL_AND __fold_and_v<_CCCL_TRAIT(is_nothrow_constructible, index_type, _Indices)...>)
    __MDSPAN_FORCE_INLINE_FUNCTION
    constexpr index_type operator()(_Indices... __idxs) const noexcept
    {
      // Should the op_impl operate in terms of `index_type` rather than `size_t`?
      // Casting `size_t` to `index_type` here.
      return static_cast<index_type>(__impl::_call_op_impl(*this, static_cast<index_type>(__idxs)...));
    }

    _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_unique() noexcept
    {
      return true;
    }
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_exhaustive() noexcept
    {
      return false;
    }
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_strided() noexcept
    {
      return true;
    }

    _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_unique() noexcept
    {
      return true;
    }
    _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_exhaustive() const noexcept
    {
      return required_span_size() == __get_size(extents(), _CUDA_VSTD::make_index_sequence<extents_type::rank()>());
    }
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_strided() noexcept
    {
      return true;
    }

    _CCCL_TEMPLATE(class _Ext = _Extents)
    _CCCL_REQUIRES((_Ext::rank() > 0))
    _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type stride(rank_type __r) const noexcept
    {
      return __strides_storage()[__r];
    }

#if !(__MDSPAN_USE_CONCEPTS && __MDSPAN_HAS_CXX_20)
    _CCCL_TEMPLATE(class _StridedLayoutMapping)
    _CCCL_REQUIRES(__detail::__is_mapping_of<typename _StridedLayoutMapping::layout_type, _StridedLayoutMapping>
                     _CCCL_AND(extents_type::rank() == _StridedLayoutMapping::extents_type::rank())
                       _CCCL_AND(_StridedLayoutMapping::is_always_strided()))
#else // ^^^ !__MDSPAN_USE_CONCEPTS ^^^ / vvv __MDSPAN_USE_CONCEPTS vvv
    template <class _StridedLayoutMapping>
      requires(__detail::__layout_mapping_alike<_StridedLayoutMapping>
               && (extents_type::rank() == _StridedLayoutMapping::extents_type::rank())
               && _StridedLayoutMapping::is_always_strided())
#endif // __MDSPAN_USE_CONCEPTS
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
    operator==(const mapping& __x, const _StridedLayoutMapping& __y) noexcept
    {
      bool __strides_match = true;
      for (rank_type __r = 0; __r != extents_type::rank(); __r++)
      {
        __strides_match = __strides_match && (__x.stride(__r) == __y.stride(__r));
      }
      return (__x.extents() == __y.extents())
          && (__impl::__OFFSET(__y) == static_cast<typename _StridedLayoutMapping::index_type>(0)) && __strides_match;
    }

    // This one is not technically part of the proposal. Just here to make implementation a bit more optimal hopefully
    _CCCL_TEMPLATE(class _OtherExtents)
    _CCCL_REQUIRES((extents_type::rank() == _OtherExtents::rank()))
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
    operator==(mapping const& __lhs, mapping<_OtherExtents> const& __rhs) noexcept
    {
      return __impl::_eq_impl(__lhs, __rhs);
    }

#if !__MDSPAN_HAS_CXX_20
    _CCCL_TEMPLATE(class _StridedLayoutMapping)
    _CCCL_REQUIRES(__detail::__is_mapping_of<typename _StridedLayoutMapping::layout_type, _StridedLayoutMapping>
                     _CCCL_AND(extents_type::rank() == _StridedLayoutMapping::extents_type::rank())
                       _CCCL_AND(_StridedLayoutMapping::is_always_strided()))
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
    operator!=(const mapping& __x, const _StridedLayoutMapping& __y) noexcept
    {
      return not(__x == __y);
    }

    _CCCL_TEMPLATE(class _OtherExtents)
    _CCCL_REQUIRES((extents_type::rank() == _OtherExtents::rank()))
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
    operator!=(mapping const& __lhs, mapping<_OtherExtents> const& __rhs) noexcept
    {
      return __impl::_not_eq_impl(__lhs, __rhs);
    }
#endif // !__MDSPAN_HAS_CXX_20
  };
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_LAYOUT_STRIDE_HPP
