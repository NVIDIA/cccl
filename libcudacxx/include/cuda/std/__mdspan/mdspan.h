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

#ifndef _LIBCUDACXX___MDSPAN_MDSPAN_HPP
#define _LIBCUDACXX___MDSPAN_MDSPAN_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/all_of.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__mdspan/compressed_pair.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/rank.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/as_const.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/span>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2014

_CCCL_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

template <class _ElementType,
          class _Extents,
          class _LayoutPolicy   = layout_right,
          class _AccessorPolicy = default_accessor<_ElementType>>
class mdspan
{
private:
  static_assert(__detail::__is_extents_v<_Extents>,
                "mdspan's Extents template parameter must be a specialization of _CUDA_VSTD::extents.");

  // Workaround for non-deducibility of the index sequence template parameter if it's given at the top level
  template <class>
  struct __deduction_workaround;

  template <size_t... _Idxs>
  struct __deduction_workaround<_CUDA_VSTD::index_sequence<_Idxs...>>
  {
    using index_type = typename _Extents::index_type;

    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __size(mdspan const& __self) noexcept
    {
      return __MDSPAN_FOLD_TIMES_RIGHT(
        (__self.__mapping_ref().extents().template __extent<_Idxs>()), /* * ... * */ size_t(1));
    }
    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __empty(mdspan const& __self) noexcept
    {
      return (__self.rank() > 0)
          && __MDSPAN_FOLD_OR((__self.__mapping_ref().extents().template __extent<_Idxs>() == index_type(0)));
    }

    template <class... _SizeTypes>
    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
    __check_index(_Extents const& exts, _SizeTypes... __indices)
    {
#  if _CCCL_STD_VER >= 2017
      return (((is_unsigned_v<index_type> ? true : static_cast<index_type>(__indices) >= 0)
               && static_cast<index_type>(__indices) < exts.extent(_Idxs))
              && ...);
#  else
      return true;
#  endif // _CCCL_STD_VER >= 2017
    }

    template <class... _SizeTypes>
    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr index_type
    __index(mdspan const& __self, _SizeTypes... __indices) noexcept
    {
      _CCCL_ASSERT(__check_index(__self.__mapping_ref().extents(), __indices...),
                   "cuda::std::mdspan subscript out of range!");
      const index_type __res = __self.__mapping_ref()(index_type(__indices)...);
      return __res;
    }
    template <class _SizeType, size_t _Np>
    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr index_type
    __index(mdspan const& __self, const _CUDA_VSTD::array<_SizeType, _Np>& __indices) noexcept
    {
      _CCCL_ASSERT(__check_index(__self.__mapping_ref().extents(), __indices[_Idxs]...),
                   "cuda::std::mdspan subscript out of range!");
      const index_type __res = __self.__mapping_ref()(__indices[_Idxs]...);
      return __res;
    }
    template <class _SizeType, size_t _Np>
    _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr index_type
    __index(mdspan const& __self, const _CUDA_VSTD::span<_SizeType, _Np>& __indices) noexcept
    {
      _CCCL_ASSERT(__check_index(__self.__mapping_ref().extents(), __indices[_Idxs]...),
                   "cuda::std::mdspan subscript out of range!");
      const index_type __res = __self.__mapping_ref()(__indices[_Idxs]...);
      return __res;
    }
  };

public:
  //--------------------------------------------------------------------------------
  // Domain and codomain types

  using extents_type     = _Extents;
  using layout_type      = _LayoutPolicy;
  using accessor_type    = _AccessorPolicy;
  using mapping_type     = typename layout_type::template mapping<extents_type>;
  using element_type     = _ElementType;
  using value_type       = _CUDA_VSTD::remove_cv_t<element_type>;
  using index_type       = typename extents_type::index_type;
  using size_type        = typename extents_type::size_type;
  using rank_type        = typename extents_type::rank_type;
  using data_handle_type = typename accessor_type::data_handle_type;
  using reference        = typename accessor_type::reference;

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t rank() noexcept
  {
    return extents_type::rank();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t rank_dynamic() noexcept
  {
    return extents_type::rank_dynamic();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t static_extent(size_t __r) noexcept
  {
    return extents_type::static_extent(__r);
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type extent(size_t __r) const noexcept
  {
    return __mapping_ref().extents().extent(__r);
  };

private:
  // Can't use defaulted parameter in the __deduction_workaround template because of a bug in MSVC warning C4348.
  using __impl = __deduction_workaround<_CUDA_VSTD::make_index_sequence<extents_type::rank()>>;

  using __map_acc_pair_t = __detail::__compressed_pair<mapping_type, accessor_type>;

public:
  //--------------------------------------------------------------------------------
  // [mdspan.basic.cons], mdspan constructors, assignment, and destructor

#  if _CCCL_STD_VER <= 2020
  _CCCL_HIDE_FROM_ABI constexpr mdspan() = default;
#  else // ^^^ C++17 ^^^ / vvv C++20 vvv
  _CCCL_HIDE_FROM_ABI constexpr mdspan()
    requires(
              // Directly using rank_dynamic()>0 here doesn't work for nvcc
              (extents_type::rank_dynamic() > 0) && _CCCL_TRAIT(is_default_constructible, data_handle_type)
              && _CCCL_TRAIT(is_default_constructible, mapping_type)
              && _CCCL_TRAIT(is_default_constructible, accessor_type))
  = default;
#  endif // _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI constexpr mdspan(const mdspan&) = default;
  _CCCL_HIDE_FROM_ABI constexpr mdspan(mdspan&&)      = default;

  _CCCL_TEMPLATE(class... _SizeTypes)
  _CCCL_REQUIRES(__fold_and_v<_CCCL_TRAIT(is_convertible, _SizeTypes, index_type)...> _CCCL_AND
                   __fold_and_v<_CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeTypes)...> _CCCL_AND(
                     (sizeof...(_SizeTypes) == rank()) || (sizeof...(_SizeTypes) == rank_dynamic()))
                     _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, extents_type)
                       _CCCL_AND _CCCL_TRAIT(is_default_constructible, accessor_type))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr mdspan(data_handle_type __p, _SizeTypes... __dynamic_extents)
      // TODO @proposal-bug shouldn't I be allowed to do `move(__p)` here?
      : __members(
          _CUDA_VSTD::move(__p),
          __map_acc_pair_t(mapping_type(extents_type(static_cast<index_type>(_CUDA_VSTD::move(__dynamic_extents))...)),
                           accessor_type()))
  {}

  _CCCL_TEMPLATE(class _SizeType, size_t _Np)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType)
                     _CCCL_AND((_Np == rank()) || (_Np == rank_dynamic()))
                       _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, extents_type)
                         _CCCL_AND _CCCL_TRAIT(is_default_constructible, accessor_type))
  __MDSPAN_CONDITIONAL_EXPLICIT(_Np != rank_dynamic())
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p,
                                             const _CUDA_VSTD::array<_SizeType, _Np>& __dynamic_extents)
      : __members(_CUDA_VSTD::move(__p),
                  __map_acc_pair_t(mapping_type(extents_type(__dynamic_extents)), accessor_type()))
  {}

  _CCCL_TEMPLATE(class _SizeType, size_t _Np)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType)
                     _CCCL_AND((_Np == rank()) || (_Np == rank_dynamic()))
                       _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, extents_type)
                         _CCCL_AND _CCCL_TRAIT(is_default_constructible, accessor_type))
  __MDSPAN_CONDITIONAL_EXPLICIT(_Np != rank_dynamic())
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, _CUDA_VSTD::span<_SizeType, _Np> __dynamic_extents)
      : __members(_CUDA_VSTD::move(__p),
                  __map_acc_pair_t(mapping_type(extents_type(_CUDA_VSTD::as_const(__dynamic_extents))), accessor_type()))
  {}

  _CCCL_TEMPLATE(bool _Is_default_constructible = _CCCL_TRAIT(is_default_constructible, accessor_type))
  _CCCL_REQUIRES(_Is_default_constructible _CCCL_AND _CCCL_TRAIT(is_constructible, mapping_type, extents_type))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const extents_type& __exts)
      : __members(_CUDA_VSTD::move(__p), __map_acc_pair_t(mapping_type(__exts), accessor_type()))
  {}

  _CCCL_TEMPLATE(bool _Is_default_constructible = _CCCL_TRAIT(is_default_constructible, accessor_type))
  _CCCL_REQUIRES(_Is_default_constructible)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const mapping_type& __m)
      : __members(_CUDA_VSTD::move(__p), __map_acc_pair_t(__m, accessor_type()))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(data_handle_type __p, const mapping_type& __m, const accessor_type& __a)
      : __members(_CUDA_VSTD::move(__p), __map_acc_pair_t(__m, __a))
  {}

  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtents, class _OtherLayoutPolicy, class _OtherAccessor)
  _CCCL_REQUIRES(
    _CCCL_TRAIT(is_constructible, mapping_type, typename _OtherLayoutPolicy::template mapping<_OtherExtents>)
      _CCCL_AND _CCCL_TRAIT(is_constructible, accessor_type, _OtherAccessor))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mdspan(
    const mdspan<_OtherElementType, _OtherExtents, _OtherLayoutPolicy, _OtherAccessor>& __other)
      : __members(__other.__ptr_ref(), __map_acc_pair_t(__other.__mapping_ref(), __other.__accessor_ref()))
  {
    static_assert(_CCCL_TRAIT(is_constructible, data_handle_type, typename _OtherAccessor::data_handle_type),
                  "Incompatible data_handle_type for mdspan construction");
    static_assert(_CCCL_TRAIT(is_constructible, extents_type, _OtherExtents),
                  "Incompatible extents for mdspan construction");
    /*
     * TODO: Check precondition
     * For each rank index __r of extents_type, static_extent(__r) == dynamic_extent || static_extent(__r) ==
     * __other.extent(__r) is true.
     */
  }

  /* Might need this on NVIDIA?
  _CCCL_HIDE_FROM_ABI
  _CCCL_HIDE_FROM_ABI ~mdspan() = default;
  */

  _CCCL_HIDE_FROM_ABI __MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(const mdspan&) = default;
  _CCCL_HIDE_FROM_ABI __MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(mdspan&&)      = default;

  //--------------------------------------------------------------------------------
  // [mdspan.basic.mapping], mdspan mapping domain multidimensional index to access codomain element

#  if __MDSPAN_USE_BRACKET_OPERATOR
  _CCCL_TEMPLATE(class... _SizeTypes)
  _CCCL_REQUIRES(__fold_and_v<_CCCL_TRAIT(is_convertible, _SizeTypes, index_type)...> _CCCL_AND
                   __fold_and_v<_CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeTypes)...> _CCCL_AND(
                     rank() == sizeof...(_SizeTypes)))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](_SizeTypes... __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices...));
  }
#  endif // __MDSPAN_USE_BRACKET_OPERATOR

  _CCCL_TEMPLATE(class _SizeType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator[](const _CUDA_VSTD::array<_SizeType, rank()>& __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices));
  }

  _CCCL_TEMPLATE(class _SizeType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator[](_CUDA_VSTD::span<_SizeType, rank()> __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices));
  }

#  if !__MDSPAN_USE_BRACKET_OPERATOR
  _CCCL_TEMPLATE(class _Index)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _Index, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _Index)
                     _CCCL_AND(extents_type::rank() == 1))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator[](_Index __idx) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __idx));
  }
#  endif // !__MDSPAN_USE_BRACKET_OPERATOR

#  if __MDSPAN_USE_PAREN_OPERATOR
  _CCCL_TEMPLATE(class... _SizeTypes)
  _CCCL_REQUIRES(__fold_and_v<_CCCL_TRAIT(is_convertible, _SizeTypes, index_type)...> _CCCL_AND
                   __fold_and_v<_CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeTypes)...> _CCCL_AND(
                     extents_type::rank() == sizeof...(_SizeTypes)))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference operator()(_SizeTypes... __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices...));
  }

  _CCCL_TEMPLATE(class _SizeType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator()(const _CUDA_VSTD::array<_SizeType, rank()>& __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices));
  }

  _CCCL_TEMPLATE(class _SizeType)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _SizeType, index_type)
                   _CCCL_AND _CCCL_TRAIT(is_nothrow_constructible, index_type, _SizeType))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr reference
  operator()(_CUDA_VSTD::span<_SizeType, rank()> __indices) const
  {
    return __accessor_ref().access(__ptr_ref(), __impl::__index(*this, __indices));
  }
#  endif // __MDSPAN_USE_PAREN_OPERATOR

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t size() const noexcept
  {
    return __impl::__size(*this);
  };

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool empty() const noexcept
  {
    return __impl::__empty(*this);
  };

  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr void swap(mdspan& __x, mdspan& __y) noexcept
  {
    swap(__x.__ptr_ref(), __y.__ptr_ref());
    swap(__x.__mapping_ref(), __y.__mapping_ref());
    swap(__x.__accessor_ref(), __y.__accessor_ref());
  }

  //--------------------------------------------------------------------------------
  // [mdspan.basic.domobs], mdspan observers of the domain multidimensional index space

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
  {
    return __mapping_ref().extents();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const data_handle_type& data_handle() const noexcept
  {
    return __ptr_ref();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const mapping_type& mapping() const noexcept
  {
    return __mapping_ref();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const accessor_type& accessor() const noexcept
  {
    return __accessor_ref();
  };

  //--------------------------------------------------------------------------------
  // [mdspan.basic.obs], mdspan observers of the mapping

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_unique() noexcept
  {
    return mapping_type::is_always_unique();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_exhaustive() noexcept
  {
    return mapping_type::is_always_exhaustive();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_strided() noexcept
  {
    return mapping_type::is_always_strided();
  };

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_unique() const noexcept
  {
    return __mapping_ref().is_unique();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_exhaustive() const noexcept
  {
    return __mapping_ref().is_exhaustive();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_strided() const noexcept
  {
    return __mapping_ref().is_strided();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type stride(size_t __r) const
  {
    return __mapping_ref().stride(__r);
  };

private:
  __detail::__compressed_pair<data_handle_type, __map_acc_pair_t> __members{};

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type& __ptr_ref() noexcept
  {
    return __members.__first();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type const& __ptr_ref() const noexcept
  {
    return __members.__first();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping_type& __mapping_ref() noexcept
  {
    return __members.__second().__first();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping_type const& __mapping_ref() const noexcept
  {
    return __members.__second().__first();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr accessor_type& __accessor_ref() noexcept
  {
    return __members.__second().__second();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr accessor_type const& __accessor_ref() const noexcept
  {
    return __members.__second().__second();
  }

  template <class, class, class, class>
  friend class mdspan;
};

#  if defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
_CCCL_TEMPLATE(class _ElementType, class... _SizeTypes)
_CCCL_REQUIRES(__fold_and_v<_CCCL_TRAIT(is_integral, _SizeTypes)...> _CCCL_AND(sizeof...(_SizeTypes) > 0))
_CCCL_HOST_DEVICE explicit mdspan(_ElementType*,
                                  _SizeTypes...) -> mdspan<_ElementType, dextents<size_t, sizeof...(_SizeTypes)>>;

_CCCL_TEMPLATE(class _Pointer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_pointer, _CUDA_VSTD::remove_reference_t<_Pointer>))
_CCCL_HOST_DEVICE
mdspan(_Pointer&&) -> mdspan<_CUDA_VSTD::remove_pointer_t<_CUDA_VSTD::remove_reference_t<_Pointer>>, extents<size_t>>;
_CCCL_TEMPLATE(class _CArray)
_CCCL_REQUIRES(_CCCL_TRAIT(is_array, _CArray) _CCCL_AND(rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE mdspan(_CArray&)
  -> mdspan<_CUDA_VSTD::remove_all_extents_t<_CArray>, extents<size_t, _CUDA_VSTD::extent_v<_CArray, 0>>>;

template <class _ElementType, class _SizeType, size_t _Np>
_CCCL_HOST_DEVICE mdspan(_ElementType*,
                         const _CUDA_VSTD::array<_SizeType, _Np>&) -> mdspan<_ElementType, dextents<size_t, _Np>>;

template <class _ElementType, class _SizeType, size_t _Np>
_CCCL_HOST_DEVICE mdspan(_ElementType*, _CUDA_VSTD::span<_SizeType, _Np>) -> mdspan<_ElementType, dextents<size_t, _Np>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <class _ElementType, class _SizeType, size_t... _ExtentsPack>
_CCCL_HOST_DEVICE mdspan(_ElementType*, const extents<_SizeType, _ExtentsPack...>&)
  -> mdspan<_ElementType, extents<_SizeType, _ExtentsPack...>>;

template <class _ElementType, class _MappingType>
_CCCL_HOST_DEVICE mdspan(_ElementType*, const _MappingType&)
  -> mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <class _MappingType, class _AccessorType>
_CCCL_HOST_DEVICE mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> mdspan<typename _AccessorType::element_type,
            typename _MappingType::extents_type,
            typename _MappingType::layout_type,
            _AccessorType>;
#  endif // __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION

_CCCL_NV_DIAG_DEFAULT(186)

#endif // _CCCL_STD_VER >= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_MDSPAN_HPP
