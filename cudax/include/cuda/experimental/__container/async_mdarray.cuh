//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_ASYNC_MDARRAY
#define __CUDAX__CONTAINERS_ASYNC_MDARRAY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>

#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/equal.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/temporary_buffer.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/unwrap_end.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/stdexcept>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>

#include <cuda/experimental/__container/heterogeneous_iterator.cuh>
#include <cuda/experimental/__container/uninitialized_async_buffer.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/get_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__stream/get_stream.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>
#include <cuda/experimental/__utility/select_execution_space.cuh>

_CCCL_PUSH_MACROS

//! @file The \c async_mdarray class provides a static N-dimensional range of contiguous memory
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-async-vector:
//!
//! async_mdarray
//! -------------
//!
//! ``async_mdarray`` is a container that provides statically sized N-dimensional typed storage allocated from a given
//! :ref:`memory resource <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment, release and growth
//! of the allocation. The elements are initialized during construction, which may require a kernel launch.
//!
//! In addition to being type-safe, ``async_mdarray`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const async_mdarray&, Property) <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _ElementType the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _ElementType, class _Extents, class _LayoutPolicy = _CUDA_VSTD::layout_right, class... _Properties>
class async_mdarray
{
public:
  using extents_type = _Extents;
  using layout_type  = _LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  using element_type           = _ElementType;
  using value_type             = _CUDA_VSTD::remove_cv_t<_ElementType>;
  using pointer                = _ElementType*;
  using const_pointer          = const _ElementType*;
  using reference              = _ElementType&;
  using const_reference        = const _ElementType&;
  using iterator               = heterogeneous_iterator<_ElementType, false, _Properties...>;
  using const_iterator         = heterogeneous_iterator<_ElementType, true, _Properties...>;
  using reverse_iterator       = _CUDA_VSTD::reverse_iterator<iterator>;
  using const_reverse_iterator = _CUDA_VSTD::reverse_iterator<const_iterator>;

  using index_type      = typename extents_type::index_type;
  using size_type       = typename extents_type::size_type;
  using rank_type       = typename extents_type::rank_type;
  using difference_type = ptrdiff_t;

  using __env_t          = ::cuda::experimental::env_t<_Properties...>;
  using __policy_t       = ::cuda::experimental::execution::execution_policy;
  using __buffer_t       = ::cuda::experimental::uninitialized_async_buffer<_ElementType, _Properties...>;
  using __resource_t     = ::cuda::experimental::any_async_resource<_Properties...>;
  using __resource_ref_t = _CUDA_VMR::async_resource_ref<_Properties...>;

  // TODO: use an accessor that has host / device guardrails
  template <class _OtherElementType,
            class _OtherExtent,
            class _OtherLayout,
            class _OtherAccessorType = _CUDA_VSTD::default_accessor<_ElementType>>
  using __mdspan_t = _CUDA_VSTD::mdspan<_OtherElementType, _OtherExtent, _OtherLayout, _OtherAccessorType>;

  template <class, class, class, class...>
  friend class async_mdarray;

  // For now we require trivially copyable type to simplify the implementation
  static_assert(_CCCL_TRAIT(_CUDA_VSTD::is_trivially_copyable, _ElementType),
                "cuda::experimental::async_mdarray requires T to be trivially copyable.");

  static_assert(_CUDA_VSTD::__detail::__is_extents_v<_Extents>,
                "mdspan's Extents template parameter must be a specialization of _CUDA_VSTD::extents.");

  // At least one of the properties must signal an execution space
  static_assert(_CUDA_VMR::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::async_mdarray must contain at least one execution space "
                "property!");

  //! @brief Convenience shortcut to detect the execution space of the async_mdarray
  static constexpr bool __is_host_only = __select_execution_space<_Properties...> == _ExecutionSpace::__host;

private:
  mapping_type __mapping_{};
  __buffer_t __buf_{};
  __policy_t __policy_ = __policy_t::invalid_execution_policy;

  //! @brief Helper to check container is compatible with this async_mdarray
  template <class _Range>
  static constexpr bool __compatible_range = _CUDA_VRANGES::__container_compatible_range<_Range, _ElementType>;

  //! @brief Helper to check whether a different async_mdarray still statisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    !_CCCL_TRAIT(_CUDA_VSTD::is_same,
                 _CUDA_VSTD::__make_type_set<_Properties...>,
                 _CUDA_VSTD::__make_type_set<_OtherProperties...>)
    && _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Helper to determine what cudaMemcpyKind we need to copy data from another async_mdarray with different
  template <class... _OtherProperties>
  static constexpr cudaMemcpyKind __transfer_kind =
    __select_execution_space<_OtherProperties...> == _ExecutionSpace::__host
      ? (__is_host_only ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice)
      : (__is_host_only ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);

  //! @brief Helper to return an async_resource_ref to the currently used resource. Used in case we need to replace the
  //! underlying allocation
  __resource_ref_t __borrow_resource() const noexcept
  {
    return const_cast<__resource_t&>(__buf_.get_memory_resource());
  }

  //! @brief Copies data from `[__first, __last)` to \p __dest, where \p __first and \p __dest reside in the same memory
  //! space
  //! @param __first Pointer to the start of the input segment.
  //! @param __last Pointer to the end of the input segment.
  //! @param __dest Pointer to the start of the output segment.
  //! @returns Pointer equal to `__dest + __count`
  _CCCL_HIDE_FROM_ABI pointer __copy_same(const_pointer __first, const_pointer __last, pointer __dest)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      return _CUDA_VSTD::copy(__first, __last, __dest);
    }
    else
    {
      const auto __count = static_cast<size_t>(__last - __first);
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_mdarray::__copy_same: failed to copy data",
        __dest,
        __first,
        sizeof(_ElementType) * __count,
        ::cudaMemcpyDeviceToDevice,
        __buf_.get_stream().get());
      return __dest + __count;
    }
    _CCCL_UNREACHABLE();
  }

  //! @brief Copies \p __count elements from `[__first, __last)` to \p __dest, where \p __first and \p __dest reside in
  //! the different memory spaces
  //! @param __first Pointer to the start of the input segment.
  //! @param __last Pointer to the end of the input segment.
  //! @param __dest Pointer to the start of the output segment.
  //! @param __count The number of elements to be copied.
  template <class _Iter, cudaMemcpyKind __kind = __is_host_only ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice>
  _CCCL_HIDE_FROM_ABI void __copy_cross(_Iter __first, _Iter __last, pointer __dest, size_type __count)
  {
    _CCCL_IF_CONSTEXPR (__kind == cudaMemcpyHostToHost)
    {
      _CUDA_VSTD::copy(__first, __last, __dest);
    }
    else _CCCL_IF_CONSTEXPR (!_CUDA_VSTD::contiguous_iterator<_Iter>)
    { // For non-contiguous iterators we need to copy into temporary host storage to use cudaMemcpy
      // This should only ever happen when passing in data from host to device
      _CCCL_ASSERT(__kind == cudaMemcpyHostToDevice, "Invalid use case!");
      auto __temp = _CUDA_VSTD::get_temporary_buffer<_ElementType>(__count).first;
      _CUDA_VSTD::copy(__first, __last, __temp);
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_mdarray::__copy_cross: failed to copy data",
        __dest,
        __temp,
        sizeof(_ElementType) * __count,
        __kind,
        __buf_.get_stream().get());
      _CUDA_VSTD::return_temporary_buffer(__temp);
    }
    else
    {
      (void) __last;
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_mdarray::__copy_cross: failed to copy data",
        __dest,
        _CUDA_VSTD::to_address(__first),
        sizeof(_ElementType) * __count,
        __kind,
        __buf_.get_stream().get());
    }
  }

  //! @brief Copy-constructs elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __fill_n(pointer __first, size_type __count, const _ElementType& __value)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      _CUDA_VSTD::fill_n(__first, __count, __value);
    }
    else
    {
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __count, __value);
    }
  }

  //! @brief Equality-compares elements from the sequence `[__first1, __last1)` with those of sequence
  //! `[__first2, __last2)`.
  //! @param __first1 Pointer to the start of the first sequence.
  //! @param __last1 Pointer to the end of the first sequence.
  //! @param __first2 Pointer to the start of the second sequence.
  //! @param __last2 Pointer to the end of the second sequence.
  _CCCL_HIDE_FROM_ABI bool
  __equality(const_pointer __first1, const_pointer __last1, const_pointer __first2, const_pointer __last2) const
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      return _CUDA_VSTD::equal(__first1, __last1, __first2, __last2);
    }
    else
    {
      return ((__last1 - __first1) == (__last2 - __first2))
          && thrust::equal(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first1, __last1, __first2);
    }
    _CCCL_UNREACHABLE();
  }

public:
  //! @addtogroup construction
  //! @{

  //! @brief Copy-constructs a async_mdarray
  //! @param __other The other async_mdarray.
  //! The new async_mdarray has capacity of \p __other.size() which is potentially less than \p __other.capacity().
  //! @note No memory is allocated if \p __other is empty
  _CCCL_HIDE_FROM_ABI async_mdarray(const async_mdarray& __other)
      : __mapping_(__other.__mapping_)
      , __buf_(__other.get_memory_resource(), __other.get_stream(), __other.size())
      , __policy_(__other.__policy_)
  {
    if (__other.size() != 0)
    {
      this->__copy_same(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
    }
  }

  //! @brief Move-constructs a async_mdarray
  //! @param __other The other async_mdarray.
  //! The new async_mdarray takes ownership of the allocation of \p __other and resets it.
  _CCCL_HIDE_FROM_ABI async_mdarray(async_mdarray&& __other) noexcept
      : __mapping_(_CUDA_VSTD::exchange(__other.__mapping_, mapping_type{}))
      , __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Copy-constructs from a async_mdarray with matching properties
  //! @param __other The other async_mdarray.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_mdarray(const async_mdarray<_ElementType, _OtherProperties...>& __other)
      : __mapping_(__other.__mapping_)
      , __buf_(__other.get_memory_resource(), __other.get_stream(), __other.size())
      , __policy_(__other.__policy_)
  {
    if (__other.size() != 0)
    {
      this->__copy_cross<const_pointer, __transfer_kind<_OtherProperties...>>(
        __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.size());
    }
  }

  //! @brief Move-constructs from a async_mdarray with matching properties
  //! @param __other The other async_mdarray.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_mdarray(async_mdarray<_ElementType, _OtherProperties...>&& __other) noexcept
      : __mapping_(_CUDA_VSTD::exchange(__other.__mapping_, mapping_type{}))
      , __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Constructs an empty async_mdarray using an environment
  //! @param __env The environment providing the needed information
  _CCCL_HIDE_FROM_ABI async_mdarray(const __env_t& __env)
      : async_mdarray(__env, extents_type{})
  {}

  //! @brief Constructs a async_mdarray of size \p __size using a memory resource and value-initializes \p __size
  //! elements
  //! @param __mr The memory resource to allocate the async_mdarray with.
  //! @param __size The size of the async_mdarray. Defaults to zero
  _CCCL_TEMPLATE(class... _IndexTypes)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, extents_type, _IndexTypes...))
  _CCCL_HIDE_FROM_ABI explicit async_mdarray(const __env_t& __env, _IndexTypes... __extents)
      : __mapping_(extents_type{__extents...})
      , __buf_(__env.query(::cuda::experimental::get_memory_resource),
               __env.query(::cuda::experimental::get_stream),
               __mapping_.required_span_size())
      , __policy_(__env.query(::cuda::experimental::execution::get_execution_policy))
  {
    const auto __size = __mapping_.required_span_size();
    if (__size != 0)
    {
      // miscco: should implement parallel specialized memory algorithms
      this->__fill_n(__unwrapped_begin(), __size, _ElementType());
    }
  }

  //! @brief Constructs a async_mdarray of size \p __size using a memory resource and copy-constructs \p __size elements
  //! from \p __value
  //! @param __mr The memory resource to allocate the async_mdarray with.
  //! @param __size The size of the async_mdarray.
  //! @param __value The value all elements are copied from.
  _CCCL_HIDE_FROM_ABI explicit async_mdarray(
    const __env_t& __env, const extents_type& __extent, const _ElementType& __value = _ElementType())
      : async_mdarray(__env, __extent, ::cuda::experimental::uninit)
  {
    if (__mapping_.required_span_size() != 0)
    {
      this->__fill_n(__unwrapped_begin(), __mapping_.required_span_size(), __value);
    }
  }

  //! @brief Constructs a async_mdarray of size \p __size using a memory and leaves all elements uninitialized
  //! @param __mr The memory resource to allocate the async_mdarray with.
  //! @param __size The size of the async_mdarray.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[0, mapping.required_span_size())` are properly initialized, e.g with
  //! `cuda::std::uninitialized_copy`. At the destruction of the \c async_mdarray all elements in the range
  //! `[0, mapping.required_span_size())` will be destroyed.
  _CCCL_HIDE_FROM_ABI explicit async_mdarray(
    const __env_t& __env, const extents_type& __extent, ::cuda::experimental::uninit_t)
      : __mapping_(__extent)
      , __buf_(__env.query(::cuda::experimental::get_memory_resource),
               __env.query(::cuda::experimental::get_stream),
               __mapping_.required_span_size())
      , __policy_(__env.query(::cuda::experimental::execution::get_execution_policy))
  {}

  //! @brief Constructs a async_mdarray using a memory resource and copy-constructs all elements from \p __ilist
  //! @param __mr The memory resource to allocate the async_mdarray with.
  //! @param __ilist The initializer_list being copied into the async_mdarray.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  _CCCL_HIDE_FROM_ABI
  async_mdarray(const __env_t& __env, const extents_type& __extent, _CUDA_VSTD::initializer_list<_ElementType> __ilist)
      : async_mdarray(__env, __extent, ::cuda::experimental::uninit)
  {
    const auto __size = __mapping_.required_span_size();
    _CCCL_ASSERT(__size == __ilist.size(),
                 "cuda::experimental::async_mdarray: Construction with initializer_list of wrong size");
    if (__mapping_.required_span_size() > 0)
    {
      this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __size);
    }
  }

  //! @brief Constructs a async_mdarray using a memory resource and an input range
  //! @param __mr The memory resource to allocate the async_mdarray with.
  //! @param __range The input range to be moved into the async_mdarray.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND
                   _CUDA_VRANGES::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI async_mdarray(const __env_t& __env, const extents_type& __extent, _Range&& __range)
      : async_mdarray(__env, __extent, ::cuda::experimental::uninit)
  {
    const auto __size  = __mapping_.required_span_size();
    const auto __rsize = static_cast<size_type>(_CUDA_VRANGES::size(__range));
    _CCCL_ASSERT(__size == __rsize, "cuda::experimental::async_mdarray: Construction with range of wrong size");
    if (__size > 0)
    {
      using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
      this->__copy_cross<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size);
    }
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI async_mdarray(const __env_t& __env, const extents_type& __extent, _Range&& __range)
      : async_mdarray(__env, __extent, ::cuda::experimental::uninit)
  {
    const auto __size = __mapping_.required_span_size();
    const auto __rsize =
      static_cast<size_type>(_CUDA_VRANGES::distance(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range)));
    _CCCL_ASSERT(__size == __rsize, "cuda::experimental::async_mdarray: Construction with range of wrong size");
    if (__size > 0)
    {
      using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
      this->__copy_cross<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size);
    }
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup assignment
  //! @{

  //! @brief Copy-assigns a async_mdarray
  //! @param __other The other async_mdarray.
  //! @note Even if the old async_mdarray would have enough storage available, we may have to reallocate if the stored
  //! memory resource is not equal to the new one. In that case no memory is allocated if \p __other is empty.
  _CCCL_HIDE_FROM_ABI async_mdarray& operator=(const async_mdarray& __other)
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    __other.wait();
    if ((__borrow_resource() != __resource_ref_t(__other.__borrow_resource()) || (size() != __other.size())))
    {
      __buffer_t __new_buf{__other.get_memory_resource(), __other.get_stream(), __other.size()};
      _CUDA_VSTD::swap(__buf_, __new_buf);
    }

    __mapping_ = __other.__mapping_;
    this->__copy_same(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
    __policy_ = __other.__policy_;
    return *this;
  }

  //! @brief Move-assigns a async_mdarray
  //! @param __other The other async_mdarray.
  //! Clears the async_mdarray and swaps the contents with \p __other.
  _CCCL_HIDE_FROM_ABI async_mdarray& operator=(async_mdarray&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    __other.wait();
    __buf_     = _CUDA_VSTD::move(__other.__buf_);
    __mapping_ = _CUDA_VSTD::exchange(__other.__mapping_, mapping_type());
    __policy_  = _CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy);
    return *this;
  }

  //! @brief Copy-assigns from a different async_mdarray
  //! @param __other The other async_mdarray.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI async_mdarray& operator=(const async_mdarray<_ElementType, _OtherProperties...>& __other)
  {
    if (this == reinterpret_cast<const async_mdarray*>(_CUDA_VSTD::addressof(__other)))
    {
      return *this;
    }

    __other.wait();
    const auto __count = __other.size();
    if ((__borrow_resource() != __resource_ref_t(__other.__borrow_resource()) || (size() != __other.size())))
    {
      __buffer_t __new_buf{__other.get_memory_resource(), __other.get_stream(), __count};
      _CUDA_VSTD::swap(__buf_, __new_buf);
    }

    __mapping_ = __other.__mapping_;
    this->__copy_cross<const_pointer, __transfer_kind<_OtherProperties...>>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __count);
    __policy_ = __other.__policy_;
    return *this;
  }

  //! @brief Move-assigns from a different async_mdarray
  //! @param __other The other async_mdarray.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI async_mdarray& operator=(async_mdarray<_ElementType, _OtherProperties...>&& __other)
  {
    if (this == reinterpret_cast<async_mdarray*>(_CUDA_VSTD::addressof(__other)))
    {
      return *this;
    }

    __other.wait();
    __buf_     = _CUDA_VSTD::move(__other.__buf_);
    __mapping_ = _CUDA_VSTD::exchange(__other.__mapping_, mapping_type());
    __policy_  = _CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy);
    return *this;
  }

  //! @brief Assigns an initializer_list to a async_mdarray, replacing its content with that of the initializer_list
  //! @param __ilist The initializer_list to be assigned
  _CCCL_HIDE_FROM_ABI async_mdarray& operator=(_CUDA_VSTD::initializer_list<_ElementType> __ilist)
  {
    const auto __count = __ilist.size();
    if (size() != __count)
    {
      __buffer_t __new_buf{get_memory_resource(), get_stream(), __count};
      _CUDA_VSTD::swap(__buf_, __new_buf);
    }

    __mapping_ = mapping_type{_CUDA_VSTD::dims<1>{__count}};
    this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __count);
    return *this;
  }

  //! @}

  //! @addtogroup conversions
  _CCCL_HIDE_FROM_ABI
  operator __mdspan_t<_ElementType, _Extents, _LayoutPolicy, _CUDA_VSTD::default_accessor<_ElementType>>() noexcept
  {
    return {data(), __mapping_};
  }

  _CCCL_HIDE_FROM_ABI
  operator __mdspan_t<const _ElementType, _Extents, _LayoutPolicy, _CUDA_VSTD::default_accessor<const _ElementType>>() noexcept
  {
    return {data(), __mapping_};
  }

  _CCCL_TEMPLATE(class _OtherElementType, class _OtherExtent, class _OtherLayout, class _OtherAccessorType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_assignable,
                             __mdspan_t<_ElementType, _Extents, _LayoutPolicy>,
                             __mdspan_t<_OtherElementType, _OtherExtent, _OtherLayout, _OtherAccessorType>))
  _CCCL_HIDE_FROM_ABI operator __mdspan_t<_OtherElementType, _OtherExtent, _OtherLayout, _OtherAccessorType>() noexcept
  {
    return __mdspan_t<_OtherElementType, _OtherExtent, _OtherLayout, _OtherAccessorType>(data(), __mapping_);
  }

  _CCCL_HIDE_FROM_ABI __mdspan_t<_ElementType, _Extents, _LayoutPolicy, _CUDA_VSTD::default_accessor<_ElementType>>
  view(const _CUDA_VSTD::default_accessor<_ElementType>& __accessor = {}) noexcept
  {
    return {data(), __mapping_, __accessor};
  }

  _CCCL_HIDE_FROM_ABI
  __mdspan_t<const _ElementType, _Extents, _LayoutPolicy, _CUDA_VSTD::default_accessor<const _ElementType>>
  view(const _CUDA_VSTD::default_accessor<const _ElementType>& __accessor = {}) const noexcept
  {
    return {data(), __mapping_, __accessor};
  }

  _CCCL_TEMPLATE(class _OtherAccessorType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_assignable,
                             __mdspan_t<_ElementType, _Extents, _LayoutPolicy>,
                             __mdspan_t<_ElementType, _Extents, _LayoutPolicy, _OtherAccessorType>))
  _CCCL_HIDE_FROM_ABI __mdspan_t<_ElementType, _Extents, _LayoutPolicy, _OtherAccessorType>
  view(const _OtherAccessorType& __accessor) noexcept
  {
    return __mdspan_t<_ElementType, _Extents, _LayoutPolicy, _OtherAccessorType>(data(), __mapping_, __accessor);
  }

  _CCCL_TEMPLATE(class _OtherAccessorType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_same, typename _OtherAccessorType::element_type, const element_type))
  _CCCL_HIDE_FROM_ABI __mdspan_t<const _ElementType, _Extents, _LayoutPolicy, _OtherAccessorType>
  view(const _OtherAccessorType& __accessor) const noexcept
  {
    return __mdspan_t<const _ElementType, _Extents, _LayoutPolicy, _OtherAccessorType>(data(), __mapping_, __accessor);
  }

  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the async_mdarray. If the async_mdarray is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_mdarray. If the async_mdarray is empty, the
  //! returned iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_mdarray. If the async_mdarray is empty, the
  //! returned iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of the async_mdarray. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator end() noexcept
  {
    return iterator{__buf_.data() + size()};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_mdarray. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + size()};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_mdarray. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + size()};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed async_mdarray. It corresponds to the last
  //! element of the non-reversed async_mdarray. If the async_mdarray is empty, the returned iterator is equal to
  //! rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_mdarray. It corresponds to
  //! the last element of the non-reversed async_mdarray. If the async_mdarray is empty, the returned iterator is equal
  //! to rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_mdarray. It corresponds to
  //! the last element of the non-reversed async_mdarray. If the async_mdarray is empty, the returned iterator is equal
  //! to rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed async_mdarray. It
  //! corresponds to the element preceding the first element of the non-reversed async_mdarray. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_mdarray. It corresponds to the element preceding the first element of the non-reversed async_mdarray. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_mdarray. It corresponds to the element preceding the first element of the non-reversed async_mdarray. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the async_mdarray. If the async_mdarray has not allocated memory
  //! the pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the async_mdarray. If the async_mdarray has not allocated memory
  //! the pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer data() const noexcept
  {
    return __buf_.data();
  }

#ifndef _CCCL_DOXYGEN_INVOKED
  //! @brief Returns a pointer to the first element of the async_mdarray. If the async_mdarray is empty, the returned
  //! pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the async_mdarray. If the async_mdarray is empty, the
  //! returned pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the async_mdarray. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + size();
  }

  //! @brief Returns a const pointer to the element following the last element of the async_mdarray. This element acts
  //! as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + size();
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @}

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the async_mdarray
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reference operator[](const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < size(), "cuda::experimental::async_mdarray subscript out of range!");
    return begin()[__mapping_(__n)];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_mdarray
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reference operator[](const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < size(), "cuda::experimental::async_mdarray subscript out of range!");
    return begin()[__mapping_(__n)];
  }

  //! @}

  //! @addtogroup size
  //! @{
  //! @brief Returns the current number of elements stored in the async_mdarray.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI size_type size() const noexcept
  {
    return __mapping_.required_span_size();
  }

  //! @brief Returns true if the async_mdarray is empty.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI bool empty() const noexcept
  {
    return size() == 0;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
  {
    return __mapping_.extents();
  };
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const mapping_type& mapping() const noexcept
  {
    return __mapping_;
  };

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource <cudax-memory-resource-any-resource>`
  //! that holds the memory resource used to allocate the async_mdarray
  //! @endrst
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const __resource_t& get_memory_resource() const noexcept
  {
    return __buf_.get_memory_resource();
  }

  //! @brief Returns the stored stream
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref get_stream() const noexcept
  {
    return __buf_.get_stream();
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  _CCCL_HIDE_FROM_ABI constexpr void change_stream(::cuda::stream_ref __new_stream)
  {
    __buf_.change_stream(__new_stream);
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @warning This does not synchronize between \p __new_stream and the current stream. It is the user's responsibility
  //! to ensure proper stream order going forward
  _CCCL_HIDE_FROM_ABI constexpr void change_stream_unsynchronized(::cuda::stream_ref __new_stream) noexcept
  {
    __buf_.change_stream_unsynchronized(__new_stream);
  }

  //! @brief Returns the execution policy
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr __policy_t get_execution_policy() const noexcept
  {
    return __policy_;
  }

  //! @brief Replaces the currently used execution policy
  //! @param __new_policy the new policy
  _CCCL_HIDE_FROM_ABI constexpr void set_execution_policy(__policy_t __new_policy) noexcept
  {
    __policy_ = __new_policy;
  }

  //! @brief Synchronizes with the stored stream
  _CCCL_HIDE_FROM_ABI void wait() const
  {
    __buf_.get_stream().wait();
  }
  //! @}

  //! @brief Swaps the contents of a async_mdarray with those of \p __other
  //! @param __other The other async_mdarray.
  _CCCL_HIDE_FROM_ABI void swap(async_mdarray& __other) noexcept
  {
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
    _CUDA_VSTD::swap(size(), __other.size());
  }

  //! @brief Swaps the contents of two async_vectors
  //! @param __lhs One async_mdarray.
  //! @param __rhs The other async_mdarray.
  _CCCL_HIDE_FROM_ABI friend void swap(async_mdarray& __lhs, async_mdarray& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }
  //! @}

#ifndef _CCCL_DOXYGEN_INVOKED // friend functions are currently broken
  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND _CUDA_VSTD::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend void get_property(const async_mdarray&, _Property) noexcept {}
#endif // _CCCL_DOXYGEN_INVOKED
};

template <class _ElementType, class _Layout = _CUDA_VSTD::layout_right>
using device_mdarray = async_mdarray<_ElementType, _Layout, _CUDA_VMR::device_accessible>;

template <class _ElementType, class _Layout = _CUDA_VSTD::layout_right>
using host_mdarray = async_mdarray<_ElementType, _Layout, _CUDA_VMR::host_accessible>;

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif //__CUDAX__CONTAINERS_ASYNC_MDARRAY
