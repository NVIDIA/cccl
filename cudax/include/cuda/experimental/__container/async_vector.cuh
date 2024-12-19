//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_ASYNC_VECTOR__
#define __CUDAX__CONTAINER_ASYNC_VECTOR__

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
#include <thrust/remove.h>

#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/equal.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/lexicographical_compare.h>
#include <cuda/std/__algorithm/move_backward.h>
#include <cuda/std/__algorithm/remove.h>
#include <cuda/std/__algorithm/remove_if.h>
#include <cuda/std/__algorithm/rotate.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/temporary_buffer.h>
#include <cuda/std/__memory/uninitialized_algorithms.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/unwrap_end.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
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
#include <cuda/experimental/__utility/select_execution_space.cuh>

_CCCL_PUSH_MACROS

//! @file The \c async_vector class provides a container of contiguous memory
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-async-vector:
//!
//! async_vector
//! -------------
//!
//! ``async_vector`` is a container that provides resizable typed storage allocated from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment, release and growth of the allocation.
//! The elements are initialized during construction, which may require a kernel launch.
//!
//! In addition to being type-safe, ``async_vector`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const async_vector&, Property) <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class async_vector
{
public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, false, _Properties...>;
  using const_iterator         = heterogeneous_iterator<_Tp, true, _Properties...>;
  using reverse_iterator       = _CUDA_VSTD::reverse_iterator<iterator>;
  using const_reverse_iterator = _CUDA_VSTD::reverse_iterator<const_iterator>;
  using size_type              = _CUDA_VSTD::size_t;
  using difference_type        = _CUDA_VSTD::ptrdiff_t;

  using __env_t          = ::cuda::experimental::env_t<_Properties...>;
  using __policy_t       = ::cuda::experimental::execution::execution_policy;
  using __buffer_t       = ::cuda::experimental::uninitialized_async_buffer<_Tp, _Properties...>;
  using __resource_t     = ::cuda::experimental::any_async_resource<_Properties...>;
  using __resource_ref_t = _CUDA_VMR::async_resource_ref<_Properties...>;

  template <class, class...>
  friend class async_vector;

  // For now we require trivially copyable type to simplify the implementation
  static_assert(_CCCL_TRAIT(_CUDA_VSTD::is_trivially_copyable, _Tp),
                "cuda::experimental::async_vector requires T to be trivially copyable.");

  // At least one of the properties must signal an execution space
  static_assert(_CUDA_VMR::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::async_vector must contain at least one execution space "
                "property!");

  //! @brief Convenience shortcut to detect the execution space of the async_vector
  static constexpr bool __is_host_only = __select_execution_space<_Properties...> == _ExecutionSpace::__host;

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  //! @brief Copies range [__src, ...) into __dest with a hole of __count elements after __pos
  //! [__src, ..., __src + __pos, undefined, ..., undefined, __src + __pos, ...)
  //! Needs to be public for thrust to use it
  struct __copy_segmented_op
  {
    const _Tp* __src_;
    _Tp* __dest_;
    size_type __pos_;
    size_type __count_;

    _CCCL_HIDE_FROM_ABI constexpr __copy_segmented_op(
      const _Tp* __src, _Tp* __dest, size_type __pos, size_type __count) noexcept
        : __src_(__src)
        , __dest_(__dest)
        , __pos_(__pos)
        , __count_(__count)
    {}

    _CCCL_DEVICE _CCCL_HIDE_FROM_ABI void operator()(_Tp& __elem) const noexcept
    {
      const auto __idx = _CUDA_VSTD::addressof(__elem) - __dest_;
      if (__idx < __pos_) // copy first __pos elements from __src
      {
        __elem = *(__src_ + __idx);
      }
      else if (__idx >= __pos_ + __count_) // copy final elements from __src again
      {
        __elem = *(__src_ + __idx - __count_);
      }
    }
  };
#endif // _CCCL_DOXYGEN_INVOKED

private:
  __buffer_t __buf_;
  size_type __size_    = 0; // initialized to 0 in case initialization of the elements might throw
  __policy_t __policy_ = __policy_t::invalid_execution_policy;

  //! @brief Helper to check container is compatible with this async_vector
  template <class _Range>
  static constexpr bool __compatible_range = _CUDA_VRANGES::__container_compatible_range<_Range, _Tp>;

  //! @brief Helper to check whether a different async_vector still statisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    !_CCCL_TRAIT(_CUDA_VSTD::is_same,
                 _CUDA_VSTD::__make_type_set<_Properties...>,
                 _CUDA_VSTD::__make_type_set<_OtherProperties...>)
    && _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Helper to determine what cudaMemcpyKind we need to copy data from another async_vector with different
  template <class... _OtherProperties>
  static constexpr cudaMemcpyKind __transfer_kind =
    __select_execution_space<_OtherProperties...> == _ExecutionSpace::__host
      ? (__is_host_only ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice)
      : (__is_host_only ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);

  //! @brief Helper to return an async_resource_ref to the currently used resource. Used to grow the async_vector
  __resource_ref_t __borrow_resource() const noexcept
  {
    return const_cast<__resource_t&>(__buf_.get_memory_resource());
  }

  //! @brief Replaces the content of the async_vector with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  template <class _Iter, cudaMemcpyKind __kind = __detect_transfer_kind<__is_host_only, _Iter>>
  _CCCL_HIDE_FROM_ABI void __assign_impl(const size_type __count, _Iter __first, _Iter __last)
  {
    if (capacity() < __count)
    {
      (void) __buf_.__replace_allocation(__count);
    }

    this->__copy_cross<_Iter, __kind>(__first, __last, __unwrapped_begin(), __count);
    __size_ = __count;
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
        "cudax::async_vector::__copy_same: failed to copy data",
        __dest,
        __first,
        sizeof(_Tp) * __count,
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
    { // For non-coniguous iterators we need to copy into temporary host storage to use cudaMemcpy
      // This should only ever happen when passing in data from host to device
      _CCCL_ASSERT(__kind == cudaMemcpyHostToDevice, "Invalid use case!");
      auto __temp = _CUDA_VSTD::get_temporary_buffer<_Tp>(__count).first;
      _CUDA_VSTD::copy(__first, __last, __temp);
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_vector::__copy_cross: failed to copy data",
        __dest,
        __temp,
        sizeof(_Tp) * __count,
        __kind,
        __buf_.get_stream().get());
      _CUDA_VSTD::return_temporary_buffer(__temp);
    }
    else
    {
      (void) __last;
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_vector::__copy_cross: failed to copy data",
        __dest,
        _CUDA_VSTD::to_address(__first),
        sizeof(_Tp) * __count,
        __kind,
        __buf_.get_stream().get());
    }
  }

  //! @brief Creates a gap in the current memory of size \p __count at position \p __pos
  //! @param __pos Pointer to the first element to be moved.
  //! @param __count The size of the empty space.
  //! Depending on the available memory this might need to reallocate. Furthermore, if this executes on device then we
  //! need temporary storage because we want to work in parallel.
  _CCCL_HIDE_FROM_ABI void __create_gap(size_type __pos, size_type __count)
  {
    pointer __first            = __unwrapped_begin();
    pointer __end              = __unwrapped_end();
    pointer __middle           = __first + __pos;
    const size_type __new_size = __size_ + __count;
    if (capacity() - __size_ < __count)
    { // need to reallocate
      __buffer_t __old_buf = __buf_.__replace_allocation(__new_size);

      // We are using __first and __end although they are pointing to __old_buf
      _CCCL_IF_CONSTEXPR (__is_host_only)
      {
        _CUDA_VSTD::copy(__first, __middle, __unwrapped_begin());
        _CUDA_VSTD::copy(__middle, __end, __unwrapped_begin() + __pos + __count);
      }
      else
      {
        // We did not increase the size if the async_vector, so we need to include `__count` elements at the end
        thrust::for_each(thrust::cuda::par_nosync.on(__buf_.get_stream().get()),
                         __unwrapped_begin(),
                         __unwrapped_end() + __count,
                         __copy_segmented_op{__first, __unwrapped_begin(), __pos, __count});
      }
    }
    else if (__pos != __size_)
    { // need to create space in the middle
      _CCCL_IF_CONSTEXPR (__is_host_only)
      {
        _CUDA_VSTD::move_backward(__middle, __end, __end + __count);
      }
      else
      { // on device we need temporary storage
        __buffer_t __temp_buf{__borrow_resource(), __buf_.get_stream(), static_cast<size_type>(__size_ - __pos)};
        this->__copy_same(__middle, __end, __temp_buf.begin());
        this->__copy_same(__temp_buf.begin(), __temp_buf.end(), __middle + __count);
      }
    }
  }

  //! @brief Rotates elements from `[__first, __middle)` after `[__middle, __last)`.
  //! @param __first Pointer to the start of the first segment.
  //! @param __middle Pointer to the end of the first segment.
  //! @param __last Pointer to the end of the second segment.
  _CCCL_HIDE_FROM_ABI void __rotate(pointer __first, pointer __middle, pointer __last)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      _CUDA_VSTD::rotate(__first, __middle, __last);
    }
    else
    { // on device we need temp storage
      const auto __size = static_cast<size_type>(__last - __first);
      __buffer_t __temp_buf{__borrow_resource(), __buf_.get_stream(), __size};
      pointer __tail = this->__copy_same(__middle, __last, __temp_buf.begin());
      this->__copy_same(__first, __middle, __tail);
      this->__copy_same(__temp_buf.begin(), __temp_buf.end(), __first);
    }
  }

  //! @brief Value-initializes elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __value_initialize_n(pointer __first, size_type __count)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      _CUDA_VSTD::fill_n(__first, __count, _Tp());
    }
    else
    {
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __count, _Tp());
    }
  }

  //! @brief Copy-constructs elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __fill_n(pointer __first, size_type __count, const _Tp& __value)
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

  //! @brief Removes elements within the sequence `[__first, __last)` that are equal to \p __value.
  //! @param __first Pointer to the start of the sequence.
  //! @param __last Pointer to the end of the sequence.
  //! @param __value The element to be removed.
  //! @note This does *not* change the size of the async_vector and should be followed by a call to erase
  _CCCL_HIDE_FROM_ABI pointer __remove_value(pointer __first, pointer __last, const _Tp& __value)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      return _CUDA_VSTD::remove(__first, __last, __value);
    }
    else
    {
      return thrust::remove(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __last, __value);
    }
    _CCCL_UNREACHABLE();
  }

  //! @brief Removes elements within the sequence `[__first, __last)` that are satisfy \p __pred.
  //! @param __first Pointer to the start of the sequence.
  //! @param __last Pointer to the end of the sequence.
  //! @param __pred The predicate to select elements for removal.
  template <class _Predicate>
  _CCCL_HIDE_FROM_ABI pointer __remove_pred(pointer __first, pointer __last, _Predicate __pred)
  {
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      return _CUDA_VSTD::remove_if(__first, __last, __pred);
    }
    else
    {
      return thrust::remove_if(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __last, __pred);
    }
    _CCCL_UNREACHABLE();
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

  //! @brief Copy-constructs a async_vector
  //! @param __other The other async_vector.
  //! The new async_vector has capacity of \p __other.size() which is potentially less than \p __other.capacity().
  //! @note No memory is allocated if \p __other is empty
  _CCCL_HIDE_FROM_ABI async_vector(const async_vector& __other)
      : __buf_(__other.get_memory_resource(), __other.get_stream(), __other.__size_)
      , __size_(__other.__size_)
      , __policy_(__other.__policy_)
  {
    if (__other.__size_ != 0)
    {
      this->__copy_same(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
    }
  }

  //! @brief Move-constructs a async_vector
  //! @param __other The other async_vector.
  //! The new async_vector takes ownership of the allocation of \p __other and resets it.
  _CCCL_HIDE_FROM_ABI async_vector(async_vector&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Copy-constructs from a async_vector with matching properties
  //! @param __other The other async_vector.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_vector(const async_vector<_Tp, _OtherProperties...>& __other)
      : __buf_(__other.get_memory_resource(), __other.get_stream(), __other.__size_)
      , __size_(__other.__size_)
      , __policy_(__other.__policy_)
  {
    if (__other.__size_ != 0)
    {
      this->__copy_cross<const_pointer, __transfer_kind<_OtherProperties...>>(
        __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.__size_);
    }
  }

  //! @brief Move-constructs from a async_vector with matching properties
  //! @param __other The other async_vector.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_vector(async_vector<_Tp, _OtherProperties...>&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Constructs an empty async_vector using an environment
  //! @param __env The environment providing the needed information
  //! @note No memory is allocated.
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env)
      : async_vector(__env, 0, ::cuda::experimental::uninit)
  {}

  //! @brief Constructs a async_vector of size \p __size using a memory resource and value-initializes \p __size
  //! elements
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __size The size of the async_vector. Defaults to zero
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_vector(const __env_t& __env, const size_type __size)
      : async_vector(__env, __size, ::cuda::experimental::uninit)
  {
    if (__size != 0)
    {
      this->__value_initialize_n(__unwrapped_begin(), __size);
    }
  }

  //! @brief Constructs a async_vector of size \p __size using a memory resource and copy-constructs \p __size elements
  //! from \p __value
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __size The size of the async_vector.
  //! @param __value The value all elements are copied from.
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_vector(const __env_t& __env, const size_type __size, const _Tp& __value)
      : async_vector(__env, __size, ::cuda::experimental::uninit)
  {
    if (__size != 0)
    {
      this->__fill_n(__unwrapped_begin(), __size, __value);
    }
  }

  //! @brief Constructs a async_vector of size \p __size using a memory and leaves all elements uninitialized
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __size The size of the async_vector.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`.
  //! At the destruction of the \c async_vector all elements in the range `[vec.begin(), vec.end())` will be destroyed.
  _CCCL_HIDE_FROM_ABI explicit async_vector(const __env_t& __env, const size_type __size, ::cuda::experimental::uninit_t)
      : __buf_(
          __env.query(::cuda::experimental::get_memory_resource), __env.query(::cuda::experimental::get_stream), __size)
      , __size_(__size)
      , __policy_(__env.query(::cuda::experimental::execution::get_execution_policy))
  {}

  //! @brief Constructs a async_vector using a memory resource and copy-constructs all elements from the input range
  //! ``[__first, __last)``
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated. Might allocate multiple times
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _CCCL_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _Iter __first, _Iter __last)
      : async_vector(__env, 0, ::cuda::experimental::uninit)
  {
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

  //! @brief Constructs a async_vector using a memory resource and copy-constructs all elements from the forward range
  //! ``[__first, __last)``
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _Iter __first, _Iter __last)
      : async_vector(__env, static_cast<size_type>(_CUDA_VSTD::distance(__first, __last)), ::cuda::experimental::uninit)
  {
    if (__size_ > 0)
    {
      this->__copy_cross<_Iter, __detect_transfer_kind<__is_host_only, _Iter>>(
        __first, __last, __unwrapped_begin(), __size_);
    }
  }

  //! @brief Constructs a async_vector using a memory resource and copy-constructs all elements from \p __ilist
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __ilist The initializer_list being copied into the async_vector.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _CUDA_VSTD::initializer_list<_Tp> __ilist)
      : async_vector(__env, __ilist.size(), ::cuda::experimental::uninit)
  {
    if (__size_ > 0)
    {
      this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __size_);
    }
  }

  //! @brief Constructs a async_vector using a memory resource and an input range
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __range The input range to be moved into the async_vector.
  //! @note If `__range.size() == 0` then no memory is allocated. May allocate multiple times in case of input ranges.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND(!_CUDA_VRANGES::forward_range<_Range>))
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _Range&& __range)
      : async_vector(__env, 0, ::cuda::experimental::uninit)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    for (; __first != __last; ++__first)
    {
      emplace_back(_CUDA_VRANGES::iter_move(__first));
    }
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  //! @brief Constructs a async_vector using a memory resource and an input range
  //! @param __mr The memory resource to allocate the async_vector with.
  //! @param __range The input range to be moved into the async_vector.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND
                   _CUDA_VRANGES::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _Range&& __range)
      : async_vector(__env, static_cast<size_type>(_CUDA_VRANGES::size(__range)), ::cuda::experimental::uninit)
  {
    if (__size_ > 0)
    {
      using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
      this->__copy_cross<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size_);
    }
  }

  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI async_vector(const __env_t& __env, _Range&& __range)
      : async_vector(
          __env,
          static_cast<size_type>(_CUDA_VRANGES::distance(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range))),
          ::cuda::experimental::uninit)
  {
    if (__size_ > 0)
    {
      using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
      this->__copy_cross<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size_);
    }
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup assignment
  //! @{

  //! @brief Copy-assigns a async_vector
  //! @param __other The other async_vector.
  //! @note Even if the old async_vector would have enough storage available, we may have to reallocate if the stored
  //! memory resource is not equal to the new one. In that case no memory is allocated if \p __other is empty.
  _CCCL_HIDE_FROM_ABI async_vector& operator=(const async_vector& __other)
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    const auto __count = __other.size();
    if (__borrow_resource() != __other.__borrow_resource())
    {
      __buffer_t __new_buf{__other.get_memory_resource(), __other.get_stream(), __count};
      _CUDA_VSTD::swap(__buf_, __new_buf);
    }
    else if (capacity() < __count)
    {
      (void) __buf_.__replace_allocation(__count);
      __buf_.change_stream(__other.get_stream());
    }

    this->__copy_same(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
    __size_   = __other.__size_;
    __policy_ = __other.__policy_;
    return *this;
  }

  //! @brief Move-assigns a async_vector
  //! @param __other The other async_vector.
  //! Clears the async_vector and swaps the contents with \p __other.
  _CCCL_HIDE_FROM_ABI async_vector& operator=(async_vector&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    __buf_    = _CUDA_VSTD::move(__other.__buf_);
    __size_   = _CUDA_VSTD::exchange(__other.__size_, 0);
    __policy_ = _CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy);
    return *this;
  }

  //! @brief Assigns an initializer_list to a async_vector, replacing its content with that of the initializer_list
  //! @param __ilist The initializer_list to be assigned
  _CCCL_HIDE_FROM_ABI async_vector& operator=(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = __ilist.size();
    if (capacity() < __count)
    {
      (void) __buf_.__replace_allocation(__count);
    }

    this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __count);
    __size_ = __count;
    return *this;
  }

  //! @brief Copy-assigns from a different async_vector
  //! @param __other The other async_vector.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI async_vector& operator=(const async_vector<_Tp, _OtherProperties...>& __other)
  {
    if (this == reinterpret_cast<const async_vector*>(_CUDA_VSTD::addressof(__other)))
    {
      return *this;
    }

    const auto __count = __other.size();
    if (__borrow_resource() != __resource_ref_t(__other.__borrow_resource()))
    {
      __buffer_t __new_buf{__other.get_memory_resource(), __other.get_stream(), __count};
      _CUDA_VSTD::swap(__buf_, __new_buf);
    }
    else if (capacity() < __count)
    {
      (void) __buf_.__replace_allocation(__count);
      __buf_.change_stream(__other.get_stream());
    }

    this->__copy_cross<const_pointer, __transfer_kind<_OtherProperties...>>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __count);
    __size_   = __count;
    __policy_ = __other.__policy_;
    return *this;
  }

  //! @brief Move-assigns from a different async_vector
  //! @param __other The other async_vector.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI async_vector& operator=(async_vector<_Tp, _OtherProperties...>&& __other)
  {
    if (this == reinterpret_cast<async_vector*>(_CUDA_VSTD::addressof(__other)))
    {
      return *this;
    }

    __buf_    = _CUDA_VSTD::move(__other.__buf_);
    __size_   = _CUDA_VSTD::exchange(__other.__size_, 0);
    __policy_ = _CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy);
    return *this;
  }

  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the async_vector. If the async_vector is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_vector. If the async_vector is empty, the
  //! returned iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_vector. If the async_vector is empty, the
  //! returned iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of the async_vector. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator end() noexcept
  {
    return iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_vector. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_vector. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed async_vector. It corresponds to the last
  //! element of the non-reversed async_vector. If the async_vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_vector. It corresponds to
  //! the last element of the non-reversed async_vector. If the async_vector is empty, the returned iterator is equal to
  //! rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_vector. It corresponds to
  //! the last element of the non-reversed async_vector. If the async_vector is empty, the returned iterator is equal to
  //! rend().
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed async_vector. It
  //! corresponds to the element preceding the first element of the non-reversed async_vector. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_vector. It corresponds to the element preceding the first element of the non-reversed async_vector. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_vector. It corresponds to the element preceding the first element of the non-reversed async_vector. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the async_vector. If the async_vector has not allocated memory
  //! the pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the async_vector. If the async_vector has not allocated memory
  //! the pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer data() const noexcept
  {
    return __buf_.data();
  }

#ifndef _CCCL_DOXYGEN_INVOKED
  //! @brief Returns a pointer to the first element of the async_vector. If the async_vector is empty, the returned
  //! pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the async_vector. If the async_vector is empty, the
  //! returned pointer will be null.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the async_vector. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns a const pointer to the element following the last element of the async_vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + __size_;
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @}

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reference operator[](const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector subscript out of range!");
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reference operator[](const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector subscript out of range!");
    return begin()[__n];
  }

  //! @brief Returns a reference to the first element of the async_vector
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reference front() noexcept
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector front() called on empty async_vector!");
    return begin()[0];
  }

  //! @brief Returns a reference to the first element of the async_vector
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reference front() const noexcept
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector front() called on empty async_vector!");
    return begin()[0];
  }

  //! @brief Returns a reference to the last element of the async_vector
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI reference back() noexcept
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector back() called on empty async_vector!");
    return begin()[__size_ - 1];
  }

  //! @brief Returns a reference to the last element of the async_vector
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const_reference back() const noexcept
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector back() called on empty async_vector!");
    return begin()[__size_ - 1];
  }
  //! @}

  //! @addtogroup capacity
  //! @{
  //! @brief Returns the current number of elements stored in the async_vector.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI size_type size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns true if the async_vector is empty.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI bool empty() const noexcept
  {
    return __size_ == 0;
  }

  //! @brief Returns the capacity of the current allocation of the async_vector..
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI size_type capacity() const noexcept
  {
    return static_cast<size_type>(__buf_.size());
  }

  //! @brief Returns the maximal size of the async_vector.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI static constexpr size_type max_size() noexcept
  {
    return static_cast<size_type>((_CUDA_VSTD::numeric_limits<difference_type>::max)());
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource <cudax-memory-resource-any-resource>`
  //! that holds the memory resource used to allocate the async_vector
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

  //! @brief Returns the execution policy
  _CCCL_HIDE_FROM_ABI void wait() const
  {
    __buf_.get_stream().wait();
  }

  //! @}

  //! @addtogroup assign
  //! @{
  //! @brief Replaces the content of the async_vector with `__count` copies of `__value`
  //! @param __count The number of elements to assign.
  //! @param __value The element to be copied.
  //! @note Neither frees not allocates memory if `__first == __last`.
  _CCCL_HIDE_FROM_ABI void assign(const size_type __count, const _Tp& __value)
  {
    if (capacity() < __count)
    {
      (void) __buf_.__replace_allocation(__count);
    }

    this->__fill_n(__unwrapped_begin(), __count, __value);
    __size_ = __count;
  }

  //! @brief Replaces the content of the async_vector with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  //! @note Neither frees not allocates memory if `__first == __last`. May allocate multiple times in case of input
  //! iterators.
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _CCCL_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  _CCCL_HIDE_FROM_ABI void assign(_Iter __first, _Iter __last)
  {
    __size_ = 0;
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

  //! @brief Replaces the content of the async_vector with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  //! @note Neither frees not allocates memory if `__first == __last`.
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  _CCCL_HIDE_FROM_ABI void assign(_Iter __first, _Iter __last)
  {
    const auto __count = static_cast<size_type>(_CUDA_VSTD::distance(__first, __last));
    this->__assign_impl(__count, __first, __last);
  }

  //! @brief Replaces the content of the async_vector with the initializer_list \p __ilist
  //! @param __ilist The initializer_list to be copied into this async_vector.
  //! @note Neither frees not allocates memory if `__ilist.size() == 0`.
  _CCCL_HIDE_FROM_ABI void assign(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = static_cast<size_type>(__ilist.size());
    this->__assign_impl(__count, __ilist.begin(), __ilist.end());
  }

  //! @brief Replaces the content of the async_vector with the range \p __range
  //! @param __range The range to be copied into this async_vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  //! @note May reallocate multiple times in case of input ranges.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND(!_CUDA_VRANGES::forward_range<_Range>))
  _CCCL_HIDE_FROM_ABI void assign_range(_Range&& __range)
  {
    __size_           = 0;
    auto __first      = _CUDA_VRANGES::begin(__range);
    const auto __last = _CUDA_VRANGES::end(__range);
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  //! @brief Replaces the content of the async_vector with the range \p __range
  //! @param __range The range to be copied into this async_vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND
                   _CUDA_VRANGES::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI void assign_range(_Range&& __range)
  {
    const auto __count = _CUDA_VRANGES::size(__range);
    using _Iter        = _CUDA_VRANGES::iterator_t<_Range>;
    this->__assign_impl<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(
      __count, _CUDA_VSTD::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }

  //! @brief Replaces the content of the async_vector with the range \p __range
  //! @param __range The range to be copied into this async_vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI void assign_range(_Range&& __range)
  {
    const auto __first = _CUDA_VRANGES::begin(__range);
    const auto __last  = _CUDA_VRANGES::__unwrap_end(__range);
    const auto __count = static_cast<size_type>(_CUDA_VRANGES::distance(__first, __last));

    using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
    this->__assign_impl<_Iter, __detect_transfer_kind<__is_host_only, _Range>>(__count, __first, __last);
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup modification
  //! @{

  //! @brief Inserts a copy of \p __value at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __value The element to be copied into the async_vector.
  //! @return Iterator to the current position of the new element.
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, const _Tp& __value)
  {
    return emplace(__cpos, __value);
  }

  //! @brief Inserts \p __value at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __value The element to be moved into the async_vector.
  //! @return Iterator to the current position of the new element.
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, _Tp&& __value)
  {
    return emplace(__cpos, _CUDA_VSTD::move(__value));
  }

  //! @brief Inserts \p __count copies of \p __value at position \p __cpos. Elements after \p __cpos are shifted to the
  //! back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __count The number of elements to be copied into the async_vector.
  //! @param __value The element to be copied into the async_vector.
  //! @return Iterator to the current position of the first new element.
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, const size_type __count, const _Tp& __value)
  {
    const auto __pos = static_cast<size_type>(__cpos - cbegin());
    _CCCL_ASSERT(__pos <= __size_, "cuda::experimental::async_vector insert called with out of bound position!");

    if (__count == 0)
    {
      return begin() + __pos;
    }

    this->__create_gap(__pos, __count);
    this->__fill_n(__unwrapped_begin() + __pos, __count, __value);
    __size_ += __count;
    return begin() + __pos;
  }

  //! @brief Inserts copies of the sequence `[__first, __last)]` at position \p __cpos. Elements after \p __cpos are
  //! shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __first Iterator to the first element to be copied into the async_vector.
  //! @param __last Iterator after to the last element to be copied into the async_vector.
  //! @return Iterator to the current position of the first new element.
  //! @note May allocate multiple time in case of input iterators
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _CCCL_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    const auto __pos = static_cast<size_type>(__cpos - cbegin());
    _CCCL_ASSERT(__pos <= __size_, "cuda::experimental::async_vector insert called with out of bound position!");

    if (__first == __last)
    {
      return begin() + __pos;
    }

    // add all new elements to the back then rotate
    const size_type __old_size = __size_;
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    if (__old_size != __pos)
    {
      this->__rotate(__unwrapped_begin() + __pos, __unwrapped_begin() + __old_size, __unwrapped_end());
    }
    return begin() + __pos;
  }

  //! @brief Inserts copies of the sequence `[__first, __last)]` at position \p __cpos. Elements after \p __cpos are
  //! shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __first Iterator to the first element to be copied into the async_vector.
  //! @param __last Iterator after to the last element to be copied into the async_vector.
  //! @return Iterator to the current position of the first new element.
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    const auto __pos = static_cast<size_type>(__cpos - cbegin());
    _CCCL_ASSERT(__pos <= __size_, "cuda::experimental::async_vector insert called with out of bound position!");

    if (__first == __last)
    {
      return begin() + __pos;
    }

    const auto __count = static_cast<size_type>(_CUDA_VSTD::distance(__first, __last));
    this->__create_gap(__pos, __count);
    this->__copy_cross(__first, __last, __unwrapped_begin() + __pos, __count);
    __size_ += __count;
    return begin() + __pos;
  }

  //! @brief Inserts an initializer_list at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __ilist The initializer_list containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  _CCCL_HIDE_FROM_ABI iterator insert(const_iterator __cpos, _CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    return insert(__cpos, __ilist.begin(), __ilist.end());
  }

  //! @brief Inserts a sequence \p __range at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __range The range containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  //! @note May allocate multiple times in case of input ranges.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND(!_CUDA_VRANGES::forward_range<_Range>))
  _CCCL_HIDE_FROM_ABI iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    const auto __pos = static_cast<size_type>(__cpos - cbegin());
    _CCCL_ASSERT(__pos <= __size_, "cuda::experimental::async_vector insert_range called with out of bound position!");

    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    if (__first == __last)
    {
      return begin() + __pos;
    }

    // add all new elements to the back then rotate
    const size_type __old_size = __size_;
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    if (__old_size != __pos)
    {
      this->__rotate(__unwrapped_begin() + __pos, __unwrapped_begin() + __old_size, __unwrapped_end());
    }
    return begin() + __pos;
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates both overloads
  //! @brief Inserts a sequence \p __range at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __range The range containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range>)
  _CCCL_HIDE_FROM_ABI iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    return insert(__cpos, _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @brief Appends a sequence \p __range at the end of the async_vector.
  //! @param __range The range containing the elements to be appended.
  //! @note May allocate multiple times in case of input ranges.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND(!_CUDA_VRANGES::forward_range<_Range>))
  _CCCL_HIDE_FROM_ABI void append_range(_Range&& __range)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates both overloads
  //! @brief Appends a sequence \p __range at the end of the async_vector.
  //! @param __range The range containing the elements to be appended.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range>)
  _CCCL_HIDE_FROM_ABI void append_range(_Range&& __range)
  {
    insert(end(), _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @brief Constructs a new element at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the new element is constructed.
  //! @param __args The arguments forwarded to the constructor.
  //! @return Iterator to the current position of the new element.
  template <class... _Args>
  _CCCL_HIDE_FROM_ABI iterator emplace(const_iterator __cpos, _Args&&... __args)
  {
    const auto __pos = static_cast<size_type>(__cpos - cbegin());
    _CCCL_ASSERT(__pos <= __size_, "cuda::experimental::async_vector emplace called with out of bound position!");

    this->__create_gap(__pos, 1);
    pointer __middle = __unwrapped_begin() + __pos;
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      _CUDA_VSTD::__construct_at(__middle, _CUDA_VSTD::forward<_Args>(__args)...);
    }
    else
    {
      _Tp __temp{_CUDA_VSTD::forward<_Args>(__args)...};
      this->__copy_cross(_CUDA_VSTD::addressof(__temp), _CUDA_VSTD::addressof(__temp) + 1, __middle, 1);
    }
    ++__size_;
    return __middle;
  }

  //! @brief Constructs a new element at the end of the async_vector.
  //! @param __args The arguments forwarded to the constructor.
  //! @return Reference to the new element.
  template <class... _Args>
  _CCCL_HIDE_FROM_ABI reference emplace_back(_Args&&... __args)
  {
    this->__create_gap(__size_, 1);
    pointer __final = __unwrapped_end();
    _CCCL_IF_CONSTEXPR (__is_host_only)
    {
      _CUDA_VSTD::__construct_at(__final, _CUDA_VSTD::forward<_Args>(__args)...);
    }
    else
    {
      _Tp __temp{_CUDA_VSTD::forward<_Args>(__args)...};
      this->__copy_cross(_CUDA_VSTD::addressof(__temp), _CUDA_VSTD::addressof(__temp) + 1, __final, 1);
    }
    ++__size_;
    return *__final;
  }

  //! @brief Copies a new element to the end of the async_vector.
  //! @param __value The element to be copied.
  //! @return Reference to the new element.
  _CCCL_HIDE_FROM_ABI reference push_back(const _Tp& __value)
  {
    return emplace_back(__value);
  }

  //! @brief Moves a new element to the end of the async_vector.
  //! @param __value The element to be copied.
  //! @return Reference to the new element.
  _CCCL_HIDE_FROM_ABI reference push_back(_Tp&& __value)
  {
    return emplace_back(_CUDA_VSTD::move(__value));
  }

  //! @brief Removes the last element of the async_vector.
  _CCCL_HIDE_FROM_ABI void pop_back() noexcept
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector::pop_back async_vector empty before pop!");
    --__size_;
  }

  //! @brief Removes the element pointed to by \p __cpos. All elements after \p __cpos are moved to the front.
  //! @param __cpos Iterator to the position of the element to be removed.
  //! @return Iterator to the new element at \p __cpos.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator
  erase(const_iterator __cpos) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    _CCCL_ASSERT(__size_ != 0, "cuda::experimental::async_vector erase called on empty async_vector!");
    return erase(__cpos, __cpos + 1);
  }

  //! @brief Removes the elements between \p __cfirst and \p __clast. All elements after \p __clast are moved to the
  //! front.
  //! @param __cfirst Iterator to the first element to be removed.
  //! @param __clast Iterator after the last element to be removed.
  //! @return Iterator to the new element at \p __cfirst.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI iterator erase(const_iterator __cfirst, const_iterator __clast) noexcept(
    _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const auto __pos = static_cast<size_type>(__cfirst - cbegin());
    if (__cfirst == __clast)
    {
      return begin() + __pos;
    }

    const auto __count = static_cast<size_type>(__clast - __cfirst);
    _CCCL_ASSERT(__size_ > __pos + __count, "cuda::experimental::async_vector::erase iterator out of bounds!");

    pointer __middle = __unwrapped_begin() + __pos;
    this->__rotate(__middle, __middle + __count, __unwrapped_end());
    __size_ -= __count;
    return __middle;
  }

  // [containers.sequences.async_vector.erasure]
  //! @brief Removes all elements that are equal to \p __value
  //! @param __value The element to be removed.
  //! @return The number of elements that have been removed.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI size_type
  __erase(const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const pointer __old_end = __unwrapped_end();
    const pointer __new_end = this->__remove_value(__unwrapped_begin(), __old_end, __value);
    __size_ -= static_cast<size_type>(__old_end - __new_end);
    return static_cast<size_type>(__old_end - __new_end);
  }

  //! @brief Removes all elements that satisfy \p __pred
  //! @param __pred The unary predicate selecting elements to be removed.
  //! @return The number of elements that have been removed.
  template <class _Pred>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI size_type
  __erase_if(_Pred __pred) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const pointer __old_end = __unwrapped_end();
    const pointer __new_end = this->__remove_pred(__unwrapped_begin(), __old_end, _CUDA_VSTD::move(__pred));
    __size_ -= static_cast<size_type>(__old_end - __new_end);
    return static_cast<size_type>(__old_end - __new_end);
  }

  //! @brief Destroys all elements in the \c async_vector and sets the size to 0
  _CCCL_HIDE_FROM_ABI void clear() noexcept
  {
    __size_ = 0;
  }

  //! @brief Provides sufficient storage for \p __count elements in the \c async_vector without creating any new
  //! elements
  //! @param __size The intended capacity of the async_vector.
  //! If `__size <= vec.capacity()` this is a noop
  _CCCL_HIDE_FROM_ABI void reserve(const size_type __size) noexcept
  {
    if (__size <= capacity())
    {
      return;
    }

    __buffer_t __old_buf = __buf_.__replace_allocation(__size);
    this->__copy_same(__old_buf.begin(), __old_buf.end(), __unwrapped_begin());
  }

  //! @brief Changes the size of the \c async_vector to \p __size and value-initializes new elements
  //! @param __size The intended size of the async_vector.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it value-initializes new elements
  _CCCL_HIDE_FROM_ABI void resize(const size_type __size) noexcept
  {
    if (__size <= __size_)
    {
      // nothing to do
    }
    else
    {
      if (__size <= capacity())
      {
        this->__value_initialize_n(__unwrapped_end(), __size - __size_);
      }
      else
      {
        __buffer_t __old_buf = __buf_.__replace_allocation(__size);
        this->__copy_same(__old_buf.begin(), __old_buf.end(), __unwrapped_begin());
        this->__value_initialize_n(__unwrapped_begin() + __size_, __size - __size_);
      }
    }
    __size_ = __size;
  }

  //! @brief Changes the size of the \c async_vector to \p __size and copy-constructs new elements from \p __value
  //! @param __size The intended size of the async_vector.
  //! @param __value The element to be copied into the async_vector when growing.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it copy-constructs new elements
  //! from \p __value
  _CCCL_HIDE_FROM_ABI void resize(const size_type __size, const _Tp& __value) noexcept
  {
    if (__size <= __size_)
    {
      // nothing to do
    }
    else
    {
      if (__size <= capacity())
      {
        this->__fill_n(__unwrapped_end(), __size - __size_, __value);
      }
      else
      {
        __buffer_t __old_buf = __buf_.__replace_allocation(__size);
        this->__copy_same(__old_buf.begin(), __old_buf.end(), __unwrapped_begin());
        this->__fill_n(__unwrapped_begin() + __size_, __size - __size_, __value);
      }
    }
    __size_ = __size;
  }

  //! @brief Changes the size of the \c async_vector to \p __size and leaves new elements uninitialized
  //! @param __size The intended size of the async_vector.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it provides sufficient storage and
  //! adjusts the size of the async_vector, but leaves the new elements uninitialized.
  _CCCL_HIDE_FROM_ABI void resize(const size_type __size, uninit_t) noexcept
  {
    if (__size <= __size_)
    {
      // nothing to do
    }
    else
    {
      if (capacity() < __size)
      {
        __buffer_t __old_buf = __buf_.__replace_allocation(__size);
        this->__copy_same(__old_buf.begin(), __old_buf.end(), __unwrapped_begin());
      }
    }
    __size_ = __size;
  }

  //! @brief Reallocates the storage, so that `vec.capacity() == vec.size()`
  //! If `vec.size() == vec.capacity()` this is a noop. If `vec.empty()` holds, then no storage is allocated
  _CCCL_HIDE_FROM_ABI void shrink_to_fit() noexcept
  {
    if (__size_ == capacity())
    {
      return;
    }

    __buffer_t __old_buf = __buf_.__replace_allocation(__size_);
    this->__copy_same(__old_buf.begin(), __old_buf.begin() + __size_, __unwrapped_begin());
  }

  //! @brief Swaps the contents of a async_vector with those of \p __other
  //! @param __other The other async_vector.
  _CCCL_HIDE_FROM_ABI void swap(async_vector& __other) noexcept
  {
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
    _CUDA_VSTD::swap(__size_, __other.__size_);
  }

  //! @brief Swaps the contents of two async_vectors
  //! @param __lhs One async_vector.
  //! @param __rhs The other async_vector.
  _CCCL_HIDE_FROM_ABI friend void swap(async_vector& __lhs, async_vector& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }
  //! @}

  //! @addtogroup comparison
  //! @{

  //! @brief Compares two async_vectors for equality
  //! @param __lhs One async_vector.
  //! @param __rhs The other async_vector.
  //! @return true, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI constexpr bool
  operator==(const async_vector& __lhs, const async_vector& __rhs) noexcept(noexcept(_CUDA_VSTD::equal(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    ::cuda::experimental::stream_ref{__lhs.get_stream()}.wait(__rhs.get_stream());
    return __lhs.__equality(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
#if _CCCL_STD_VER <= 2017
  //! @brief Compares two async_vectors for inequality
  //! @param __lhs One async_vector.
  //! @param __rhs The other async_vector.
  //! @return false, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI constexpr bool
  operator!=(const async_vector& __lhs, const async_vector& __rhs) noexcept(noexcept(_CUDA_VSTD::equal(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    ::cuda::experimental::stream_ref{__lhs.get_stream()}.wait(__rhs.get_stream());
    return !__lhs.__equality(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
#endif // _CCCL_STD_VER <= 2017

  //! @}

#ifndef _CCCL_DOXYGEN_INVOKED // friend functions are currently broken
  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND _CUDA_VSTD::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend void get_property(const async_vector&, _Property) noexcept {}
#endif // _CCCL_DOXYGEN_INVOKED
};

// [containers.sequences.inplace.async_vector.erasure]
//! @brief Removes all elements that are equal to \p __value from \p __cont.
//! @param __cont The async_vector storing the elements.
//! @param __value The element to be removed.
//! @return The number of elements that have been removed.
template <class _Tp, class... _Properties>
_CCCL_HIDE_FROM_ABI size_t erase(async_vector<_Tp, _Properties...>& __cont,
                                 const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
{
  return __cont.__erase(__value);
}

//! @brief Removes all elements that satisfy \p __pred from \p __cont.
//! @param __cont The async_vector storing the elements.
//! @param __pred The unary predicate selecting elements to be removed.
//! @return The number of elements that have been removed.
template <class _Tp, class _Pred, class... _Properties>
_CCCL_HIDE_FROM_ABI size_t erase_if(async_vector<_Tp, _Properties...>& __cont,
                                    _Pred __pred) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
{
  return __cont.__erase_if(__pred);
}

template <class _Tp>
using async_device_vector = async_vector<_Tp, _CUDA_VMR::device_accessible>;

template <class _Tp>
using async_host_vector = async_vector<_Tp, _CUDA_VMR::host_accessible>;

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif //__CUDAX__CONTAINER_ASYNC_VECTOR__
