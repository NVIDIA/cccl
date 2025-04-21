//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_ASYNC_BUFFER__
#define __CUDAX__CONTAINER_ASYNC_BUFFER__

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
#include <cuda/std/__algorithm/lexicographical_compare.h>
#include <cuda/std/__algorithm/move_backward.h>
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
#include <cuda/experimental/__launch/host_launch.cuh>
#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/get_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__stream/get_stream.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>
#include <cuda/experimental/__utility/select_execution_space.cuh>

_CCCL_PUSH_MACROS

//! @file The \c async_buffer class provides a container of contiguous memory
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-async-vector:
//!
//! async_buffer
//! -------------
//!
//! ``async_buffer`` is a container that provides resizable typed storage allocated from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment, release and growth of the allocation.
//! The elements are initialized during construction, which may require a kernel launch.
//!
//! In addition to being type-safe, ``async_buffer`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const async_buffer&, Property) <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class async_buffer
{
public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, _Properties...>;
  using const_iterator         = heterogeneous_iterator<const _Tp, _Properties...>;
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
  friend class async_buffer;

  // For now we require trivially copyable type to simplify the implementation
  static_assert(_CCCL_TRAIT(_CUDA_VSTD::is_trivially_copyable, _Tp),
                "cuda::experimental::async_buffer requires T to be trivially copyable.");

  // At least one of the properties must signal an execution space
  static_assert(_CUDA_VMR::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::async_buffer must contain at least one execution space "
                "property!");

  //! @brief Convenience shortcut to detect the execution space of the async_buffer
  static constexpr bool __is_host_only = __select_execution_space<_Properties...> == _ExecutionSpace::__host;

private:
  __buffer_t __buf_;
  size_type __size_    = 0; // initialized to 0 in case initialization of the elements might throw
  __policy_t __policy_ = __policy_t::invalid_execution_policy;

  //! @brief Helper to check container is compatible with this async_buffer
  template <class _Range>
  static constexpr bool __compatible_range = _CUDA_VRANGES::__container_compatible_range<_Range, _Tp>;

  //! @brief Helper to check whether a different async_buffer still satisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Helper to determine what cudaMemcpyKind we need to copy data from another async_buffer with different
  //!        properties. Needed for compilers that have issues handling packs in constraints
  template <class... _OtherProperties>
  static constexpr cudaMemcpyKind __transfer_kind =
    __select_execution_space<_OtherProperties...> == _ExecutionSpace::__host
      ? (__is_host_only ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice)
      : (__is_host_only ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);

  //! @brief Helper to return an async_resource_ref to the currently used resource. Used to grow the async_buffer
  __resource_ref_t __borrow_resource() const noexcept
  {
    return const_cast<__resource_t&>(__buf_.get_memory_resource());
  }

  //! @brief Replaces the content of the async_buffer with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  template <class _Iter>
  _CCCL_HIDE_FROM_ABI void __assign_impl(const size_type __count, _Iter __first, _Iter __last)
  {
    if (__size_ < __count)
    {
      (void) __buf_.__replace_allocation(__count);
    }

    this->__copy_cross<_Iter>(__first, __last, __unwrapped_begin(), __count);
    __size_ = __count;
  }

  //! @brief Copies \p __count elements from `[__first, __last)` to \p __dest, where \p __first and \p __dest reside in
  //! the different memory spaces
  //! @param __first Pointer to the start of the input segment.
  //! @param __last Pointer to the end of the input segment.
  //! @param __dest Pointer to the start of the output segment.
  //! @param __count The number of elements to be copied.
  //! @note This function is inherently asynchronous. We need to ensure that the memory pointed to by \p __first and
  //! \p __last lives long enough
  template <class _Iter>
  _CCCL_HIDE_FROM_ABI void __copy_cross(_Iter __first, [[maybe_unused]] _Iter __last, pointer __dest, size_type __count)
  {
    if (__count == 0)
    {
      return;
    }

    if constexpr (!_CUDA_VSTD::contiguous_iterator<_Iter>)
    { // For non-coniguous iterators we need to copy into temporary host storage to use cudaMemcpy
      // Currently only supported from host because no one should use non-contiguous data on device
      auto __temp = _CUDA_VSTD::get_temporary_buffer<_Tp>(__count).first;
      ::cuda::experimental::host_launch(__buf_.get_stream(), _CUDA_VSTD::copy<_Iter, pointer>, __first, __last, __temp);
      // FIXME: Something is fishy here. We need to wait otherwise the data is not properly set.
      // The test passes fine with compute-sanitizer but we really do not want to take the performance hit for this.
      // See https://github.com/NVIDIA/cccl/issues/3814
      __buf_.get_stream().sync();
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_buffer::__copy_cross: failed to copy data",
        __dest,
        __temp,
        sizeof(_Tp) * __count,
        ::cudaMemcpyDefault,
        __buf_.get_stream().get());
      // We need to free the temporary buffer in stream order to ensure the memory survives
      ::cuda::experimental::host_launch(__buf_.get_stream(), _CUDA_VSTD::return_temporary_buffer<_Tp>, __temp);
    }
    else
    {
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "cudax::async_buffer::__copy_cross: failed to copy data",
        __dest,
        _CUDA_VSTD::to_address(__first),
        sizeof(_Tp) * __count,
        ::cudaMemcpyDefault,
        __buf_.get_stream().get());
    }
  }

  //! @brief Value-initializes elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __value_initialize_n(pointer __first, size_type __count)
  {
    if (__count == 0)
    {
      return;
    }

    if constexpr (__is_host_only)
    {
      ::cuda::experimental::host_launch(
        __buf_.get_stream(), _CUDA_VSTD::uninitialized_value_construct_n<pointer, size_type>, __first, __count);
    }
    else
    {
      ::cuda::experimental::__ensure_current_device __guard(__buf_.get_stream());
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __count, _Tp());
    }
  }

  //! @brief Copy-constructs elements in the range `[__first, __first + __count)`.
  //! @param __first Pointer to the first element to be initialized.
  //! @param __count The number of elements to be initialized.
  _CCCL_HIDE_FROM_ABI void __fill_n(pointer __first, size_type __count, const _Tp& __value)
  {
    if (__count == 0)
    {
      return;
    }

    if constexpr (__is_host_only)
    {
      ::cuda::experimental::host_launch(
        __buf_.get_stream(), _CUDA_VSTD::uninitialized_fill_n<pointer, size_type, _Tp>, __first, __count, __value);
    }
    else
    {
      ::cuda::experimental::__ensure_current_device __guard(__buf_.get_stream());
      thrust::fill_n(thrust::cuda::par_nosync.on(__buf_.get_stream().get()), __first, __count, __value);
    }
  }

public:
  //! @addtogroup construction
  //! @{

  //! @brief Copy-constructs from a async_buffer
  //! @param __other The other async_buffer.
  _CCCL_HIDE_FROM_ABI async_buffer(const async_buffer& __other)
      : __buf_(__other.get_memory_resource(), __other.get_stream(), __other.__size_)
      , __size_(__other.__size_)
      , __policy_(__other.__policy_)
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.__size_);
  }

  //! @brief Move-constructs from a async_buffer
  //! @param __other The other async_buffer.
  _CCCL_HIDE_FROM_ABI async_buffer(async_buffer&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Copy-constructs from a async_buffer with matching properties
  //! @param __other The other async_buffer.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const async_buffer<_Tp, _OtherProperties...>& __other)
      : __buf_(__other.get_memory_resource(), __other.get_stream(), __other.__size_)
      , __size_(__other.__size_)
      , __policy_(__other.__policy_)
  {
    this->__copy_cross<const_pointer>(
      __other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin(), __other.__size_);
  }

  //! @brief Move-constructs from a async_buffer with matching properties
  //! @param __other The other async_buffer.
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI explicit async_buffer(async_buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
      , __policy_(_CUDA_VSTD::exchange(__other.__policy_, __policy_t::invalid_execution_policy))
  {}

  //! @brief Constructs an empty async_buffer using an environment
  //! @param __env The environment providing the needed information
  //! @note No memory is allocated.
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env)
      : async_buffer(__env, 0, ::cuda::experimental::uninit)
  {}

  //! @brief Constructs a async_buffer of size \p __size using a memory resource and value-initializes \p __size
  //! elements
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer. Defaults to zero
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const __env_t& __env, const size_type __size)
      : async_buffer(__env, __size, ::cuda::experimental::uninit)
  {
    this->__value_initialize_n(__unwrapped_begin(), __size);
  }

  //! @brief Constructs a async_buffer of size \p __size using a memory resource and copy-constructs \p __size elements
  //! from \p __value
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer.
  //! @param __value The value all elements are copied from.
  //! @note If `__size == 0` then no memory is allocated.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const __env_t& __env, const size_type __size, const _Tp& __value)
      : async_buffer(__env, __size, ::cuda::experimental::uninit)
  {
    this->__fill_n(__unwrapped_begin(), __size, __value);
  }

  //! @brief Constructs a async_buffer of size \p __size using a memory and leaves all elements uninitialized
  //! @param __env The environment used to query the memory resource.
  //! @param __size The size of the async_buffer.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`.
  //! At the destruction of the \c async_buffer all elements in the range `[vec.begin(), vec.end())` will be destroyed.
  _CCCL_HIDE_FROM_ABI explicit async_buffer(const __env_t& __env, const size_type __size, ::cuda::experimental::uninit_t)
      : __buf_(::cuda::experimental::get_memory_resource(__env), ::cuda::experimental::get_stream(__env), __size)
      , __size_(__size)
      , __policy_(__env.query(::cuda::experimental::execution::get_execution_policy))
  {}

  //! @brief Constructs a async_buffer using a memory resource and copy-constructs all elements from the forward range
  //! ``[__first, __last)``
  //! @param __env The environment used to query the memory resource.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Iter __first, _Iter __last)
      : async_buffer(__env, static_cast<size_type>(_CUDA_VSTD::distance(__first, __last)), ::cuda::experimental::uninit)
  {
    this->__copy_cross<_Iter>(__first, __last, __unwrapped_begin(), __size_);
  }

  //! @brief Constructs a async_buffer using a memory resource and copy-constructs all elements from \p __ilist
  //! @param __env The environment used to query the memory resource.
  //! @param __ilist The initializer_list being copied into the async_buffer.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _CUDA_VSTD::initializer_list<_Tp> __ilist)
      : async_buffer(__env, __ilist.size(), ::cuda::experimental::uninit)
  {
    this->__copy_cross(__ilist.begin(), __ilist.end(), __unwrapped_begin(), __size_);
  }

  //! @brief Constructs a async_buffer using a memory resource and an input range
  //! @param __env The environment used to query the memory resource.
  //! @param __range The input range to be moved into the async_buffer.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND
                   _CUDA_VRANGES::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Range&& __range)
      : async_buffer(__env, static_cast<size_type>(_CUDA_VRANGES::size(__range)), ::cuda::experimental::uninit)
  {
    using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size_);
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI async_buffer(const __env_t& __env, _Range&& __range)
      : async_buffer(
          __env,
          static_cast<size_type>(_CUDA_VRANGES::distance(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range))),
          ::cuda::experimental::uninit)
  {
    using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
    this->__copy_cross<_Iter>(
      _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin(), __size_);
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the async_buffer. If the async_buffer is empty, the returned
  //! iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned iterator will be equal to end().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of the async_buffer. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI iterator end() noexcept
  {
    return iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_buffer. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the async_buffer. This element
  //! acts as a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed async_buffer. It corresponds to the last
  //! element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_buffer. It corresponds to
  //! the last element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to
  //! rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed async_buffer. It corresponds to
  //! the last element of the non-reversed async_buffer. If the async_buffer is empty, the returned iterator is equal to
  //! rend().
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed async_buffer. It
  //! corresponds to the element preceding the first element of the non-reversed async_buffer. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_buffer. It corresponds to the element preceding the first element of the non-reversed async_buffer. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed
  //! async_buffer. It corresponds to the element preceding the first element of the non-reversed async_buffer. This
  //! element acts as a placeholder, attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer has not allocated memory
  //! the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer has not allocated memory
  //! the pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer data() const noexcept
  {
    return __buf_.data();
  }

#ifndef _CCCL_DOXYGEN_INVOKED
  //! @brief Returns a pointer to the first element of the async_buffer. If the async_buffer is empty, the returned
  //! pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the async_buffer. If the async_buffer is empty, the
  //! returned pointer will be null.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the async_buffer. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns a const pointer to the element following the last element of the async_buffer. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + __size_;
  }
#endif // _CCCL_DOXYGEN_INVOKED

  //! @}

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Always synchronizes with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reference get(const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector::get out of range!");
    this->sync();
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Always synchronizes with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reference get(const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector::get out of range!");
    this->sync();
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI reference get_unsynchronized(const size_type __n) noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector::get_unsynchronized out of range!");
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the async_vector
  //! @param __n The index of the element we want to access
  //! @note Does not synchronize with the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const_reference get_unsynchronized(const size_type __n) const noexcept
  {
    _CCCL_ASSERT(__n < __size_, "cuda::experimental::async_vector::get_unsynchronized out of range!");
    return begin()[__n];
  }

  //! @}

  //! @addtogroup size
  //! @{
  //! @brief Returns the current number of elements stored in the async_buffer.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI size_type size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns true if the async_buffer is empty.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI bool empty() const noexcept
  {
    return __size_ == 0;
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource <cudax-memory-resource-any-resource>`
  //! that holds the memory resource used to allocate the async_buffer
  //! @endrst
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __resource_t& get_memory_resource() const noexcept
  {
    return __buf_.get_memory_resource();
  }

  //! @brief Returns the stored stream
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref get_stream() const noexcept
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
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr __policy_t get_execution_policy() const noexcept
  {
    return __policy_;
  }

  //! @brief Replaces the currently used execution policy
  //! @param __new_policy the new policy
  _CCCL_HIDE_FROM_ABI constexpr void set_execution_policy(__policy_t __new_policy) noexcept
  {
    __policy_ = __new_policy;
  }

  //! @brief Synchronizes the currently stored stream
  _CCCL_HIDE_FROM_ABI void sync() const
  {
    __buf_.get_stream().sync();
  }

  //! @}

  //! @addtogroup assign
  //! @{
  //! @brief Replaces the content of the async_buffer with `__count` copies of `__value`
  //! @param __count The number of elements to assign.
  //! @param __value The element to be copied.
  //! @note Neither frees not allocates memory if `__first == __last`.
  _CCCL_HIDE_FROM_ABI void assign(const size_type __count, const _Tp& __value)
  {
    if (__size_ < __count)
    {
      (void) __buf_.__replace_allocation(__count);
    }

    this->__fill_n(__unwrapped_begin(), __count, __value);
    __size_ = __count;
  }

  //! @brief Replaces the content of the async_buffer with the sequence `[__first, __last)`
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

  //! @brief Replaces the content of the async_buffer with the initializer_list \p __ilist
  //! @param __ilist The initializer_list to be copied into this async_buffer.
  //! @note Neither frees not allocates memory if `__ilist.size() == 0`.
  _CCCL_HIDE_FROM_ABI void assign(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = static_cast<size_type>(__ilist.size());
    this->__assign_impl(__count, __ilist.begin(), __ilist.end());
  }

  //! @brief Replaces the content of the async_buffer with the range \p __range
  //! @param __range The range to be copied into this async_buffer.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND
                   _CUDA_VRANGES::sized_range<_Range>)
  _CCCL_HIDE_FROM_ABI void assign_range(_Range&& __range)
  {
    const auto __count = _CUDA_VRANGES::size(__range);
    using _Iter        = _CUDA_VRANGES::iterator_t<_Range>;
    this->__assign_impl<_Iter>(__count, _CUDA_VSTD::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen conflates the overloads
  //! @brief Replaces the content of the async_buffer with the range \p __range
  //! @param __range The range to be copied into this async_buffer.
  //! @note Neither frees nor allocates memory if `__range.size() == 0`.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(__compatible_range<_Range> _CCCL_AND _CUDA_VRANGES::forward_range<_Range> _CCCL_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CCCL_HIDE_FROM_ABI void assign_range(_Range&& __range)
  {
    const auto __first = _CUDA_VRANGES::begin(__range);
    const auto __last  = _CUDA_VRANGES::__unwrap_end(__range);
    const auto __count = static_cast<size_type>(_CUDA_VRANGES::distance(__first, __last));

    using _Iter = _CUDA_VRANGES::iterator_t<_Range>;
    this->__assign_impl<_Iter>(__count, __first, __last);
  }
#endif // _CCCL_DOXYGEN_INVOKED
  //! @}

  //! @brief Swaps the contents of a async_buffer with those of \p __other
  //! @param __other The other async_buffer.
  _CCCL_HIDE_FROM_ABI void swap(async_buffer& __other) noexcept
  {
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
    _CUDA_VSTD::swap(__size_, __other.__size_);
  }

  //! @brief Swaps the contents of two async_buffers
  //! @param __lhs One async_buffer.
  //! @param __rhs The other async_buffer.
  _CCCL_HIDE_FROM_ABI friend void swap(async_buffer& __lhs, async_buffer& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }
  //! @}

  //! @addtogroup comparison
  //! @{

  //! @brief Compares two async_buffers for equality
  //! @param __lhs One async_buffer.
  //! @param __rhs The other async_buffer.
  //! @return true, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI bool operator==(const async_buffer& __lhs, const async_buffer& __rhs)
  {
    if constexpr (__is_host_only)
    {
      // need to wait here because `host_launch` does not return values, so we cannot easily put it in stream order
      __lhs.sync();
      __rhs.sync();
      return _CUDA_VSTD::equal(
        __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
    }
    else
    {
      ::cuda::experimental::__ensure_current_device __guard(__lhs.get_stream().get());
      return (__lhs.size() == __rhs.size())
          && thrust::equal(thrust::cuda::par_nosync.on(__lhs.get_stream().get()),
                           __lhs.__unwrapped_begin(),
                           __lhs.__unwrapped_end(),
                           __rhs.__unwrapped_begin());
    }
    _CCCL_UNREACHABLE();
  }
#if _CCCL_STD_VER <= 2017
  //! @brief Compares two async_buffers for inequality
  //! @param __lhs One async_buffer.
  //! @param __rhs The other async_buffer.
  //! @return false, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI bool operator!=(const async_buffer& __lhs, const async_buffer& __rhs)
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017

  //! @}

#ifndef _CCCL_DOXYGEN_INVOKED // friend functions are currently broken
  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND _CUDA_VSTD::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend void get_property(const async_buffer&, _Property) noexcept {}
#endif // _CCCL_DOXYGEN_INVOKED
};

template <class _Tp>
using async_device_buffer = async_buffer<_Tp, _CUDA_VMR::device_accessible>;

template <class _Tp>
using async_host_buffer = async_buffer<_Tp, _CUDA_VMR::host_accessible>;

template <class _Tp, class... _TargetProperties, class... _SourceProperties>
async_buffer<_Tp, _TargetProperties...> make_async_buffer(
  const async_buffer<_Tp, _SourceProperties...>& __source,
  any_async_resource<_TargetProperties...> __mr,
  cuda::stream_ref __stream)
{
  env_t<_TargetProperties...> __env{__mr, __stream};
  async_buffer<_Tp, _TargetProperties...> __res{__env, __source.size(), uninit};
  __source.sync();

  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "cudax::async_buffer::__copy_cross: failed to copy data",
    __res.__unwrapped_begin(),
    __source.__unwrapped_begin(),
    sizeof(_Tp) * __source.size(),
    cudaMemcpyKind::cudaMemcpyDefault,
    __stream.get());

  return __res;
}

template <class _Tp, class... _TargetProperties, class... _SourceProperties>
async_buffer<_Tp, _TargetProperties...> make_async_buffer(
  const async_buffer<_Tp, _SourceProperties...>& __source, any_async_resource<_TargetProperties...> __mr)
{
  return ::cuda::experimental::make_async_buffer(__source, __mr, __source.get_stream());
}

template <class _Tp, class... _SourceProperties>
async_buffer<_Tp, _SourceProperties...> make_async_buffer(const async_buffer<_Tp, _SourceProperties...>& __source)
{
  return ::cuda::experimental::make_async_buffer(__source, __source.get_memory_resource(), __source.get_stream());
}

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif //__CUDAX__CONTAINER_ASYNC_BUFFER__
