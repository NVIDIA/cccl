//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SPAN_SHARED_MEMORY_SPAN_H
#define _CUDA___SPAN_SHARED_MEMORY_SPAN_H

#include <cuda/std/detail/__config>

#include <cstddef>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__memory/address_space.h>
#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/__iterator/reverse_iterator.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__memory/pointer_traits.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_constant_evaluated.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>
#  include <cuda/std/initializer_list>
#  include <cuda/std/span>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

using __smem_size_t = ::cuda::std::uint32_t;

template <typename _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp* __smem_to_pointer(__smem_size_t __raw_smem_ptr) noexcept
{
  auto __ptr = reinterpret_cast<_Tp*>(::__cvta_shared_to_generic(__raw_smem_ptr));
  bool __p   = ::__isShared(__ptr);
  _CCCL_ASSUME(__p);
  return __ptr;
}

[[nodiscard]] _CCCL_DEVICE_API __smem_size_t __pointer_to_smem(const void* __smem_ptr) noexcept
{
  _CCCL_ASSERT(::cuda::device::is_address_from(__smem_ptr, ::cuda::device::address_space::shared),
               "invalid smem pointer");
  return static_cast<__smem_size_t>(::__cvta_generic_to_shared(__smem_ptr));
}

[[nodiscard]] _CCCL_DEVICE_API __smem_size_t __max_smem_allocation_bytes() noexcept
{
  const auto __total_smem_size   = ::cuda::ptx::get_sreg_total_smem_size();
  const auto __dynamic_smem_size = ::cuda::ptx::get_sreg_dynamic_smem_size();
  const auto __static_smem_size  = __total_smem_size - __dynamic_smem_size;
  const auto __max_smem_size     = ::max(__static_smem_size, __dynamic_smem_size);
  return __max_smem_size;
}

/***********************************************************************************************************************
 * Static Extent
 **********************************************************************************************************************/

struct shared_memory_tag_t;

template <typename _Tag, typename _Tp, ::cuda::std::size_t _Extent>
class _CCCL_TYPE_VISIBILITY_DEFAULT span;

template <typename _Tp, ::cuda::std::size_t _Extent>
class _CCCL_TYPE_VISIBILITY_DEFAULT span<shared_memory_tag_t, _Tp, _Extent>
{
  using __dynamic_span_t = span<shared_memory_tag_t, _Tp, ::cuda::std::dynamic_extent>;

  template <::cuda::std::size_t _Count>
  using __fixed_span_t = span<shared_memory_tag_t, _Tp, _Count>;

public:
  //  constants and types
  using element_type     = _Tp;
  using value_type       = ::cuda::std::remove_cv_t<_Tp>;
  using size_type        = ::cuda::std::size_t;
  using difference_type  = ::cuda::std::ptrdiff_t;
  using pointer          = _Tp*;
  using const_pointer    = const _Tp*;
  using reference        = _Tp&;
  using const_reference  = const _Tp&;
  using iterator         = pointer;
  using reverse_iterator = ::cuda::std::reverse_iterator<iterator>;

  static constexpr auto extent = _Extent;

  static_assert(_Extent <= ::cuda::std::numeric_limits<__smem_size_t>::max(), "extent too large");

  //--------------------------------------------------------------------------------------------------------------------
  // Special members

  // [span.cons], span constructors, copy, assignment, and destructor
  _CCCL_TEMPLATE(::cuda::std::size_t _Sz = _Extent)
  _CCCL_REQUIRES((_Sz == 0))
  _CCCL_DEVICE_API constexpr span() noexcept
      : __data_{0}
  {}

  _CCCL_TEMPLATE(typename _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_const_v<_Tp2>)
  _CCCL_DEVICE_API explicit span(::cuda::std::initializer_list<value_type> __il) noexcept
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__il.begin()))}
  {
    _CCCL_ASSERT(_Extent == __il.size(), "size mismatch in span's constructor (initializer_list).");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_HIDE_FROM_ABI span(const span&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI span& operator=(const span&) noexcept = default;

  _CCCL_TEMPLATE(typename _It)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_iterator<_It, element_type>)
  _CCCL_DEVICE_API explicit span(_It __first, [[maybe_unused]] size_type __count)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__first))}
  {
    _CCCL_ASSERT(_Extent == __count, "size mismatch in span's constructor (iterator, len)");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _It, typename _End)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_iterator<_It, element_type>
                   _CCCL_AND ::cuda::std::__span_compatible_sentinel_for<_End, _It>)
  _CCCL_DEVICE_API explicit span(_It __first, [[maybe_unused]] _End __last)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__first))}
  {
    _CCCL_ASSERT(__last >= __first, "invalid range in span's constructor (iterator, sentinel)");
    _CCCL_ASSERT(__last - __first == _Extent,
                 "invalid range in span's constructor (iterator, sentinel): last - first != extent");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(size_t _Sz = _Extent)
  _CCCL_REQUIRES((_Sz != 0))
  _CCCL_DEVICE_API span(::cuda::std::type_identity_t<element_type> (&__arr)[_Sz]) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr)}
  {
    _CCCL_ASSERT(_Sz == _Extent, "size mismatch in span's constructor (array).");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(::cuda::std::array<_OtherElementType, _Extent>& __arr) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr.data())}
  {
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<const _OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const ::cuda::std::array<_OtherElementType, _Extent>& __arr) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr.data())}
  {
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_range<_Range, element_type>)
  _CCCL_DEVICE_API explicit span(_Range&& __r)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::ranges::data(__r))}
  {
    _CCCL_ASSERT(::cuda::std::ranges::size(__r) == _Extent, "size mismatch in span's constructor (range)");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType, size_type _Extent2 = _Extent)
  _CCCL_REQUIRES((_Extent2 != ::cuda::std::dynamic_extent)
                   _CCCL_AND ::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const ::cuda::std::span<_OtherElementType, _Extent2>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
  {
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherTag, typename _OtherElementType, size_type _Extent2 = _Extent)
  _CCCL_REQUIRES((_Extent2 != ::cuda::std::dynamic_extent)
                   _CCCL_AND ::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const span<_OtherTag, _OtherElementType, _Extent2>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
  {
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API explicit span(
    const ::cuda::std::span<_OtherElementType, ::cuda::std::dynamic_extent>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
  {
    _CCCL_ASSERT(_Extent == __other.size(), "size mismatch in span's constructor (other span)");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherTag, typename _OtherElementType)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API explicit span(const span<_OtherTag, _OtherElementType, ::cuda::std::dynamic_extent>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
  {
    _CCCL_ASSERT(_Extent == __other.size(), "size mismatch in span's constructor (other span)");
    _CCCL_ASSERT(__fits_in_smem(), "span does not fit in shared memory");
  }

  //  ~span() noexcept = default;

  //--------------------------------------------------------------------------------------------------------------------
  // Other member functions

  template <size_type _Count>
  [[nodiscard]] _CCCL_DEVICE_API constexpr span<shared_memory_tag_t, element_type, _Count> first() const noexcept
  {
    static_assert(_Count <= _Extent, "span<T, N>::first<Count>(): Count out of range");
    return span<shared_memory_tag_t, element_type, _Count>{__data_, _Count};
  }

  template <size_type _Count>
  [[nodiscard]] _CCCL_DEVICE_API constexpr span<shared_memory_tag_t, element_type, _Count> last() const noexcept
  {
    static_assert(_Count <= _Extent, "span<T, N>::last<Count>(): Count out of range");
    constexpr auto __count1 = __smem_size_t{_Count};
    const auto __data1      = __data_ + (__size_ - __count1) * __smem_size_t{sizeof(element_type)};
    return span<shared_memory_tag_t, element_type, _Count>{__data1, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t first(size_type __count) const noexcept
  {
    _CCCL_ASSERT(__count <= size(), "span<T, N>::first(count): count out of range");
    const auto __count1 = static_cast<__smem_size_t>(__count);
    return __dynamic_span_t{__data_, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t last(size_type __count) const noexcept
  {
    _CCCL_ASSERT(__count <= size(), "span<T, N>::last(count): count out of range");
    const auto __count1 = static_cast<__smem_size_t>(__count);
    const auto __data1  = __data_ + (__size_ - __count1) * __smem_size_t{sizeof(element_type)};
    return __dynamic_span_t{__data1, __count1};
  }

  template <size_type _Offset, size_type _Count>
  using __subspan_t =
    span<shared_memory_tag_t, element_type, (_Count != ::cuda::std::dynamic_extent) ? _Count : _Extent - _Offset>;

  template <size_type _Offset, size_type _Count = ::cuda::std::dynamic_extent>
  [[nodiscard]] _CCCL_DEVICE_API constexpr __subspan_t<_Offset, _Count> subspan() const noexcept
  {
    static_assert(_Offset <= _Extent, "span<T, N>::subspan<Offset, Count>(): Offset out of range");
    static_assert(_Count == ::cuda::std::dynamic_extent || _Count <= _Extent - _Offset,
                  "span<T, N>::subspan<Offset, Count>(): Offset + Count out of range");
    constexpr auto __offset1 = __smem_size_t{_Offset};
    constexpr auto __count1  = __smem_size_t{_Count};
    constexpr auto __size1   = (_Count == ::cuda::std::dynamic_extent) ? __size_ - __offset1 : __count1;
    const auto __data1       = __data_ + __offset1 * __smem_size_t{sizeof(element_type)};
    return __subspan_t<_Offset, _Count>{__data1, __size1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t
  subspan(size_type __offset, size_type __count = ::cuda::std::dynamic_extent) const noexcept
  {
    _CCCL_ASSERT(__offset <= size(), "span<T, N>::subspan(offset, count): offset out of range");
    _CCCL_ASSERT(__count <= size() || __count == ::cuda::std::dynamic_extent,
                 "span<T, N>::subspan(offset, count): count out of range");
    const auto __offset1 = static_cast<__smem_size_t>(__offset);
    const auto __data1   = __data_ + __offset1 * __smem_size_t{sizeof(element_type)};
    if (__count == ::cuda::std::dynamic_extent)
    {
      return {__data1, __size_ - __offset1};
    }
    _CCCL_ASSERT(__count <= size() - __offset, "span<T, N>::subspan(offset, count): offset + count out of range");
    const auto __count1 = static_cast<__smem_size_t>(__count);
    return {__data1, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr size_type size() const noexcept
  {
    return _Extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr size_type size_bytes() const noexcept
  {
    return _Extent * sizeof(element_type);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool empty() const noexcept
  {
    return (_Extent == 0);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference operator[](size_type __idx) const noexcept
  {
    _CCCL_ASSERT(__idx < size(), "span<T, N>::operator[](index): index out of range");
    const auto __idx1  = static_cast<__smem_size_t>(__idx);
    const auto __data1 = __data_ + __idx1 * __smem_size_t{sizeof(element_type)};
    return *::cuda::device::__smem_to_pointer<value_type>(__data1);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference at(size_type __idx) const
  {
    if (__idx >= size())
    {
      _CCCL_ASSERT(false, "span::at");
      ::cuda::std::terminate();
    }
    return operator[](__idx);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference front() const noexcept
  {
    _CCCL_ASSERT(!empty(), "span<T, N>::front() on empty span");
    return operator[](0);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference back() const noexcept
  {
    _CCCL_ASSERT(!empty(), "span<T, N>::back() on empty span");
    return operator[](__size_ - 1);
  }

  [[nodiscard]] _CCCL_DEVICE_API pointer data() const noexcept
  {
    return ::cuda::device::__smem_to_pointer<value_type>(__data_);
  }

  // [span.iter], span iterator support
  [[nodiscard]] _CCCL_DEVICE_API iterator begin() const noexcept
  {
    return iterator(data());
  }

  [[nodiscard]] _CCCL_DEVICE_API iterator end() const noexcept
  {
    const auto __data1 = __data_ + __size_ * __smem_size_t{sizeof(element_type)};
    return iterator(::cuda::device::__smem_to_pointer<value_type>(__data1));
  }

  [[nodiscard]] _CCCL_DEVICE_API reverse_iterator rbegin() const noexcept
  {
    return reverse_iterator(end());
  }

  [[nodiscard]] _CCCL_DEVICE_API reverse_iterator rend() const noexcept
  {
    return reverse_iterator(begin());
  }

private:
  static constexpr auto __size_ = __smem_size_t{_Extent};
  __smem_size_t __data_;

  _CCCL_DEVICE_API constexpr span(__smem_size_t __data, [[maybe_unused]] size_type __count) noexcept
      : __data_{__data}
  {
    _CCCL_ASSERT(_Extent == __count, "size mismatch in span's constructor (iterator, len)");
  }

  _CCCL_DEVICE_API static constexpr bool __fits_in_smem() noexcept
  {
    return _Extent <= ::cuda::device::__max_smem_allocation_bytes() / sizeof(element_type);
  }
};

/***********************************************************************************************************************
 * Dynamic Extent
 **********************************************************************************************************************/

template <typename _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT span<shared_memory_tag_t, _Tp, ::cuda::std::dynamic_extent>
{
  using __dynamic_span_t = span<shared_memory_tag_t, _Tp, ::cuda::std::dynamic_extent>;

  template <::cuda::std::size_t _Count>
  using __fixed_span_t = span<shared_memory_tag_t, _Tp, _Count>;

public:
  //  constants and types
  using element_type     = _Tp;
  using value_type       = ::cuda::std::remove_cv_t<_Tp>;
  using size_type        = ::cuda::std::size_t;
  using difference_type  = ::cuda::std::ptrdiff_t;
  using pointer          = _Tp*;
  using const_pointer    = const _Tp*;
  using reference        = _Tp&;
  using const_reference  = const _Tp&;
  using iterator         = pointer;
  using reverse_iterator = ::cuda::std::reverse_iterator<iterator>;

  static constexpr auto extent = ::cuda::std::dynamic_extent;

  //--------------------------------------------------------------------------------------------------------------------
  // Special members

  // [span.cons], span constructors, copy, assignment, and destructor
  _CCCL_DEVICE_API constexpr span() noexcept
      : __data_{0}
      , __size_{0}
  {}

  _CCCL_TEMPLATE(typename _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_const_v<_Tp2>)
  _CCCL_DEVICE_API span(::cuda::std::initializer_list<value_type> __il) noexcept
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__il.begin()))}
      , __size_{static_cast<__smem_size_t>(__il.size())}
  {
    _CCCL_ASSERT(__fits_in_smem(__il.size()), "span does not fit in shared memory");
  }

  _CCCL_HIDE_FROM_ABI span(const span&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI span& operator=(const span&) noexcept = default;

  _CCCL_TEMPLATE(typename _It)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_iterator<_It, element_type>)
  _CCCL_DEVICE_API span(_It __first, size_type __count)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__first))}
      , __size_{static_cast<__smem_size_t>(__count)}
  {
    _CCCL_ASSERT(__fits_in_smem(__count), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _It, typename _End)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_iterator<_It, element_type>
                   _CCCL_AND ::cuda::std::__span_compatible_sentinel_for<_End, _It>)
  _CCCL_DEVICE_API span(_It __first, _End __last)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::to_address(__first))}
      , __size_{static_cast<__smem_size_t>(__last - __first)}
  {
    _CCCL_ASSERT(__last >= __first, "invalid range in span's constructor (iterator, sentinel)");
    _CCCL_ASSERT(__fits_in_smem(__last - __first), "span does not fit in shared memory");
  }

  template <size_type _Sz>
  _CCCL_DEVICE_API span(::cuda::std::type_identity_t<element_type> (&__arr)[_Sz]) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr)}
      , __size_{_Sz}
  {
    _CCCL_ASSERT(__fits_in_smem(_Sz), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType, size_type _Sz)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(::cuda::std::array<_OtherElementType, _Sz>& __arr) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr.data())}
      , __size_{_Sz}
  {
    _CCCL_ASSERT(__fits_in_smem(_Sz), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType, size_type _Sz)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<const _OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const ::cuda::std::array<_OtherElementType, _Sz>& __arr) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__arr.data())}
      , __size_{_Sz}
  {
    _CCCL_ASSERT(__fits_in_smem(_Sz), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _Range)
  _CCCL_REQUIRES(::cuda::std::__span_compatible_range<_Range, element_type>)
  _CCCL_DEVICE_API span(_Range&& __r)
      : __data_{::cuda::device::__pointer_to_smem(::cuda::std::ranges::data(__r))}
      , __size_{static_cast<__smem_size_t>(::cuda::std::ranges::size(__r))}
  {
    _CCCL_ASSERT(__fits_in_smem(::cuda::std::ranges::size(__r)), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherElementType, size_type _OtherExtent)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const ::cuda::std::span<_OtherElementType, _OtherExtent>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
      , __size_{__other.size()}
  {
    _CCCL_ASSERT(__fits_in_smem(__other.size()), "span does not fit in shared memory");
  }

  _CCCL_TEMPLATE(typename _OtherTag, typename _OtherElementType, size_type _OtherExtent)
  _CCCL_REQUIRES(::cuda::std::__span_array_convertible<_OtherElementType, element_type>)
  _CCCL_DEVICE_API span(const span<_OtherTag, _OtherElementType, _OtherExtent>& __other) noexcept
      : __data_{::cuda::device::__pointer_to_smem(__other.data())}
      , __size_{__other.size()}
  {
    _CCCL_ASSERT(__fits_in_smem(__other.size()), "span does not fit in shared memory");
  }

  //    ~span() noexcept = default;

  //--------------------------------------------------------------------------------------------------------------------
  // Other member functions

  template <size_type _Count>
  [[nodiscard]] _CCCL_DEVICE_API constexpr span<shared_memory_tag_t, element_type, _Count> first() const noexcept
  {
    // ternary avoids "pointless comparison of unsigned integer with zero" warning
    _CCCL_ASSERT(_Count == 0 ? true : _Count <= size(), "span<T>::first<Count>(): Count out of range");
    constexpr auto __count1 = __smem_size_t{_Count};
    return span<shared_memory_tag_t, element_type, _Count>{__data_, __count1};
  }

  template <size_type _Count>
  [[nodiscard]] _CCCL_DEVICE_API constexpr span<shared_memory_tag_t, element_type, _Count> last() const noexcept
  {
    // ternary avoids "pointless comparison of unsigned integer with zero" warning
    _CCCL_ASSERT(_Count == 0 ? true : _Count <= size(), "span<T>::last<Count>(): Count out of range");
    constexpr auto __count1 = __smem_size_t{_Count};
    const auto __data1      = __data_ + (__size_ - __count1) * __smem_size_t{sizeof(element_type)};
    return span<shared_memory_tag_t, element_type, _Count>{__data1, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t first(size_type __count) const noexcept
  {
    _CCCL_ASSERT(__count <= size(), "span<T>::first(count): count out of range");
    const auto __count1 = static_cast<__smem_size_t>(__count);
    return __dynamic_span_t{__data_, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t last(size_type __count) const noexcept
  {
    _CCCL_ASSERT(__count <= size(), "span<T>::last(count): count out of range");
    const auto __count1 = static_cast<__smem_size_t>(__count);
    const auto __data1  = __data_ + (__size_ - __count1) * __smem_size_t{sizeof(element_type)};
    return __dynamic_span_t{__data1, __count1};
  }

  template <size_type _Offset, size_type _Count>
  using __subspan_t = span<shared_memory_tag_t, element_type, _Count>;

  template <size_type _Offset, size_type _Count = ::cuda::std::dynamic_extent>
  [[nodiscard]] _CCCL_DEVICE_API constexpr __subspan_t<_Offset, _Count> subspan() const noexcept
  {
    // ternary avoids "pointless comparison of unsigned integer with zero" warning
    _CCCL_ASSERT(_Offset == 0 ? true : _Offset <= size(), "span<T>::subspan<Offset, Count>(): Offset out of range");
    _CCCL_ASSERT(_Count == ::cuda::std::dynamic_extent || _Count == 0 ? true : _Count <= size() - _Offset,
                 "span<T>::subspan<Offset, Count>(): Offset + Count out of range");
    constexpr auto __offset1 = __smem_size_t{_Offset};
    constexpr auto __count1  = __smem_size_t{_Count};
    const auto __data1       = __data_ + __offset1 * __smem_size_t{sizeof(element_type)};
    return span<shared_memory_tag_t, element_type, _Count>{
      __data1, (_Count == ::cuda::std::dynamic_extent) ? __size_ - __offset1 : __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr __dynamic_span_t
  subspan(size_type __offset, size_type __count = ::cuda::std::dynamic_extent) const noexcept
  {
    _CCCL_ASSERT(__offset <= size(), "span<T>::subspan(offset, count): offset out of range");
    _CCCL_ASSERT(__count <= size() || __count == ::cuda::std::dynamic_extent,
                 "span<T>::subspan(offset, count): count out of range");
    const auto __offset1 = static_cast<__smem_size_t>(__offset);
    const auto __count1  = static_cast<__smem_size_t>(__count);
    const auto __data1   = __data_ + __offset1 * __smem_size_t{sizeof(element_type)};
    using __span_t       = __dynamic_span_t;
    if (__count == ::cuda::std::dynamic_extent) // potential performance penalty
    {
      return __span_t{__data1, __size_ - __offset1};
    }
    _CCCL_ASSERT(__count <= size() - __offset, "span<T>::subspan(offset, count): offset + count out of range");
    return __span_t{__data1, __count1};
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr size_type size() const noexcept
  {
    return __size_;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr size_type size_bytes() const noexcept
  {
    return __size_ * sizeof(element_type);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool empty() const noexcept
  {
    return __size_ == 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API reference operator[](size_type __idx) const noexcept
  {
    _CCCL_ASSERT(__idx < size(), "span<T>::operator[](index): index out of range");
    const auto __idx1  = static_cast<__smem_size_t>(__idx);
    const auto __data1 = __data_ + __idx1 * __smem_size_t{sizeof(element_type)};
    return *::cuda::device::__smem_to_pointer<value_type>(__data1);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference at(size_type __idx) const
  {
    if (__idx >= size())
    {
      _CCCL_ASSERT(false, "span::at");
      ::cuda::std::terminate();
    }
    return operator[](__idx);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference front() const noexcept
  {
    _CCCL_ASSERT(!empty(), "span<T>::front() on empty span");
    return operator[](0);
  }

  [[nodiscard]] _CCCL_DEVICE_API reference back() const noexcept
  {
    _CCCL_ASSERT(!empty(), "span<T>::back() on empty span");
    return operator[](__size_ - 1);
  }

  [[nodiscard]] _CCCL_DEVICE_API pointer data() const noexcept
  {
    return ::cuda::device::__smem_to_pointer<value_type>(__data_);
  }

  // [span.iter], span iterator support
  [[nodiscard]] _CCCL_DEVICE_API iterator begin() const noexcept
  {
    return iterator(data());
  }

  [[nodiscard]] _CCCL_DEVICE_API iterator end() const noexcept
  {
    const auto __data1 = __data_ + __size_ * __smem_size_t{sizeof(element_type)};
    return iterator(::cuda::device::__smem_to_pointer<value_type>(__data1));
  }

  [[nodiscard]] _CCCL_DEVICE_API reverse_iterator rbegin() const noexcept
  {
    return reverse_iterator(end());
  }

  [[nodiscard]] _CCCL_DEVICE_API reverse_iterator rend() const noexcept
  {
    return reverse_iterator(begin());
  }

private:
  __smem_size_t __data_;
  __smem_size_t __size_;

  _CCCL_DEVICE_API constexpr span(__smem_size_t __data, __smem_size_t __count) noexcept
      : __data_{__data}
      , __size_{__count}
  {}

  _CCCL_DEVICE_API static constexpr bool __fits_in_smem(size_type __size) noexcept
  {
    return __size <= ::cuda::device::__max_smem_allocation_bytes() / sizeof(element_type);
  }
};

/***********************************************************************************************************************
 * Non-Member Functions and Deduction Guides
 **********************************************************************************************************************/

//  as_bytes & as_writable_bytes
// template <class _Tp, size_t _Extent>
//_CCCL_DEVICE_API inline auto as_bytes(span<_Tp, _Extent> __s) noexcept
//{
//  return __s.__as_bytes();
//}
//
//_CCCL_TEMPLATE(typename _Tp, size_t _Extent)
//_CCCL_REQUIRES((!is_const<_Tp>::value))
//_CCCL_DEVICE_API inline auto as_writable_bytes(span<_Tp, _Extent> __s) noexcept
//{
//  return __s.__as_writable_bytes();
//}

//  Deduction guides
// template <class _Tp, ::cuda::std::size_t _Sz>
//_CCCL_HOST_DEVICE span(_Tp (&)[_Sz]) -> span<_Tp, _Sz>;

// template <class _Tp, ::cuda::std::size_t _Sz>
//_CCCL_HOST_DEVICE span(::cuda::std::array<_Tp, _Sz>&) -> span<_Tp, _Sz>;

// template <class _Tp, ::cuda::std::size_t _Sz>
//_CCCL_HOST_DEVICE span(const ::cuda::std::array<_Tp, _Sz>&) -> span<const _Tp, _Sz>;

//_CCCL_TEMPLATE(typename _It, typename _EndOrSize)
//_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_It>)
//_CCCL_HOST_DEVICE span(_It, _EndOrSize)
//  -> span<::cuda::std::remove_reference_t<::cuda::std::iter_reference_t<_It>>,
//  __maybe_static_ext<_EndOrSize>>;
//
//_CCCL_TEMPLATE(typename _Range)
//_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range>)
//_CCCL_HOST_DEVICE span(_Range&&)
//  -> span<::cuda::std::remove_reference_t<::cuda::std::ranges::range_reference_t<_Range>>>;
template <typename _Tp, ::cuda::std::size_t _Extent = ::cuda::std::dynamic_extent>
using shared_memory_span = span<shared_memory_tag_t, _Tp, _Extent>;

_CCCL_END_NAMESPACE_CUDA_DEVICE

//_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
// template <class _Tp, size_t _Extent>
// inline constexpr bool enable_borrowed_range<span<_Tp, _Extent>> = true;
//
// template <class _Tp, size_t _Extent>
// inline constexpr bool enable_view<span<_Tp, _Extent>> = true;
//_LIBCUDACXX_END_NAMESPACE_RANGES

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___SPAN_SHARED_MEMORY_SPAN_H
