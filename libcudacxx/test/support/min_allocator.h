//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIN_ALLOCATOR_H
#define MIN_ALLOCATOR_H

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdlib>

#include "test_macros.h"

template <class T>
class bare_allocator
{
public:
  typedef T value_type;

  __host__ __device__ bare_allocator() TEST_NOEXCEPT {}

  template <class U>
  __host__ __device__ bare_allocator(bare_allocator<U>) TEST_NOEXCEPT
  {}

  __host__ __device__ T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  __host__ __device__ void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  __host__ __device__ friend bool operator==(bare_allocator, bare_allocator)
  {
    return true;
  }
  __host__ __device__ friend bool operator!=(bare_allocator, bare_allocator)
  {
    return false;
  }
};

template <class T>
class no_default_allocator
{
  no_default_allocator() = delete;
  struct construct_tag
  {};
  __host__ __device__ explicit no_default_allocator(construct_tag) {}

public:
  __host__ __device__ static no_default_allocator create()
  {
    construct_tag tag;
    return no_default_allocator(tag);
  }

public:
  typedef T value_type;

  template <class U>
  __host__ __device__ no_default_allocator(no_default_allocator<U>) TEST_NOEXCEPT
  {}

  __host__ __device__ T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  __host__ __device__ void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  __host__ __device__ friend bool operator==(no_default_allocator, no_default_allocator)
  {
    return true;
  }
  __host__ __device__ friend bool operator!=(no_default_allocator, no_default_allocator)
  {
    return false;
  }
};

STATIC_TEST_GLOBAL_VAR size_t malloc_allocator_base_outstanding_bytes         = 0;
STATIC_TEST_GLOBAL_VAR size_t malloc_allocator_base_alloc_count               = 0;
STATIC_TEST_GLOBAL_VAR size_t malloc_allocator_base_dealloc_count             = 0;
STATIC_TEST_GLOBAL_VAR bool malloc_allocator_base_disable_default_constructor = false;

struct malloc_allocator_base
{
  __host__ __device__ static size_t outstanding_alloc()
  {
    assert(malloc_allocator_base_alloc_count >= malloc_allocator_base_dealloc_count);
    return (malloc_allocator_base_alloc_count - malloc_allocator_base_dealloc_count);
  }

  __host__ __device__ static void reset()
  {
    assert(outstanding_alloc() == 0);
    malloc_allocator_base_disable_default_constructor = false;
    malloc_allocator_base_outstanding_bytes           = 0;
    malloc_allocator_base_alloc_count                 = 0;
    malloc_allocator_base_dealloc_count               = 0;
  }
};

template <class T>
class malloc_allocator : public malloc_allocator_base
{
public:
  typedef T value_type;

  __host__ __device__ malloc_allocator() TEST_NOEXCEPT
  {
    assert(!malloc_allocator_base_disable_default_constructor);
  }

  template <class U>
  __host__ __device__ malloc_allocator(malloc_allocator<U>) TEST_NOEXCEPT
  {}

  __host__ __device__ T* allocate(cuda::std::size_t n)
  {
    const size_t nbytes = n * sizeof(T);
    ++malloc_allocator_base_alloc_count;
    malloc_allocator_base_outstanding_bytes += nbytes;
    return static_cast<T*>(cuda::std::malloc(nbytes));
  }

  __host__ __device__ void deallocate(T* p, cuda::std::size_t n) noexcept
  {
    const size_t nbytes = n * sizeof(T);
    ++malloc_allocator_base_dealloc_count;
    malloc_allocator_base_outstanding_bytes -= nbytes;
    cuda::std::free(static_cast<void*>(p));
  }

  __host__ __device__ friend bool operator==(malloc_allocator, malloc_allocator)
  {
    return true;
  }
  __host__ __device__ friend bool operator!=(malloc_allocator, malloc_allocator)
  {
    return false;
  }
};

STATIC_TEST_GLOBAL_VAR bool cpp03_allocator_construct_called = false;
template <class T>
struct cpp03_allocator : bare_allocator<T>
{
  typedef T value_type;
  typedef value_type* pointer;

  // Returned value is not used but it's not prohibited.
  __host__ __device__ pointer construct(pointer p, const value_type& val)
  {
    ::new (p) value_type(val);
    cpp03_allocator_construct_called = true;
    return p;
  }

  __host__ __device__ cuda::std::size_t max_size() const
  {
    return UINT_MAX / sizeof(T);
  }
};

STATIC_TEST_GLOBAL_VAR bool cpp03_overload_allocator_construct_called = false;
template <class T>
struct cpp03_overload_allocator : bare_allocator<T>
{
  typedef T value_type;
  typedef value_type* pointer;

  __host__ __device__ void construct(pointer p, const value_type& val)
  {
    construct(p, val, cuda::std::is_class<T>());
  }
  __host__ __device__ void construct(pointer p, const value_type& val, cuda::std::true_type)
  {
    ::new (p) value_type(val);
    cpp03_overload_allocator_construct_called = true;
  }
  __host__ __device__ void construct(pointer p, const value_type& val, cuda::std::false_type)
  {
    ::new (p) value_type(val);
    cpp03_overload_allocator_construct_called = true;
  }

  __host__ __device__ cuda::std::size_t max_size() const
  {
    return UINT_MAX / sizeof(T);
  }
};

template <class T, class = cuda::std::integral_constant<size_t, 0>>
class min_pointer;
template <class T, class ID>
class min_pointer<const T, ID>;
template <class ID>
class min_pointer<void, ID>;
template <class ID>
class min_pointer<const void, ID>;
template <class T>
class min_allocator;

template <class ID>
class min_pointer<const void, ID>
{
  const void* ptr_;

public:
  min_pointer() TEST_NOEXCEPT = default;
  __host__ __device__ min_pointer(cuda::std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
  template <class T>
  __host__ __device__ min_pointer(min_pointer<T, ID> p) TEST_NOEXCEPT : ptr_(p.ptr_)
  {}

  __host__ __device__ explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  __host__ __device__ friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  __host__ __device__ friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class ID>
class min_pointer<void, ID>
{
  void* ptr_;

public:
  min_pointer() TEST_NOEXCEPT = default;
  __host__ __device__ min_pointer(cuda::std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
  template <class T, class = typename cuda::std::enable_if<!cuda::std::is_const<T>::value>::type>
  __host__ __device__ min_pointer(min_pointer<T, ID> p) TEST_NOEXCEPT : ptr_(p.ptr_)
  {}

  __host__ __device__ explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  __host__ __device__ friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  __host__ __device__ friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class T, class ID>
class min_pointer
{
  T* ptr_;

  __host__ __device__ explicit min_pointer(T* p) TEST_NOEXCEPT : ptr_(p) {}

public:
  min_pointer() TEST_NOEXCEPT = default;
  __host__ __device__ min_pointer(cuda::std::nullptr_t) TEST_NOEXCEPT : ptr_(nullptr) {}
  __host__ __device__ explicit min_pointer(min_pointer<void, ID> p) TEST_NOEXCEPT : ptr_(static_cast<T*>(p.ptr_)) {}

  __host__ __device__ explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  typedef cuda::std::ptrdiff_t difference_type;
  typedef T& reference;
  typedef T* pointer;
  typedef T value_type;
  typedef cuda::std::random_access_iterator_tag iterator_category;

  __host__ __device__ reference operator*() const
  {
    return *ptr_;
  }
  __host__ __device__ pointer operator->() const
  {
    return ptr_;
  }

  __host__ __device__ min_pointer& operator++()
  {
    ++ptr_;
    return *this;
  }
  __host__ __device__ min_pointer operator++(int)
  {
    min_pointer tmp(*this);
    ++ptr_;
    return tmp;
  }

  __host__ __device__ min_pointer& operator--()
  {
    --ptr_;
    return *this;
  }
  __host__ __device__ min_pointer operator--(int)
  {
    min_pointer tmp(*this);
    --ptr_;
    return tmp;
  }

  __host__ __device__ min_pointer& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }
  __host__ __device__ min_pointer& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  __host__ __device__ min_pointer operator+(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp += n;
    return tmp;
  }

  __host__ __device__ friend min_pointer operator+(difference_type n, min_pointer x)
  {
    return x + n;
  }

  __host__ __device__ min_pointer operator-(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp -= n;
    return tmp;
  }

  __host__ __device__ friend difference_type operator-(min_pointer x, min_pointer y)
  {
    return x.ptr_ - y.ptr_;
  }

  __host__ __device__ reference operator[](difference_type n) const
  {
    return ptr_[n];
  }

  __host__ __device__ friend bool operator<(min_pointer x, min_pointer y)
  {
    return x.ptr_ < y.ptr_;
  }
  __host__ __device__ friend bool operator>(min_pointer x, min_pointer y)
  {
    return y < x;
  }
  __host__ __device__ friend bool operator<=(min_pointer x, min_pointer y)
  {
    return !(y < x);
  }
  __host__ __device__ friend bool operator>=(min_pointer x, min_pointer y)
  {
    return !(x < y);
  }

  __host__ __device__ static min_pointer pointer_to(T& t)
  {
    return min_pointer(cuda::std::addressof(t));
  }

  __host__ __device__ friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  __host__ __device__ friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
  template <class U>
  friend class min_allocator;
};

template <class T, class ID>
class min_pointer<const T, ID>
{
  const T* ptr_;

  __host__ __device__ explicit min_pointer(const T* p)
      : ptr_(p)
  {}

public:
  min_pointer() TEST_NOEXCEPT = default;
  __host__ __device__ min_pointer(cuda::std::nullptr_t)
      : ptr_(nullptr)
  {}
  __host__ __device__ min_pointer(min_pointer<T, ID> p)
      : ptr_(p.ptr_)
  {}
  __host__ __device__ explicit min_pointer(min_pointer<const void, ID> p)
      : ptr_(static_cast<const T*>(p.ptr_))
  {}

  __host__ __device__ explicit operator bool() const
  {
    return ptr_ != nullptr;
  }

  typedef cuda::std::ptrdiff_t difference_type;
  typedef const T& reference;
  typedef const T* pointer;
  typedef const T value_type;
  typedef cuda::std::random_access_iterator_tag iterator_category;

  __host__ __device__ reference operator*() const
  {
    return *ptr_;
  }
  __host__ __device__ pointer operator->() const
  {
    return ptr_;
  }

  __host__ __device__ min_pointer& operator++()
  {
    ++ptr_;
    return *this;
  }
  __host__ __device__ min_pointer operator++(int)
  {
    min_pointer tmp(*this);
    ++ptr_;
    return tmp;
  }

  __host__ __device__ min_pointer& operator--()
  {
    --ptr_;
    return *this;
  }
  __host__ __device__ min_pointer operator--(int)
  {
    min_pointer tmp(*this);
    --ptr_;
    return tmp;
  }

  __host__ __device__ min_pointer& operator+=(difference_type n)
  {
    ptr_ += n;
    return *this;
  }
  __host__ __device__ min_pointer& operator-=(difference_type n)
  {
    ptr_ -= n;
    return *this;
  }

  __host__ __device__ min_pointer operator+(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp += n;
    return tmp;
  }

  __host__ __device__ friend min_pointer operator+(difference_type n, min_pointer x)
  {
    return x + n;
  }

  __host__ __device__ min_pointer operator-(difference_type n) const
  {
    min_pointer tmp(*this);
    tmp -= n;
    return tmp;
  }

  __host__ __device__ friend difference_type operator-(min_pointer x, min_pointer y)
  {
    return x.ptr_ - y.ptr_;
  }

  __host__ __device__ reference operator[](difference_type n) const
  {
    return ptr_[n];
  }

  __host__ __device__ friend bool operator<(min_pointer x, min_pointer y)
  {
    return x.ptr_ < y.ptr_;
  }
  __host__ __device__ friend bool operator>(min_pointer x, min_pointer y)
  {
    return y < x;
  }
  __host__ __device__ friend bool operator<=(min_pointer x, min_pointer y)
  {
    return !(y < x);
  }
  __host__ __device__ friend bool operator>=(min_pointer x, min_pointer y)
  {
    return !(x < y);
  }

  __host__ __device__ static min_pointer pointer_to(const T& t)
  {
    return min_pointer(cuda::std::addressof(t));
  }

  __host__ __device__ friend bool operator==(min_pointer x, min_pointer y)
  {
    return x.ptr_ == y.ptr_;
  }
  __host__ __device__ friend bool operator!=(min_pointer x, min_pointer y)
  {
    return !(x == y);
  }
  template <class U, class XID>
  friend class min_pointer;
};

template <class T, class ID>
__host__ __device__ inline bool operator==(min_pointer<T, ID> x, cuda::std::nullptr_t)
{
  return !static_cast<bool>(x);
}

template <class T, class ID>
__host__ __device__ inline bool operator==(cuda::std::nullptr_t, min_pointer<T, ID> x)
{
  return !static_cast<bool>(x);
}

template <class T, class ID>
__host__ __device__ inline bool operator!=(min_pointer<T, ID> x, cuda::std::nullptr_t)
{
  return static_cast<bool>(x);
}

template <class T, class ID>
__host__ __device__ inline bool operator!=(cuda::std::nullptr_t, min_pointer<T, ID> x)
{
  return static_cast<bool>(x);
}

template <class T>
class min_allocator
{
public:
  typedef T value_type;
  typedef min_pointer<T> pointer;

  min_allocator() = default;
  template <class U>
  __host__ __device__ min_allocator(min_allocator<U>)
  {}

  __host__ __device__ pointer allocate(cuda::std::ptrdiff_t n)
  {
    return pointer(static_cast<T*>(::operator new(n * sizeof(T))));
  }

  __host__ __device__ void deallocate(pointer p, cuda::std::ptrdiff_t) noexcept
  {
    return ::operator delete(p.ptr_);
  }

  __host__ __device__ friend bool operator==(min_allocator, min_allocator)
  {
    return true;
  }
  __host__ __device__ friend bool operator!=(min_allocator, min_allocator)
  {
    return false;
  }
};

template <class T>
class explicit_allocator
{
public:
  typedef T value_type;

  __host__ __device__ explicit_allocator() TEST_NOEXCEPT {}

  template <class U>
  __host__ __device__ explicit explicit_allocator(explicit_allocator<U>) TEST_NOEXCEPT
  {}

  __host__ __device__ T* allocate(cuda::std::size_t n)
  {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  __host__ __device__ void deallocate(T* p, cuda::std::size_t) noexcept
  {
    return ::operator delete(static_cast<void*>(p));
  }

  __host__ __device__ friend bool operator==(explicit_allocator, explicit_allocator)
  {
    return true;
  }
  __host__ __device__ friend bool operator!=(explicit_allocator, explicit_allocator)
  {
    return false;
  }
};

#endif // MIN_ALLOCATOR_H
