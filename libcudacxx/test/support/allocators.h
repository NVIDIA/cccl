//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
class A1
{
  int id_;

public:
  __device__ __host__ explicit A1(int id = 0) TEST_NOEXCEPT : id_(id) {}

  typedef T value_type;

  __device__ __host__ int id() const
  {
    return id_;
  }

  STATIC_MEMBER_VAR(copy_called, bool);
  STATIC_MEMBER_VAR(move_called, bool);
  STATIC_MEMBER_VAR(allocate_called, bool);

  __device__ __host__ static cuda::std::pair<T*, cuda::std::size_t>& deallocate_called()
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (__shared__ cuda::std::pair<T*, cuda::std::size_t> v; return v;),
                      (static cuda::std::pair<T*, cuda::std::size_t> v = 0; return v;))
  }

  __device__ __host__ A1(const A1& a) TEST_NOEXCEPT : id_(a.id())
  {
    copy_called() = true;
  }
  __device__ __host__ A1(A1&& a) TEST_NOEXCEPT : id_(a.id())
  {
    move_called() = true;
  }
  __device__ __host__ A1& operator=(const A1& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    copy_called() = true;
    return *this;
  }
  __device__ __host__ A1& operator=(A1&& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    move_called() = true;
    return *this;
  }

  template <class U>
  __device__ __host__ A1(const A1<U>& a) TEST_NOEXCEPT : id_(a.id())
  {
    copy_called() = true;
  }
  template <class U>
  __device__ __host__ A1(A1<U>&& a) TEST_NOEXCEPT : id_(a.id())
  {
    move_called() = true;
  }

  __device__ __host__ T* allocate(cuda::std::size_t n)
  {
    allocate_called() = true;
    return (T*) n;
  }

  __device__ __host__ void deallocate(T* p, cuda::std::size_t n) noexcept
  {
    deallocate_called() = cuda::std::pair<T*, cuda::std::size_t>(p, n);
  }

  __device__ __host__ cuda::std::size_t max_size() const
  {
    return id_;
  }
};

template <class T, class U>
inline __device__ __host__ bool operator==(const A1<T>& x, const A1<U>& y)
{
  return x.id() == y.id();
}

template <class T, class U>
inline __device__ __host__ bool operator!=(const A1<T>& x, const A1<U>& y)
{
  return !(x == y);
}

template <class T>
class A2
{
  int id_;

public:
  __device__ __host__ explicit A2(int id = 0) TEST_NOEXCEPT : id_(id) {}

  typedef T value_type;

  typedef unsigned size_type;
  typedef int difference_type;

  typedef cuda::std::true_type propagate_on_container_move_assignment;

  __device__ __host__ int id() const
  {
    return id_;
  }

  STATIC_MEMBER_VAR(copy_called, bool);
  STATIC_MEMBER_VAR(move_called, bool);
  STATIC_MEMBER_VAR(allocate_called, bool);

  __device__ __host__ A2(const A2& a) TEST_NOEXCEPT : id_(a.id())
  {
    copy_called() = true;
  }
  __device__ __host__ A2(A2&& a) TEST_NOEXCEPT : id_(a.id())
  {
    move_called() = true;
  }
  __device__ __host__ A2& operator=(const A2& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    copy_called() = true;
    return *this;
  }
  __device__ __host__ A2& operator=(A2&& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    move_called() = true;
    return *this;
  }

  __device__ __host__ T* allocate(cuda::std::size_t, const void* hint)
  {
    allocate_called() = true;
    return (T*) const_cast<void*>(hint);
  }
};

template <class T, class U>
inline __device__ __host__ bool operator==(const A2<T>& x, const A2<U>& y)
{
  return x.id() == y.id();
}

template <class T, class U>
inline __device__ __host__ bool operator!=(const A2<T>& x, const A2<U>& y)
{
  return !(x == y);
}

template <class T>
class A3
{
  int id_;

public:
  __device__ __host__ explicit A3(int id = 0) TEST_NOEXCEPT : id_(id) {}

  typedef T value_type;

  typedef cuda::std::true_type propagate_on_container_copy_assignment;
  typedef cuda::std::true_type propagate_on_container_swap;

  __device__ __host__ int id() const
  {
    return id_;
  }

  STATIC_MEMBER_VAR(copy_called, bool);
  STATIC_MEMBER_VAR(move_called, bool);
  STATIC_MEMBER_VAR(constructed, bool);
  STATIC_MEMBER_VAR(destroy_called, bool);

  __device__ __host__ A3(const A3& a) TEST_NOEXCEPT : id_(a.id())
  {
    copy_called() = true;
  }
  __device__ __host__ A3(A3&& a) TEST_NOEXCEPT : id_(a.id())
  {
    move_called() = true;
  }
  __device__ __host__ A3& operator=(const A3& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    copy_called() = true;
    return *this;
  }
  __device__ __host__ A3& operator=(A3&& a) TEST_NOEXCEPT
  {
    id_           = a.id();
    move_called() = true;
    return *this;
  }

  template <class U, class... Args>
  __device__ __host__ void construct(U* p, Args&&... args)
  {
    ::new (p) U(cuda::std::forward<Args>(args)...);
    constructed() = true;
  }

  template <class U>
  __device__ __host__ void destroy(U* p) noexcept
  {
    p->~U();
    destroy_called() = true;
  }

  __device__ __host__ A3 select_on_container_copy_construction() const
  {
    return A3(-1);
  }
};

template <class T, class U>
inline __device__ __host__ bool operator==(const A3<T>& x, const A3<U>& y)
{
  return x.id() == y.id();
}

template <class T, class U>
inline __device__ __host__ bool operator!=(const A3<T>& x, const A3<U>& y)
{
  return !(x == y);
}

#endif // ALLOCATORS_H
