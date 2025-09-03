//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Reusable utilities to build asynchronous memory allocators and deal with memory pinning
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <algorithm>
#include <cstdint>

namespace cuda::experimental::stf
{

namespace reserved
{

// host-allocated pool, modeled as a hashtable keyed by size
inline auto& host_pool()
{
  static ::std::unordered_multimap<size_t, void*> result;
  return result;
}

// managed memory pool, modeled as a hashtable keyed by size
inline auto& managed_pool()
{
  static ::std::unordered_multimap<size_t, void*> result;
  return result;
}

// maximum number of entries in the host-allocated pool
enum : size_t
{
  maxPoolEntries = 16 * 1024
};

} // namespace reserved

/**
 * @brief Allocates memory on host and returns a pointer to it. Uses a pool of previous allocations.
 *
 * @param sz Number of bytes to allocate
 * @return void* pointer to allocated data
 */
inline void* allocateHostMemory(size_t sz)
{
  void* result;
  auto& pool = reserved::host_pool();
  if (auto i = pool.find(sz); i != pool.end())
  {
    result = i->second;
    pool.erase(i);
    return result;
  }
  if (pool.size() > reserved::maxPoolEntries)
  {
    // Lots of unused slots, so wipe pooled memory and start anew.
    for (auto& entry : pool)
    {
      cuda_safe_call(cudaFreeHost(entry.second));
    }
    pool.clear();
  }
  cuda_safe_call(cudaMallocHost(&result, sz));
  return result;
}

/**
 * @brief Allocates managed memory and returns a pointer to it. Uses a pool of previous allocations.
 *
 * @param sz Number of bytes to allocate
 * @return void* pointer to allocated data
 */
inline void* allocateManagedMemory(size_t sz)
{
  void* result;
  auto& pool = reserved::managed_pool();

  if (auto i = pool.find(sz); i != pool.end())
  {
    result = i->second;
    pool.erase(i);
    return result;
  }
  if (pool.size() > reserved::maxPoolEntries)
  {
    // Lots of unused slots, so wipe pooled memory and start anew.
    for (auto& entry : pool)
    {
      cuda_safe_call(cudaFree(entry.second));
    }
    pool.clear();
  }
  cuda_safe_call(cudaMallocManaged(&result, sz));
  return result;
}

/**
 * @brief Deallocates memory allocated with `allocateHostMemory` immediately.
 *
 * @param p pointer to memory chunk
 * @param sz size in bytes
 * @param loc location of the call, defaulted
 */
inline void deallocateHostMemory(
  void* p, size_t sz, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  ::std::ignore = loc;
  assert([&] {
    auto r = reserved::host_pool().equal_range(sz);
    for (auto i = r.first; i != r.second; ++i)
    {
      if (i->second == p)
      {
        fprintf(stderr, "%s(%u) WARNING: double deallocation\n", loc.file_name(), loc.line());
        return false;
      }
    }
    return true;
  }());
  reserved::host_pool().insert(::std::make_pair(sz, p));
}

/**
 * @brief Deallocates managed memory allocated with `allocateManagedMemory` immediately.
 *
 * @param p pointer to memory chunk
 * @param sz size in bytes
 * @param loc location of the call, defaulted
 */
inline void deallocateManagedMemory(
  void* p, size_t sz, const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  ::std::ignore = loc;
  assert([&] {
    auto r = reserved::managed_pool().equal_range(sz);
    for (auto i = r.first; i != r.second; ++i)
    {
      if (i->second == p)
      {
        fprintf(stderr, "%s(%u) WARNING: double deallocation\n", loc.file_name(), loc.line());
        return false;
      }
    }
    return true;
  }());
  reserved::managed_pool().insert(::std::make_pair(sz, p));
}

/**
 * @brief Deallocates memory allocated with `allocateHostMemory` in a stream-ordered fashion. Will perform
 * deallocation when all preceding operations on `stream` have completed.
 *
 * @param p pointer
 * @param sz size in bytes
 * @param stream the stream used for ordering
 */
inline void deallocateHostMemory(void* p, size_t sz, cudaStream_t stream)
{
  cuda_safe_call(cudaLaunchHostFunc(
    stream,
    [](void* vp) {
      auto args = static_cast<::std::pair<size_t, void*>*>(vp);
      deallocateHostMemory(args->second, args->first);
      delete args;
    },
    new ::std::pair<size_t, void*>(sz, p)));
}

/**
 * @brief Deallocates managed memory allocated with `allocateManagedMemory` in a stream-ordered fashion. Will perform
 * deallocation when all preceding operations on `stream` have completed.
 *
 * @param p pointer
 * @param sz size in bytes
 * @param stream the stream used for ordering
 */
inline void deallocateManagedMemory(void* p, size_t sz, cudaStream_t stream)
{
  cuda_safe_call(cudaLaunchHostFunc(
    stream,
    [](void* vp) {
      auto args = static_cast<::std::pair<size_t, void*>*>(vp);
      deallocateManagedMemory(args->second, args->first);
      delete args;
    },
    new ::std::pair<size_t, void*>(sz, p)));
}

/**
 * @brief Deallocates memory allocated with `allocateHostMemory` in a graph-ordered fashion. Will perform
 * deallocation when all dependent graph operations have completed.
 *
 * @param p pointer
 * @param sz size in byutes
 * @param graph graph used for ordering
 * @param pDependencies array of dependencies
 * @param numDependencies number of elements in `pDependencies`
 * @return cudaGraphNode_t the newly inserted node
 */
inline cudaGraphNode_t deallocateHostMemory(
  void* p, size_t sz, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies)
{
  const cudaHostNodeParams params = {
    .fn =
      [](void* vp) {
        auto args = static_cast<::std::pair<size_t, void*>*>(vp);
        deallocateHostMemory(args->second, args->first);
        delete args;
      },
    .userData = new ::std::pair<size_t, void*>(sz, p)};
  cudaGraphNode_t result;
  cuda_safe_call(cudaGraphAddHostNode(&result, graph, pDependencies, numDependencies, &params));
  return result;
}

// TODO deallocateManagedMemory graph...

/**
 * @brief Checks whether an address is pinned or not.
 *
 * Note that this call will erase the last CUDA error (see cudaGetLastError).
 *
 * @param p address to check
 * @return bool value indicating if p corresponds to a piece of memory that is already pinned or not.
 */
template <typename T>
bool address_is_pinned(T* p)
{
  cudaPointerAttributes attr;
  cuda_safe_call(cudaPointerGetAttributes(&attr, p));
  EXPECT(attr.type != cudaMemoryTypeDevice);
  return attr.type != cudaMemoryTypeUnregistered;
}

/**
 * @brief Pins host memory for efficient use with CUDA primitives
 *
 * @tparam T memory type
 * @param p pointer to beginning of memory block
 * @param n number of elements in the block
 */
template <typename T>
cudaError_t pin_memory(T* p, size_t n)
{
  assert(p);
  cudaError_t result = cudaSuccess;
  if (!address_is_pinned(p))
  {
    // We cast to (void *) because T may be a const type : we are not going
    // to modify the content, so this is legit ...
    using NonConstT = typename std::remove_const<T>::type;
    cudaHostRegister(const_cast<NonConstT*>(p), n * sizeof(T), cudaHostRegisterPortable);
    // Fetch the result and clear the last error
    result = cudaGetLastError();
  }
  return result;
}

/**
 * @brief Unpins host memory previously pinned with `pin_memory`
 *
 * @tparam T memory type
 * @param p pointer to beginning of memory block
 */
template <typename T>
void unpin_memory(T* p)
{
  assert(p);

  // Make sure no one did a mistake before ignoring the one that may come !
  cuda_safe_call(cudaGetLastError());

  // We cast to non const T * because T may be a const type : we are not going
  // to modify the content, so this is legit ...
  using NonConstT = typename std::remove_const<T>::type;
  if (cudaHostUnregister(const_cast<NonConstT*>(p)) == cudaErrorHostMemoryNotRegistered)
  {
    // Ignore that error, we probably also ignored an error about registering that buffer too !
    cudaGetLastError();
  }
}

/**
 * @brief Pins arrays in host memory
 */
template <typename T, size_t N>
cudaError_t pin_memory(T (&array)[N])
{
  return pin_memory(array, N);
}

/**
 * @brief Pins vectors in host memory
 */
template <typename T>
cudaError_t pin_memory(::std::vector<T>& v)
{
  return pin_memory(v.data(), v.size());
}

/**
 * @brief Unpin vectors in host memory
 */
template <typename T>
void unpin_memory(::std::vector<T>& v)
{
  unpin_memory(v.data());
}

#ifdef UNITTESTED_FILE
UNITTEST("pin_memory")
{
  ::std::vector<double> a(1024);

  EXPECT(!address_is_pinned(&a[0]));
  EXPECT(!address_is_pinned(&a[1023]));

  pin_memory(&a[0], 1024);
  EXPECT(address_is_pinned(&a[0]));
  EXPECT(address_is_pinned(&a[1023]));
  unpin_memory(&a[0]);

  EXPECT(!address_is_pinned(&a[0]));
  EXPECT(!address_is_pinned(&a[1023]));
};

UNITTEST("pin_memory const")
{
  ::std::vector<double> a(1024);

  const double* ca = &a[0];

  EXPECT(!address_is_pinned(ca));

  pin_memory(ca, 1024);
  EXPECT(address_is_pinned(ca));
  unpin_memory(ca);

  EXPECT(!address_is_pinned(ca));
};

#endif // UNITTESTED_FILE

template <typename T, size_t small_cap>
class small_vector
{
  // TODO: in some cases, uint16_t or uint8_t may be better.
  using small_size_t = uint32_t;

public:
  // Type definitions
  using value_type             = T;
  using size_type              = ::std::size_t;
  using difference_type        = ::std::ptrdiff_t;
  using reference              = value_type&;
  using const_reference        = const value_type&;
  using pointer                = value_type*;
  using const_pointer          = const value_type*;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = ::std::reverse_iterator<iterator>;
  using const_reverse_iterator = ::std::reverse_iterator<const_iterator>;

  // Constructors
  small_vector() = default;

  small_vector(const small_vector& rhs)
  {
    if (rhs.size() <= small_cap)
    {
      auto b = rhs.begin(), e = b + rhs.size();
      loop([&](auto i) {
        if (b >= e)
        {
          return false;
        }
        new (small_begin() + i) T(*b++);
        ++small_length;
        return true;
      });
    }
    else
    {
      new (&big())::std::vector<T>(rhs.big());
      small_length = small_size_t(-1);
    }
  }

  small_vector(small_vector&& rhs) noexcept
  {
    if (rhs.size() <= small_cap)
    {
      auto b = rhs.begin(), e = b + rhs.size();
      loop([&](auto i) {
        if (b >= e)
        {
          return false;
        }
        new (small_begin() + i) T(mv(*b++));
        ++small_length;
        return true;
      });
    }
    else
    {
      new (&big())::std::vector<T>(mv(rhs.big()));
      small_length = small_size_t(-1);
    }
  }

  small_vector(::std::initializer_list<T> init)
  {
    if (init.size() <= small_cap)
    {
      loop([&](auto i) {
        if (i >= init.size())
        {
          return false;
        }
        new (small_begin() + i) T(init.begin()[i]);
        ++small_length;
        return true;
      });
    }
    else
    {
      new (&big())::std::vector<T>(init);
      small_length = small_size_t(-1);
    }
  }

  // Assignment operators
  small_vector& operator=(const small_vector& rhs)
  {
    if (this == &rhs)
    {
      return *this;
    }

    clear();
    reserve(rhs.size());

    if (is_small())
    {
      loop([&](auto i) {
        if (i >= rhs.size())
        {
          return false;
        }
        new (small_begin() + i) T(rhs[i]);
        ++small_length;
        return true;
      });
    }
    else
    {
      big().assign(rhs.begin(), rhs.end());
    }

    return *this;
  }

  small_vector& operator=(small_vector&& rhs) noexcept
  {
    if (this == &rhs)
    {
      return *this;
    }

    if (is_small())
    {
      clear();
      if (rhs.is_small())
      {
        loop([&](auto i) {
          if (i >= rhs.small_length)
          {
            return false;
          }
          new (small_begin() + i) T(mv(rhs.small_begin()[i]));
          ++small_length;
          return true;
        });
      }
      else
      {
        new (&big())::std::vector<T>(mv(rhs.big()));
        small_length = small_size_t(-1);
      }
    }
    else
    {
      if (rhs.is_small())
      {
        big().assign(rhs.small_begin(), rhs.small_begin() + rhs.small_length);
      }
      else
      {
        big() = mv(rhs.big());
      }
    }

    return *this;
  }

  small_vector& operator=(::std::initializer_list<T> ilist);

  ~small_vector()
  {
    if (is_small())
    {
      loop([&](auto i) {
        if (i >= small_length)
        {
          return false;
        }
        small_begin()[i].~T();
        return true;
      });
    }
    else
    {
      using goner = ::std::vector<T>;
      big().~goner();
    }
  }

  // Element access
  T& operator[](size_t pos)
  {
    if (is_small())
    {
      assert(pos < small_length);
      return small_begin()[pos];
    }
    assert(pos < big().size());
    return big()[pos];
  }

  const T& operator[](size_t pos) const
  {
    if (is_small())
    {
      assert(pos < small_length);
      return small_begin()[pos];
    }
    assert(pos < big().size());
    return big()[pos];
  }

  T& at(size_t pos);
  const T& at(size_t pos) const;
  T& front()
  {
    return *begin();
  }
  const T& front() const
  {
    return *begin();
  }

  T& back();
  const T& back() const;
  T* data() noexcept;
  const T* data() const noexcept;

  // Iterators
  iterator begin() noexcept
  {
    return is_small() ? small_begin() : big().data();
  }
  const_iterator begin() const noexcept
  {
    return is_small() ? small_begin() : big().data();
  }

  const_iterator cbegin() const noexcept
  {
    return is_small() ? small_begin() : big().data();
  }

  iterator end() noexcept
  {
    return begin() + size();
  }

  const_iterator end() const noexcept
  {
    return begin() + size();
  }
  const_iterator cend() const noexcept
  {
    return begin() + size();
  }

  reverse_iterator rbegin() noexcept;
  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator crbegin() const noexcept;
  reverse_iterator rend() noexcept;
  const_reverse_iterator rend() const noexcept;
  const_reverse_iterator crend() const noexcept;

  // Capacity
  bool empty() const noexcept
  {
    return is_small() ? small_length == 0 : big().empty();
  }

  size_t size() const noexcept
  {
    return is_small() ? small_length : big().size();
  }

  size_t max_size() const noexcept
  {
    return big().max_size();
  }

  void reserve(size_t new_cap)
  {
    if (is_small())
    {
      if (new_cap <= small_cap)
      {
        return;
      }
      if (small_length > 0)
      {
        // Non-empty small_begin vector is a bit tricky
        ::std::vector<T> copy;
        copy.reserve(new_cap);
        for (auto& e : *this)
        {
          copy.push_back(mv(e));
        }
        clear();
        new (&big())::std::vector<T>(mv(copy));
        small_length = small_size_t(-1);
        return;
      }
      new (&big())::std::vector<T>();
      small_length = small_size_t(-1);
      // fall through to call reserve()
    }
    big().reserve(new_cap);
  }

  size_t capacity() const noexcept
  {
    return is_small() ? small_cap : big().capacity();
  }
  void shrink_to_fit();

  // Modifiers
  void clear() noexcept
  {
    if (is_small())
    {
      loop([&](auto i) {
        if (i >= small_length)
        {
          return false;
        }
        small_begin()[i].~T();
        return true;
      });
      small_length = 0;
    }
    else
    {
      big().clear();
    }
  }

  iterator insert(const_iterator pos, const T& value);
  iterator insert(const_iterator pos, T&& value);
  iterator insert(const_iterator pos, size_t count, const T& value);

  template <class InputIt>
  iterator insert(iterator pos, InputIt first, InputIt last)
  {
    if (is_small())
    {
      // Todo: support non-random iterators
      const std::size_t new_size = small_length + (last - first);
      if (new_size <= small_cap)
      {
        auto b = small_begin(), e = b + small_length;
        assert(b <= pos && pos <= e);
        assert(first <= last);
        auto result = ::std::move_backward(pos, e, e + (last - first));
        for (; first != last; ++first, ++pos, ++small_length)
        {
          new (pos) T(*first);
        }
        assert(size() == new_size);
        return result;
      }
      ::std::vector<T> copy;
      copy.reserve(new_size);
      copy.insert(copy.end(), ::std::move_iterator(small_begin()), ::std::move_iterator(pos));
      auto result = copy.insert(copy.end(), first, last);
      copy.insert(copy.end(), ::std::move_iterator(pos), ::std::move_iterator(small_begin() + small_length));
      new (&big())::std::vector<T>(mv(copy));
      small_length = small_size_t(-1);
      assert(size() == new_size);
      return big().data() + (result - big().begin());
    }
    else
    {
      auto result = big().insert(big().begin() + (pos - big().data()), first, last);
      return big().data() + (result - big().begin());
    }
  }

  iterator insert(const_iterator pos, ::std::initializer_list<T> ilist);
  template <class... Args>
  iterator emplace(const_iterator pos, Args&&... args);
  iterator erase(const_iterator pos);

  iterator erase(iterator first, iterator last)
  {
    if (is_small())
    {
      auto b = small_begin(), e = b + small_length;
      assert(b <= first && first <= last && last <= e);
      auto result = ::std::move(last, e, first);
      ::std::destroy(result, e);
      small_length -= static_cast<decltype(small_length)>(last - first);
      return result;
    }
    else
    {
      auto f_     = big().begin() + (first - big().data());
      auto l_     = big().begin() + (last - big().data());
      auto result = big().erase(f_, l_);
      return big().data() + (result - big().begin());
    }
  }

  void push_back(const T& value)
  {
    push_back(T(value)); // force an rvalue
  }

  void push_back(T&& value)
  {
    if (is_small())
    {
      if (small_length < small_cap)
      {
        new (small_begin() + small_length) T(mv(value));
        ++small_length;
        return;
      }
      reserve(small_cap * 2 + 1);
      assert(!is_small());
      // fall through to big case
    }
    big().push_back(mv(value));
  }

  template <class... Args>
  void emplace_back(Args&&... args);

  void pop_back()
  {
    if (is_small())
    {
      assert(small_length > 0);
      small_begin()[--small_length].~T();
    }
    else
    {
      assert(big().size() > 0);
      big().pop_back();
    }
  }

  void resize(size_t new_size)
  {
    if (is_small())
    {
      if (new_size <= small_length)
      {
        shrink_small(small_length - new_size);
      }
      else
      {
        resize(new_size, T());
      }
    }
    else
    {
      big().resize(new_size);
    }
  }

  void resize(size_t new_size, const value_type& value)
  {
    if (is_small())
    {
      if (new_size <= small_length)
      {
        shrink_small(small_length - new_size);
      }
      else
      {
        if (new_size <= small_cap)
        {
          ::std::uninitialized_fill(small_begin() + small_length, small_begin() + new_size, value);
          small_length = small_size_t(new_size);
        }
        else
        {
          // Small to big conversion, just create a new vector
          ::std::vector<T> copy;
          copy.reserve(new_size);
          copy.insert(
            copy.end(), ::std::move_iterator(small_begin()), ::std::move_iterator(small_begin() + small_length));
          copy.resize(new_size, value);
          clear(); // call destructors for the moved-from elements
          new (&big())::std::vector<T>(mv(copy));
          small_length = small_size_t(-1);
        }
      }
    }
    else
    {
      big().resize(new_size, value);
    }
  }

  void swap(small_vector& other) noexcept
  {
    if (is_small())
    {
      if (other.is_small())
      {
        if (small_length < other.small_length)
        {
          return other.swap(*this);
        }
        // We are longer, the other is shorter
        auto b = small_begin(), e = small_begin() + small_length, ob = other.small_begin(),
             oe = other.small_begin() + other.small_length;
        ::std::swap_ranges(ob, oe, b);
        ::std::uninitialized_move(b + other.small_length, e, oe);
        ::std::destroy(b + other.small_length, e);
        ::std::swap(small_length, other.small_length);
      }
      else
      {
        auto tmp    = mv(other.big());
        using goner = ::std::vector<T>;
        other.big().~goner();
        other.small_length = 0; // for exception safety
        new (&other) small_vector(mv(*this)); // this could theoretically throw
        // nothrow code from here down
        this->~small_vector();
        new (&big())::std::vector<T>(mv(tmp));
        small_length = small_size_t(-1);
      }
    }
    else
    {
      if (other.is_small())
      {
        other.swap(*this);
      }
      else
      {
        big().swap(other.big());
      }
    }
  }

  // Non-member functions
  friend bool operator==(const small_vector& lhs, const small_vector& rhs)
  {
    return lhs.size() == rhs.size() && ::std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }
  // friend bool operator!=(const small_vector& lhs, const small_vector& rhs);
  // friend bool operator<(const small_vector& lhs, const small_vector& rhs);
  // friend bool operator<=(const small_vector& lhs, const small_vector& rhs);
  // friend bool operator>(const small_vector& lhs, const small_vector& rhs);
  // friend bool operator>=(const small_vector& lhs, const small_vector& rhs);
  // friend void swap(small_vector& lhs, small_vector& rhs) noexcept;

private:
  auto small_begin()
  {
    return reinterpret_cast<T*>(&small_);
  }

  auto small_begin() const
  {
    return reinterpret_cast<const T*>(&small_);
  }

  auto& big()
  {
    return *reinterpret_cast<::std::vector<T>*>(&big_);
  }

  auto& big() const
  {
    return *reinterpret_cast<const ::std::vector<T>*>(&big_);
  }

  bool is_small() const
  {
    return small_length <= small_cap;
  }

  void shrink_small(size_t delta)
  {
    assert(delta <= small_length);
    assert(is_small());
    loop([&](auto i) {
      if (i >= delta)
      {
        return false;
      }
      small_begin()[small_length - 1 - i].~T();
      return true;
    });
    small_length -= delta;
  }

  template <typename F>
  void loop(F&& f)
  {
    if constexpr (small_cap < 16)
    {
      unroll<small_cap>(::std::forward<F>(f));
    }
    else
    {
      for (auto i : each(0, small_cap))
      {
        using result_t = decltype(f(::std::integral_constant<size_t, 0>()));
        if constexpr (::std::is_same_v<result_t, void>)
        {
          f(i);
        }
        else
        {
          if (!f(i))
          {
            return;
          }
        }
      }
    }
  }

  union
  {
    ::std::aligned_storage_t<sizeof(T), alignof(T)> small_[small_cap];
    ::std::aligned_storage_t<sizeof(::std::vector<T>), alignof(::std::vector<T>)> big_;
  };
  small_size_t small_length = 0;
};

#ifdef UNITTESTED_FILE
UNITTEST("small_vector basics")
{
  // Construction and initialization
  EXPECT(small_vector<int, 5>{}.empty());
  EXPECT(small_vector<char, 5>().capacity() == 5);
  EXPECT(small_vector<double, 1>({1.0, 2.5, 3.14})[2] == 3.14);

  // Element access and modification
  small_vector<::std::string, 1> v;
  v.push_back("Hello");
  v.push_back("World");
  EXPECT(v[0] + v[1] == "HelloWorld");
  v.pop_back();
  EXPECT(v.size() == 1);

  // Capacity and memory management
  small_vector<char, 1> v2;
  v2.reserve(10);
  EXPECT(v2.capacity() == 10);
  v2.push_back(true);
  EXPECT(v2.capacity() >= 10); // capacity may not change immediately

  // Basic algorithms
  small_vector<int, 3> v3 = {3, 1, 4, 2};
  ::std::sort(v3.begin(), v3.end());
  EXPECT(v3[0] == 1);
  EXPECT(v3[3] == 4);

  // Iteration
  for (auto element : small_vector<char, 3>{'a', 'b', 'c'})
  {
    EXPECT(element < 'd');
  }

  // Vector of non-copyable objects
  small_vector<::std::unique_ptr<int>, 8> v4;
  v4.reserve(10);
  v4.push_back(::std::make_unique<int>(42));
  v4.push_back(::std::make_unique<int>(5));
};
#endif // UNITTESTED_FILE

namespace reserved
{

/*!
 * @brief A simple object pool with linear search for managing objects of type `T`.
 *
 * The `linear_pool` class provides a basic mechanism for reusing objects of a
 * specific type. It stores a collection of objects and allows retrieval of
 * existing objects with matching parameters or creation of new objects if
 * necessary.
 *
 * @tparam T The type of objects to be managed by the pool.
 */
template <class T>
class linear_pool
{
public:
  /*!
   * @brief Constructs a new, empty linear pool.
   */
  linear_pool() = default;

  /*!
   * @brief Adds an object to the pool.
   *
   * @param p A pointer to the object to be added. Must be non-null.
   */
  void put(::std::unique_ptr<T> p)
  {
    EXPECT(p); // Enforce that the pointer is not null.
    payload.push_back(mv(p));
  }

  /*!
   * @brief Retrieves an object from the pool with matching parameters or
   *        creates a new one if necessary.
   *
   * @tparam P The types of the parameters to match.
   * @param p The parameters to match.
   * @return A pointer to an object with the specified parameters.
   */
  template <typename... P>
  ::std::unique_ptr<T> get(P&&... p)
  {
    for (auto it = payload.begin(); it != payload.end(); ++it)
    {
      T* e = it->get();
      assert(e);
      if (*e == ::std::tuple<const P&...>(p...))
      {
        it->release();
        // Move the last element to replace the retrieved element,
        // maintaining a compact pool.
        if (it + 1 < payload.end())
        {
          *it = mv(payload.back());
        }
        payload.pop_back();
        return ::std::unique_ptr<T>(e);
      }
    }

    // If no matching object is found, create a new one.
    return ::std::make_unique<T>(::std::forward<P>(p)...);
  }

  /*!
   * @brief Calls a function object on each object in the pool.
   *
   * @tparam F The type of the function to be called.
   * @param f The function to call on each object.
   */
  template <typename F>
  void each(F&& f)
  {
    for (auto& ptr : payload)
    {
      assert(ptr);
      f(*ptr);
    }
  }

private:
  /*!
   * @brief The collection of objects in the pool.
   */
  ::std::vector<::std::unique_ptr<T>> payload;
};

} // end namespace reserved

} // namespace cuda::experimental::stf
