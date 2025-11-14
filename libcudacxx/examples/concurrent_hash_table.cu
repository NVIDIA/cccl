// Copyright (c) 2018-2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Released under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

#include <cuda/std/atomic>
#include <cuda/std/cstdint>

// TODO: It would be great if this example could NOT depend on Thrust.
#include <thrust/allocate_unique.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/universal_allocator.h>
#include <thrust/universal_vector.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <random>

template <typename Key,
          typename Value,
          typename Hash           = cuda::std::identity,
          typename KeyEqual       = cuda::std::equal_to<Key>,
          typename MemoryResource = thrust::universal_memory_resource>
struct concurrent_hash_table
{
  // Elements transition from state_empty -> state_reserved ->
  // state_filled; no other transitions are allowed.
  enum state_type
  {
    state_empty,
    state_reserved,
    state_filled
  };

  using key_type    = Key;
  using mapped_type = Value;
  using size_type   = cuda::std::uint64_t;

  using key_allocator    = thrust::mr::stateless_resource_allocator<key_type, MemoryResource>;
  using mapped_allocator = thrust::mr::stateless_resource_allocator<mapped_type, MemoryResource>;
  using state_allocator  = thrust::mr::stateless_resource_allocator<cuda::std::atomic<state_type>, MemoryResource>;

  using key_iterator   = typename key_allocator::pointer;
  using value_iterator = typename mapped_allocator::pointer;
  using state_iterator = typename state_allocator::pointer;

  // This whole thing is silly and should be a lambda, or at least a private
  // nested class, but alas, NVCC doesn't like that.
  struct element_destroyer
  {
  private:
    size_type capacity_;
    key_iterator keys_;
    value_iterator values_;
    state_iterator states_;

  public:
    __host__ __device__
    element_destroyer(size_type capacity, key_iterator keys, value_iterator values, state_iterator states)
        : capacity_(capacity)
        , keys_(keys)
        , values_(values)
        , states_(states)
    {}

    element_destroyer(element_destroyer const&) = default;

    __host__ __device__ void operator()(size_type i)
    {
      if (state_empty != states_[i])
      {
        (keys_ + i)->~key_type();
        (values_ + i)->~mapped_type();
      }
    }
  };

private:
  size_type capacity_;
  key_iterator keys_;
  value_iterator values_;
  state_iterator states_;
  Hash hash_;
  KeyEqual key_equal_;

public:
  __host__ concurrent_hash_table(size_type capacity, Hash hash = Hash(), KeyEqual key_equal = KeyEqual())
      : capacity_(capacity)
      , keys_(key_allocator{}.allocate(capacity_))
      , values_(mapped_allocator{}.allocate(capacity_))
      , states_(state_allocator{}.allocate(capacity_))
      , hash_(std::move(hash))
      , key_equal_(std::move(key_equal))
  {
    thrust::uninitialized_fill(thrust::device, states_, states_ + capacity_, state_empty);
  }

  __host__ ~concurrent_hash_table()
  {
    thrust::for_each(thrust::device,
                     thrust::counting_iterator<size_type>(0),
                     thrust::counting_iterator<size_type>(capacity_),
                     element_destroyer(capacity_, keys_, values_, states_));
  }

  // TODO: Change return type to an enum with three possible values, succeeded,
  // exists, and full.
  template <typename UKey, typename... Args>
  __host__ __device__ cuda::std::pair<value_iterator, bool> try_emplace(UKey&& key, Args&&... args)
  {
    auto index{hash_(key) % capacity_};
    // Linearly probe the storage space up to `capacity_` times; if we haven't
    // succeeded by then, the container is full.
    for (size_type i = 0; i < capacity_; ++i)
    {
      state_type old = states_[index].load(cuda::std::memory_order_acquire);
      while (old == state_empty)
      {
        // As long as the state of this element is empty, attempt to set it to
        // reserved.
        if (states_[index].compare_exchange_weak(old, state_reserved, cuda::std::memory_order_acq_rel))
        {
          // We succeeded; the element is now "locked" as reserved.
          new (keys_ + index) key_type(std::forward<UKey>(key));
          new (values_ + index) mapped_type(std::forward<Args>(args)...);
          states_[index].store(state_filled, cuda::std::memory_order_release);
          return cuda::std::make_pair(values_ + index, true);
        }
      }
      // If we are here, the element we are probing is not empty and we didn't
      // fill it, so we need to wait for it to be filled.
      while (state_filled != states_[index].load(cuda::std::memory_order_acquire))
        ;
      // Now we know that the element we are probing has been filled by someone
      // else, so we check if our key is equal to it.
      if (key_equal_(keys_[index], key))
      {
        // It is, so the element already exists.
        return cuda::std::make_pair(values_ + index, false);
      }
      // Otherwise, the element isn't a match, so move on to the next element.
      index = (index + 1) % capacity_;
    }
    // If we are here, the container is full.
    return cuda::std::make_pair(value_iterator{}, false);
  }

  __host__ __device__ mapped_type& operator[](key_type const& key)
  {
    return (*try_emplace(key).first);
  }
  __host__ __device__ mapped_type& operator[](key_type&& key)
  {
    return (*try_emplace(std::move(key)).first);
  }
};

template <typename T>
struct identity_modulo
{
private:
  T const modulo_;

public:
  __host__ __device__ identity_modulo(T modulo)
      : modulo_(std::move(modulo))
  {}

  identity_modulo(identity_modulo const&) = default;

  __host__ __device__ T operator()(T i)
  {
    return i % modulo_;
  }
};

int main()
{
  {
    using table = concurrent_hash_table<int, cuda::std::atomic<int>>;

    auto freq = thrust::allocate_unique<table>(thrust::universal_allocator<table>{}, 8);

    thrust::universal_vector<int> input = [] {
      thrust::universal_vector<int> v(2048);
      std::mt19937 gen(1337);
      std::uniform_int_distribution<long> dis(0, 7);
      thrust::generate(v.begin(), v.end(), [&] {
        return dis(gen);
      });
      return v;
    }();

    thrust::for_each(thrust::device, input.begin(), input.end(), [freq = freq.get()] __device__(int i) {
      (*freq)[i].fetch_add(1, cuda::std::memory_order_relaxed);
    });

    thrust::host_vector<int> gold(8);
    thrust::for_each(input.begin(), input.end(), [&](int i) {
      ++gold[i];
    });

    for (cuda::std::uint64_t i = 0; i < 8; ++i)
    {
      std::cout << "i: " << i << " gold: " << gold[i] << " observed: " << (*freq)[i] << "\n";
    }

    assert(cudaSuccess == cudaDeviceSynchronize());
  }
  {
    using table = concurrent_hash_table<int, cuda::std::atomic<int>, identity_modulo<int>>;

    auto freq = thrust::allocate_unique<table>(thrust::universal_allocator<table>{}, 8, identity_modulo<int>(4));

    thrust::universal_vector<int> input = [] {
      thrust::universal_vector<int> v(2048);
      std::mt19937 gen(1337);
      std::uniform_int_distribution<long> dis(0, 7);
      thrust::generate(v.begin(), v.end(), [&] {
        return dis(gen);
      });
      return v;
    }();

    thrust::for_each(thrust::device, input.begin(), input.end(), [freq = freq.get()] __device__(int i) {
      (*freq)[i].fetch_add(1, cuda::std::memory_order_relaxed);
    });

    thrust::host_vector<int> gold(8);
    thrust::for_each(input.begin(), input.end(), [&](int i) {
      ++gold[i];
    });

    for (cuda::std::uint64_t i = 0; i < 8; ++i)
    {
      std::cout << "i: " << i << " gold: " << gold[i] << " observed: " << (*freq)[i] << "\n";
    }

    assert(cudaSuccess == cudaDeviceSynchronize());
  }
}
