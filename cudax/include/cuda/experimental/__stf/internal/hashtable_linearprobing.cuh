//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief A simple hashtable implementation based on https://nosferalatu.com/SimpleGPUHashTable.html
 * The goal of this class is to illustrate the extensibility of our data interface mechanism.
 */

#pragma once

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

namespace cuda::experimental::stf
{

const uint32_t kEmpty = 0xffffffff;

/**
 * @brief Key/value pair for storing in a hashtable in device memory
 *
 */
class KeyValue
{
public:
  ///@{ @name Constructors
  KeyValue() = default;
  CUDASTF_HOST_DEVICE KeyValue(uint32_t key, uint32_t value)
      : key(key)
      , value(value)
  {}
  ///@}
  uint32_t key;
  uint32_t value;
};

/* Default capacity */
const ::std::uint32_t kHashTableCapacity = 64 * 1024 * 1024;

/**
 * @brief A simple hashtable that maps `size_t` to `size_t`
 *
 */
class hashtable
{
public:
  /** @brief Default constructor (runs on host) */
  hashtable(uint32_t capacity = kHashTableCapacity)
      : capacity(capacity)
  {
    init();
  }

  /** @brief Constructor from pointer to `KeyValue` (runs  on host or device) */
  CUDASTF_HOST_DEVICE hashtable(KeyValue* addr, uint32_t capacity = kHashTableCapacity)
      : addr(addr)
      , capacity(capacity)
  {}

  hashtable(const hashtable&) = default;

  CUDASTF_HOST_DEVICE ~hashtable()
  {
    if (automatically_allocated)
    {
      // TODO destroy
    }
  }

  /**
   * @brief TODO
   *
   */
  void cpu_cat() const
  {
    for (size_t i = 0; i < capacity; i++)
    {
      if (addr[i].value != kEmpty)
      {
        fprintf(stderr, "VALID ENTRY at slot %ld, value %d key %d\n", i, addr[i].value, addr[i].key);
      }
    }
  }

  /**
   * @brief Get the entry with the requested key, if any
   *
   */
  CUDASTF_HOST_DEVICE uint32_t get(uint32_t key) const
  {
    uint32_t slot = hash(key);
    while (true)
    {
      uint32_t prev = addr[slot].key;
      if (prev == kEmpty || prev == key)
      {
        return addr[slot].value;
      }
      slot = (slot + 1) & (capacity - 1);
    }
  }

  /**
   * @brief TODO
   *
   */
  CUDASTF_HOST_DEVICE void insert(uint32_t key, uint32_t value)
  {
    insert(KeyValue(key, value));
  }

  /**
   * @brief Introduce a pair of key/value in a hashtable
   *
   */
  CUDASTF_HOST_DEVICE void insert(const KeyValue& kvs)
  {
    uint32_t key   = kvs.key;
    uint32_t value = kvs.value;
    uint32_t slot  = hash(key);

    while (true)
    {
#if defined(__CUDA_ARCH__)
      uint32_t prev = atomicCAS(&addr[slot].key, kEmpty, key);
#else
      uint32_t prev = addr[slot].key;
#endif
      if (prev == kEmpty || prev == key)
      {
        addr[slot].value = value;
#if !defined(__CUDA_ARCH__)
        addr[slot].key = key;
#endif
        // printf("INSERT VALUE %d key %d at slot %d\n", value, key, slot);
        return;
      }
      slot = (slot + 1) & (capacity - 1);
    }
  }

  KeyValue* addr;

  CUDASTF_HOST_DEVICE
  size_t get_capacity() const
  {
    return capacity;
  }

private:
  mutable size_t capacity;

  // 32 bit Murmur3 hash
  inline CUDASTF_HOST_DEVICE uint32_t hash(uint32_t k) const
  {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity - 1);
  }

  // Initialization of the table (host memory)
  void init()
  {
    size_t sz = capacity * sizeof(KeyValue);
    cuda_try(cudaHostAlloc(&addr, sz, cudaHostAllocMapped));
    memset(addr, 0xff, sz);
    automatically_allocated = true;
  }

  bool automatically_allocated = false;
};

template <typename>
class shape_of;

/**
 * @brief defines the shape of a hashtable
 *
 * @extends shape_of
 */
template <>
class shape_of<hashtable>
{
public:
  /**
   * @brief Initialize with a specific capacity
   */
  shape_of(uint32_t capacity = kHashTableCapacity)
      : capacity(capacity)
  {}

  /**
   * @name Copies a shape.
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of(const shape_of&) = default;

  /**
   * @brief Extracts the shape from a hashtable
   *
   * @param h hashtable to get the shape from
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of(const hashtable& h)
      : shape_of(h.get_capacity())
  {}

  // This size() value does not correspond to the actual amount of bytes
  // transfered, but this value is only used for stat / scheduling purposes.
  // We may just have an approximate reporting ...
  //
  // In practice, the current implementation effectively transfers the whole
  // array so it is correct.
  CUDASTF_HOST_DEVICE
  size_t size() const
  {
    return capacity * sizeof(uint32_t);
  }

  CUDASTF_HOST_DEVICE
  uint32_t get_capacity() const
  {
    return capacity;
  }

private:
  mutable uint32_t capacity;
};

} // end namespace cuda::experimental::stf

template <>
struct std::hash<cuda::experimental::stf::hashtable>
{
  ::std::size_t operator()(cuda::experimental::stf::hashtable const& s) const noexcept
  {
    return ::std::hash<cuda::experimental::stf::KeyValue*>{}(s.addr);
  }
};