/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>

#include <cub/detail/cpp_compatibility.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <climits>
#include <cstdint>

#include "c2h/utility.cuh"
#include "catch2_test_helper.h"

// The launchers defined in catch2_test_launch_helper.h do not support
// passing objects by reference since the device-launch tests cannot
// pass references to a __global__ function. The DoubleBuffer object
// must be passed by reference to the radix sort APIs so that the selector
// can be updated appropriately for the caller. This wrapper allows the
// selector to be updated in a way that's compatible with the launch helpers.
// Call initialize() before using to allocate temporary memory, and finalize()
// when finished to release.
struct double_buffer_sort_t
{
private:
  bool m_is_descending;
  int* m_selector;

public:
  explicit double_buffer_sort_t(bool is_descending)
  : m_is_descending(is_descending),
    m_selector(nullptr)
  {
  }

  void initialize()
  {
    cudaMallocHost(&m_selector, sizeof(int));
  }

  void finalize()
  {
    cudaFreeHost(m_selector);
    m_selector = nullptr;
  }

  int selector() const { return *m_selector;}

  template <class KeyT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t
  operator()(std::uint8_t* d_temp_storage, std::size_t& temp_storage_bytes, cub::DoubleBuffer<KeyT> keys, As... as)
  {
    const cudaError_t status =
      m_is_descending ? cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, keys, as...)
                      : cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, keys, as...);

    *m_selector = keys.selector;
    return status;
  }

  template <class KeyT, class ValueT, class... As>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(
    std::uint8_t* d_temp_storage,
    std::size_t& temp_storage_bytes,
    cub::DoubleBuffer<KeyT> keys,
    cub::DoubleBuffer<ValueT> values,
    As... as)
  {
    const cudaError_t status =
      m_is_descending ? cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, keys, values, as...)
                      : cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, values, as...);

    *m_selector = keys.selector;
    return status;
  }
};

template <class KeyT>
thrust::host_vector<KeyT>
get_striped_keys(const thrust::host_vector<KeyT> &h_keys,
                 int begin_bit,
                 int end_bit)
{
  thrust::host_vector<KeyT> h_striped_keys(h_keys);
  KeyT *h_striped_keys_data = thrust::raw_pointer_cast(h_striped_keys.data());

  using traits_t                    = cub::Traits<KeyT>;
  using bit_ordered_t               = typename traits_t::UnsignedBits;

  const int num_bits = end_bit - begin_bit;

  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    bit_ordered_t key = c2h::bit_cast<bit_ordered_t>(h_keys[i]);

    CUB_IF_CONSTEXPR(traits_t::CATEGORY == cub::FLOATING_POINT)
    {
      const bit_ordered_t negative_zero = bit_ordered_t(1) << bit_ordered_t(sizeof(bit_ordered_t) * 8 - 1);

      if (key == negative_zero)
      {
        key = 0;
      }
    }

    key = traits_t::TwiddleIn(key);

    if ((begin_bit > 0) || (end_bit < static_cast<int>(sizeof(KeyT) * 8)))
    {
      key &= ((bit_ordered_t{1} << num_bits) - 1) << begin_bit;
    }

    // striped keys are used to compare bit ordered representation of keys,
    // so we do not twiddle-out the key here:
    // key = traits_t::TwiddleOut(key);

    memcpy(h_striped_keys_data + i, &key, sizeof(KeyT));
}

  return h_striped_keys;
}

template <class T>
struct indirect_binary_comparator_t
{
  const T* h_ptr{};
  bool is_descending{};

  indirect_binary_comparator_t(const T* h_ptr, bool is_descending)
      : h_ptr(h_ptr)
      , is_descending(is_descending)
  {}

  bool operator()(std::size_t a, std::size_t b)
  {
    if (is_descending)
    {
      return h_ptr[a] > h_ptr[b];
    }

    return h_ptr[a] < h_ptr[b];
  }
};

template <class KeyT>
thrust::host_vector<std::size_t>
get_permutation(const thrust::host_vector<KeyT> &h_keys,
                bool is_descending,
                int begin_bit,
                int end_bit)
{
  thrust::host_vector<KeyT> h_striped_keys =
    get_striped_keys(h_keys, begin_bit, end_bit);

  thrust::host_vector<std::size_t> h_permutation(h_keys.size());
  thrust::sequence(h_permutation.begin(), h_permutation.end());

  using traits_t = cub::Traits<KeyT>;
  using bit_ordered_t = typename traits_t::UnsignedBits;

  auto bit_ordered_striped_keys =
    reinterpret_cast<const bit_ordered_t*>(thrust::raw_pointer_cast(h_striped_keys.data()));

  std::stable_sort(h_permutation.begin(),
                   h_permutation.end(),
                   indirect_binary_comparator_t<bit_ordered_t>{bit_ordered_striped_keys, is_descending});

  return h_permutation;
}

template <class KeyT>
thrust::host_vector<KeyT>
radix_sort_reference(const thrust::device_vector<KeyT> &d_keys,
                     bool is_descending,
                     int begin_bit = 0,
                     int end_bit = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  thrust::host_vector<KeyT> h_keys(d_keys);
  thrust::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, begin_bit, end_bit);
  thrust::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT>
std::pair<thrust::host_vector<KeyT>, thrust::host_vector<ValueT>>
radix_sort_reference(const thrust::device_vector<KeyT> &d_keys,
                     const thrust::device_vector<ValueT> &d_values,
                     bool is_descending,
                     int begin_bit = 0,
                     int end_bit = static_cast<int>(sizeof(KeyT) * CHAR_BIT))
{
  std::pair<thrust::host_vector<KeyT>, thrust::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  thrust::host_vector<KeyT> h_keys(d_keys);
  thrust::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, begin_bit, end_bit);

  thrust::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}
