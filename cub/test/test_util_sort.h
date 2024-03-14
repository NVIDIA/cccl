/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <set>
#include <utility>

/**
 * Check that an array is a permutation of another.
 */
template <typename KeyT>
bool IsPermutationOf(const KeyT* h_keys0, const KeyT* h_keys1, size_t size)
{
  std::multiset<KeyT> key_set;

  // Add elements of the first array to the multiset.
  for (size_t i = 0; i < size; i++)
  {
    key_set.insert(h_keys0[i]);
  }

  // Remove elements of the second array from the multiset, return false
  // if not found.
  for (size_t i = 0; i < size; i++)
  {
    auto it = key_set.find(h_keys1[i]);
    if (it == key_set.end())
    {
      return false;
    }
    key_set.erase(it);
  }

  return true;
}

/**
 * Check that the key-value pairs from two arrays are a permutation of the
 * key-value pairs from two other arrays.
 */
template <typename KeyT, typename ValueT>
bool IsPermutationOf(
  const KeyT* h_keys0, const KeyT* h_keys1, const ValueT* h_values0, const ValueT* h_values1, size_t size)
{
  std::multiset<std::pair<KeyT, ValueT>> key_set;

  // Add pairs of the first array pair to the multiset.
  for (size_t i = 0; i < size; i++)
  {
    key_set.insert(std::make_pair(h_keys0[i], h_values0[i]));
  }

  // Remove pairs of the second array pair from the multiset, return false
  // if not found.
  for (size_t i = 0; i < size; i++)
  {
    auto it = key_set.find(std::make_pair(h_keys1[i], h_values1[i]));
    if (it == key_set.end())
    {
      return false;
    }
    key_set.erase(it);
  }

  return true;
}

/**
 * Check that keys are sorted according to the given comparison operator.
 */
template <typename RandomItT, typename CompareOp>
bool IsSorted(RandomItT data_begin, RandomItT data_end, const CompareOp& compare_op)
{
  if (data_begin == data_end)
  {
    return true;
  }
  for (auto it = data_begin; it + 1 != data_end; it++)
  {
    if (compare_op(*(it + 1), *it))
    {
      return false;
    }
  }
  return true;
}
