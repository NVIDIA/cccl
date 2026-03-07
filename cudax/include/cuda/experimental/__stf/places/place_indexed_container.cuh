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
 * @brief Container that stores one value per place, optimized for device places.
 *
 * Storage is split by the place's devid:
 * - Places with a positive devid (GPU device index 0, 1, 2, ...) use a vector<T>
 *   indexed by devid, for O(1) access without map overhead.
 * - Places with any other devid (host, managed, invalid) or non-simple places
 *   (extensions, composite) use an unordered_map keyed by the place.
 *
 * Supports both data_place and exec_place via overloaded at() and operator[].
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#include <cuda/experimental/__stf/places/places.cuh>

#include <cuda_runtime.h>

#include <unordered_map>
#include <vector>

namespace cuda::experimental::stf
{

/**
 * @brief Stores one instance of T per place, with vector storage for device places (positive devid).
 *
 * Access by data_place or exec_place. When the place has a positive devid (simple device 0, 1, ...),
 * the value is stored in a vector; otherwise it is stored in a map keyed by the place.
 *
 * @tparam T Value type; must be default-constructible for device slots (vector storage).
 */
template <typename T>
class place_indexed_container
{
public:
  place_indexed_container()
  {
    int ndev = 0;
    cuda_safe_call(cudaGetDeviceCount(&ndev));
    _CCCL_ASSERT(ndev > 0, "At least one device is required");
    ndevices_ = static_cast<size_t>(ndev);
    device_storage_.resize(ndevices_);
  }
  
  /** @return Reference to the value for the given data_place; inserts a default T in the map if needed. */
  T& operator[](const data_place& p)
  {
    if (p.is_device())
    {
      const int d = device_ordinal(p);
      _CCCL_ASSERT(d >= 0 && static_cast<size_t>(d) < ndevices_, "device ordinal out of range");
      return device_storage_[static_cast<size_t>(d)];
    }
    return data_place_map_[p];
  }

  /** @return Reference to the value for the given exec_place; inserts a default T in the map if needed. */
  T& operator[](const exec_place& p)
  {
    if (p.is_device())
    {
      const int d = device_ordinal(p.affine_data_place());
      _CCCL_ASSERT(d >= 0 && static_cast<size_t>(d) < ndevices_, "device ordinal out of range");
      return device_storage_[static_cast<size_t>(d)];
    }
    return exec_place_map_[p];
  }

  /** @return Const reference for the given data_place; throws if not present in map. */
  const T& at(const data_place& p) const
  {
    if (p.is_device())
    {
      const int d = device_ordinal(p);
      if (d >= 0 && static_cast<size_t>(d) < ndevices_)
      {
        return device_storage_[static_cast<size_t>(d)];
      }
    }
    return data_place_map_.at(p);
  }

  /** @return Const reference for the given exec_place; throws if not present in map. */
  const T& at(const exec_place& p) const
  {
    if (p.is_device())
    {
      const int d = device_ordinal(p.affine_data_place());
      if (d >= 0 && static_cast<size_t>(d) < ndevices_)
      {
        return device_storage_[static_cast<size_t>(d)];
      }
    }
    return exec_place_map_.at(p);
  }

  /** @return Number of devices used for vector storage (device places with positive devid). */
  size_t device_count() const
  {
    return ndevices_;
  }

private:
  size_t ndevices_;
  ::std::vector<T> device_storage_;
  ::std::unordered_map<data_place, T, hash<data_place>> data_place_map_;
  ::std::unordered_map<exec_place, T, hash<exec_place>> exec_place_map_;
};

#ifdef UNITTESTED_FILE
UNITTEST("place_indexed_container device_count and data_place access")
{
  place_indexed_container<int> c;
  EXPECT(c.device_count() >= 1);

  c[data_place::device(0)] = 42;
  EXPECT(c[data_place::device(0)] == 42);
  EXPECT(c.at(data_place::device(0)) == 42);

  c[data_place::host()] = 10;
  c[data_place::managed()] = 20;
  EXPECT(c[data_place::host()] == 10);
  EXPECT(c[data_place::managed()] == 20);
  EXPECT(c.at(data_place::host()) == 10);
  EXPECT(c.at(data_place::managed()) == 20);

  c[data_place::device(0)] = 100;
  EXPECT(c[data_place::device(0)] == 100);
};

UNITTEST("place_indexed_container exec_place access")
{
  place_indexed_container<int> c;

  c[exec_place::device(0)] = 7;
  EXPECT(c[exec_place::device(0)] == 7);
  EXPECT(c.at(exec_place::device(0)) == 7);

  c[exec_place::host()] = 99;
  EXPECT(c[exec_place::host()] == 99);
  EXPECT(c.at(exec_place::host()) == 99);
};

UNITTEST("place_indexed_container data_place and exec_place device share vector")
{
  place_indexed_container<int> c;
  c[data_place::device(0)] = 11;
  EXPECT(c[exec_place::device(0)] == 11);
  c[exec_place::device(0)] = 22;
  EXPECT(c[data_place::device(0)] == 22);
};

UNITTEST("place_indexed_container multiple devices")
{
  int ndevices = cuda_try<cudaGetDeviceCount>();
  if (ndevices < 2)
  {
    return;
  }
  place_indexed_container<int> c;
  c[data_place::device(0)] = 1;
  c[data_place::device(1)] = 2;
  EXPECT(c[data_place::device(0)] == 1);
  EXPECT(c[data_place::device(1)] == 2);

  c[exec_place::device(0)] = 10;
  c[exec_place::device(1)] = 20;
  EXPECT(c[exec_place::device(0)] == 10);
  EXPECT(c[exec_place::device(1)] == 20);
};

UNITTEST("place_indexed_container at throws for missing map key")
{
  place_indexed_container<int> c;
  c[data_place::device(0)] = 1;
  bool threw = false;
  try
  {
    (void) c.at(data_place::host());
  }
  catch (const ::std::out_of_range&)
  {
    threw = true;
  }
  EXPECT(threw);
};
#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
