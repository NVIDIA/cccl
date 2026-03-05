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
 * @brief Resource container indexed by place (data_place or exec_place).
 *        Uses a vector for common device places and a map for other place types.
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

#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

#include <optional>
#include <unordered_map>
#include <vector>

namespace cuda::experimental::stf
{

// Traits to unify data_place and exec_place for the container.
namespace place_indexed_detail
{
template <typename Place>
struct traits;

template <>
struct traits<data_place>
{
  static bool is_device(const data_place& p)
  {
    return p.is_device();
  }
  static int device_ordinal(const data_place& p)
  {
    return device_ordinal(p);
  }
};

template <>
struct traits<exec_place>
{
  static bool is_device(const exec_place& p)
  {
    return p.is_device();
  }
  static int device_ordinal(const exec_place& p)
  {
    return device_ordinal(p.affine_data_place());
  }
};
} // namespace place_indexed_detail

/**
 * @brief Container for resources indexed by place (data_place or exec_place).
 *
 * Device places (e.g. data_place::device(i), exec_place::device(i)) are stored in a vector
 * indexed by device ordinal for fast access. All other place types (host, managed, extension,
 * grid, green context, etc.) are stored in an unordered_map.
 *
 * @tparam Place Either data_place or exec_place.
 * @tparam T     The resource type (e.g. stream_pool, or a struct holding per-place state).
 */
template <typename Place, typename T>
class place_indexed_resource
{
  using traits_t = place_indexed_detail::traits<Place>;

  ::std::vector<::std::optional<T>> device_resources_;
  ::std::unordered_map<Place, T, hash<Place>> other_resources_;
  mutable int device_count_ = -1;

  int device_count() const
  {
    if (device_count_ < 0)
    {
      device_count_ = cuda_try<cudaGetDeviceCount>();
    }
    return device_count_;
  }

public:
  place_indexed_resource() = default;

  /**
   * @brief Find resource for a place. Does not insert.
   * @return Pointer to the resource if present, nullptr otherwise.
   */
  T* find(Place p)
  {
    if (traits_t::is_device(p))
    {
      int ord = traits_t::device_ordinal(p);
      if (ord < 0 || ord >= int(device_resources_.size()))
      {
        return nullptr;
      }
      ::std::optional<T>& slot = device_resources_.at(static_cast<size_t>(ord));
      return slot.has_value() ? &slot.value() : nullptr;
    }

    auto it = other_resources_.find(p);
    return it != other_resources_.end() ? &it->second : nullptr;
  }

  /** @brief Const overload of find(). */
  const T* find(Place p) const
  {
    return const_cast<place_indexed_resource*>(this)->find(p);
  }

  /**
   * @brief Get or create resource for a place.
   * @param p      The place key.
   * @param factory Called with the place to construct T when the slot is empty. For device places,
   *                the factory may use device_ordinal(p) or device_ordinal(p.affine_data_place()).
   * @return Reference to the resource for this place.
   */
  template <typename Factory>
  T& get_or_create(Place p, Factory&& factory)
  {
    if (traits_t::is_device(p))
    {
      int ord = traits_t::device_ordinal(p);
      _CCCL_ASSERT(ord >= 0 && ord < device_count(),
                   "Device ordinal out of range: ",
                   ord,
                   " (device count ",
                   device_count(),
                   ")");
      size_t idx = static_cast<size_t>(ord);
      if (device_resources_.size() <= idx)
      {
        device_resources_.resize(static_cast<size_t>(device_count()));
      }
      ::std::optional<T>& slot = device_resources_.at(idx);
      if (!slot.has_value())
      {
        slot.emplace(factory(p));
      }
      return slot.value();
    }

    auto it = other_resources_.find(p);
    if (it != other_resources_.end())
    {
      return it->second;
    }
    return other_resources_.emplace(p, factory(p)).first->second;
  }

  /** @brief Number of device slots (capacity for device ordinals 0..device_count()-1). */
  size_t device_capacity() const
  {
    return device_resources_.size();
  }

  /** @brief Number of non-device places stored in the map. */
  size_t other_size() const
  {
    return other_resources_.size();
  }
};

} // namespace cuda::experimental::stf
