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
 * @brief CUDA stream execution place implementation
 */

#pragma once

#include <cuda/__cccl_config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__host_stdlib/memory>

#include <cuda/experimental/__places/exec_place_resources.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>

#include <functional>
#include <string>

namespace cuda::experimental::places
{
/**
 * @brief Implementation for CUDA stream execution places
 */
class exec_place_cuda_stream_impl : public exec_place::impl
{
public:
  _CCCL_HOST_API exec_place_cuda_stream_impl(const augmented_stream& dstream)
      : exec_place::impl(data_place::device(dstream.dev_id))
      , dstream_(dstream)
      , dummy_pool_(dstream)
  {}

  _CCCL_HOST_API ::std::shared_ptr<exec_place::impl> get_place([[maybe_unused]] size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    return shared_from_this();
  }

  _CCCL_HOST_API exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    return exec_place::device(dstream_.dev_id).get_impl()->activate(idx);
  }

  // Preserve the optional index for callers using the concrete stream place.
  // NOLINTNEXTLINE(google-default-arguments)
  _CCCL_HOST_API void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    exec_place::device(dstream_.dev_id).get_impl()->deactivate(prev, idx);
  }

  _CCCL_HOST_API bool is_device() const override
  {
    return true;
  }

  _CCCL_HOST_API stream_pool& get_stream_pool(bool, exec_place_resources&, const exec_place&) const override
  {
    // User-stream places carry their own single-stream pool and intentionally
    // ignore the registry.
    return dummy_pool_;
  }

  _CCCL_HOST_API ::std::string to_string() const override
  {
    return "cuda_stream(id=" + ::std::to_string(dstream_.id) + " dev=" + ::std::to_string(dstream_.dev_id) + ")";
  }

  _CCCL_HOST_API int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast) -- typeid checked just above
    const auto& other = static_cast<const exec_place_cuda_stream_impl&>(rhs);
    return (other.dstream_.stream < dstream_.stream) - (dstream_.stream < other.dstream_.stream);
  }

  _CCCL_HOST_API size_t hash() const override
  {
    return ::std::hash<cudaStream_t>()(dstream_.stream);
  }

private:
  augmented_stream dstream_;
  mutable stream_pool dummy_pool_;
};

_CCCL_HOST_API inline exec_place exec_place::cuda_stream(cudaStream_t stream)
{
  const int devid = get_device_from_stream(stream);
  return exec_place{
    ::std::make_shared<exec_place_cuda_stream_impl>(augmented_stream(stream, get_stream_id(stream), devid))};
}

_CCCL_HOST_API inline exec_place exec_place::cuda_stream(const augmented_stream& dstream)
{
  return exec_place{::std::make_shared<exec_place_cuda_stream_impl>(dstream)};
}
} // end namespace cuda::experimental::places
