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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/places.cuh>

namespace cuda::experimental::places
{
/**
 * @brief Implementation for CUDA stream execution places
 */
class exec_place_cuda_stream_impl : public exec_place::impl
{
public:
  exec_place_cuda_stream_impl(const decorated_stream& dstream)
      : exec_place::impl(data_place::device(dstream.dev_id))
      , dstream_(dstream)
      , dummy_pool_(dstream)
  {}

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    return shared_from_this();
  }

  exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    return exec_place::device(dstream_.dev_id).get_impl()->activate(0);
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_stream exec_place");
    exec_place::device(dstream_.dev_id).get_impl()->deactivate(prev, 0);
  }

  bool is_device() const override
  {
    return true;
  }

  stream_pool& get_stream_pool(bool, exec_place_resources&, const exec_place&) const override
  {
    // User-stream places carry their own single-stream pool and intentionally
    // ignore the registry.
    return dummy_pool_;
  }

  ::std::string to_string() const override
  {
    return "cuda_stream(id=" + ::std::to_string(dstream_.id) + " dev=" + ::std::to_string(dstream_.dev_id) + ")";
  }

  int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    const auto& other = static_cast<const exec_place_cuda_stream_impl&>(rhs);
    return (other.dstream_.stream < dstream_.stream) - (dstream_.stream < other.dstream_.stream);
  }

  size_t hash() const override
  {
    return ::std::hash<cudaStream_t>()(dstream_.stream);
  }

private:
  decorated_stream dstream_;
  mutable stream_pool dummy_pool_;
};

inline exec_place exec_place::cuda_stream(cudaStream_t stream)
{
  int devid = get_device_from_stream(stream);
  return exec_place{
    ::std::make_shared<exec_place_cuda_stream_impl>(decorated_stream(stream, get_stream_id(stream), devid))};
}

inline exec_place exec_place::cuda_stream(const decorated_stream& dstream)
{
  return exec_place{::std::make_shared<exec_place_cuda_stream_impl>(dstream)};
}
} // end namespace cuda::experimental::places
