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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/utility/stream_to_dev.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Designates execution that is to run on a specific CUDA stream
 *
 */
class exec_place_cuda_stream : public exec_place
{
public:
  class impl : public exec_place::impl
  {
  public:
    impl(const decorated_stream& _dstream)
        : exec_place::impl(data_place::device(_dstream.dev_id))
        , dstream(_dstream)
    {
      // Create a dummy pool
      dummy_pool.payload.push_back(dstream);
    }

    /* We set the current device to be the device on which the CUDA stream was created */
    exec_place activate() const override
    {
      return exec_place::device(dstream.dev_id).activate();
    }

    void deactivate(const exec_place& prev) const override
    {
      return exec_place::device(dstream.dev_id).deactivate(prev);
    }

    stream_pool& get_stream_pool(async_resources_handle&, bool) const override
    {
      return dummy_pool;
    }

    ::std::string to_string() const override
    {
      return "exec(stream id=" + ::std::to_string(dstream.id) + " dev=" + ::std::to_string(dstream.dev_id) + ")";
    }

    bool operator==(const exec_place::impl& rhs) const override
    {
      if (typeid(*this) != typeid(rhs))
      {
        return false;
      }
      const auto& other = static_cast<const impl&>(rhs);
      // Compare by stream handle
      return dstream.stream == other.dstream.stream;
    }

    size_t hash() const override
    {
      // Hash the stream handle, not the affine data place
      return ::std::hash<cudaStream_t>()(dstream.stream);
    }

    bool operator<(const exec_place::impl& rhs) const override
    {
      if (typeid(*this) != typeid(rhs))
      {
        return typeid(*this).before(typeid(rhs));
      }
      const auto& other = static_cast<const impl&>(rhs);
      return dstream.stream < other.dstream.stream;
    }

  private:
    decorated_stream dstream;
    // We create a dummy pool of streams which only consists in a single stream in practice.
    mutable stream_pool dummy_pool;
  };

public:
  exec_place_cuda_stream(const decorated_stream& dstream)
      : exec_place(::std::make_shared<impl>(dstream))
  {
    static_assert(sizeof(exec_place_cuda_stream) == sizeof(exec_place),
                  "exec_place_cuda_stream cannot add state; it would be sliced away.");
  }
};

inline exec_place_cuda_stream exec_place::cuda_stream(cudaStream_t stream)
{
  int devid = get_device_from_stream(stream);
  return exec_place_cuda_stream(decorated_stream(stream, -1, devid));
}

inline exec_place_cuda_stream exec_place::cuda_stream(const decorated_stream& dstream)
{
  return exec_place_cuda_stream(dstream);
}
} // end namespace cuda::experimental::stf
