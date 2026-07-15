//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Execution place wrapping an externally-owned CUDA driver context
 *
 * This makes it possible to use a CUcontext created outside CUDASTF (e.g. a
 * green context created through cuda.core in Python, or any other library)
 * as an execution place. The place is non-owning: the caller must keep the
 * context alive while the place (and the streams lazily created in its pool)
 * is in use.
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

#include <cuda/__runtime/ensure_current_context.h>

#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

#include <functional>

namespace cuda::experimental::places
{
/**
 * @brief Implementation for execution places backed by an externally-owned CUcontext
 *
 * The context is used as-is: `activate()` saves the current context and makes
 * this one current, `deactivate()` restores the saved context. Streams are
 * created lazily in the place's own pool while the context is current, so they
 * inherit whatever resources the context carries (e.g. the SM partition of a
 * green context converted with `cuCtxFromGreenCtx`).
 *
 * Identity (`hash`/`cmp`) is keyed on the CUcontext handle, which uniquely
 * identifies the underlying resources: `cuCtxFromGreenCtx` returns the same
 * CUcontext for a given CUgreenCtx on every call.
 */
class exec_place_cuda_ctx_impl : public exec_place::impl
{
public:
  /**
   * @brief Construct an execution place from an externally-owned CUDA context
   *
   * @param ctx The CUDA driver context. Non-owning: the caller keeps it alive.
   * @param devid The device ordinal the context belongs to, or -1 to derive it
   *        from the context (via cuCtxGetDevice).
   * @param pool_size Number of streams in the place's stream pool.
   */
  exec_place_cuda_ctx_impl(CUcontext ctx, int devid = -1, size_t pool_size = exec_place::impl::pool_size)
      : exec_place_cuda_ctx_impl(ctx, resolve_devid(ctx, devid), stream_pool(pool_size))
  {}

  /**
   * @brief Full-control constructor, also used by the green-context place
   *
   * @param ctx The CUDA driver context (already resolved, e.g. from cuCtxFromGreenCtx)
   * @param devid The device ordinal (must be valid)
   * @param pool A stream pool to use for this place (shared handle)
   * @param affine The affine data place for this place
   */
  exec_place_cuda_ctx_impl(CUcontext ctx, int devid, stream_pool pool, data_place affine)
      : exec_place::impl(mv(affine))
      , devid_(devid)
      , driver_context_(ctx)
      , pool_(mv(pool))
  {
    _CCCL_ASSERT(ctx != nullptr, "cuda_ctx exec_place requires a valid CUcontext");
    _CCCL_ASSERT(devid_ >= 0, "cuda_ctx exec_place requires a valid device ordinal");
  }

  ::std::shared_ptr<exec_place::impl> get_place(size_t idx) override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_ctx exec_place");
    return shared_from_this();
  }

  exec_place activate(size_t idx) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_ctx exec_place");

    // Save the current context wrapped as a place so deactivate can restore it
    CUcontext current_ctx = cuda_try<cuCtxGetCurrent>();
    exec_place result     = exec_place(::std::make_shared<exec_place_cuda_ctx_impl>(saved_tag{}, current_ctx));

    cuda_try<cuCtxSetCurrent>(driver_context_);

    return result;
  }

  void deactivate(const exec_place& prev, size_t idx = 0) const override
  {
    _CCCL_ASSERT(idx == 0, "Index out of bounds for cuda_ctx exec_place");

    auto prev_impl      = ::std::static_pointer_cast<exec_place_cuda_ctx_impl>(prev.get_impl());
    CUcontext saved_ctx = prev_impl->driver_context_;

    cuda_try<cuCtxSetCurrent>(saved_ctx);
  }

  bool is_device() const override
  {
    return true;
  }

  ::std::string to_string() const override
  {
    return "cuda_ctx(ctx=" + ::std::to_string(reinterpret_cast<::std::uintptr_t>(driver_context_))
         + " dev=" + ::std::to_string(devid_) + ")";
  }

  stream_pool& get_stream_pool(bool, exec_place_resources&, const exec_place&) const override
  {
    // This place carries its own pool and bypasses the registry. The user is
    // responsible for keeping the underlying CUcontext alive while the pool
    // is in use.
    return pool_;
  }

  int cmp(const exec_place::impl& rhs) const override
  {
    if (typeid(*this) != typeid(rhs))
    {
      return typeid(*this).before(typeid(rhs)) ? -1 : 1;
    }
    const auto& other = static_cast<const exec_place_cuda_ctx_impl&>(rhs);
    return ::std::less<CUcontext>{}(other.driver_context_, driver_context_)
         - ::std::less<CUcontext>{}(driver_context_, other.driver_context_);
  }

  size_t hash() const override
  {
    return ::std::hash<CUcontext>()(driver_context_);
  }

protected:
  // Tag type for the internal "saved context" wrapper used by activate/deactivate.
  // The type itself is protected so only this class (and derived classes) can
  // name it; the constructor below must be public for make_shared.
  struct saved_tag
  {};

public:
  // Wrap an existing context with no pool: only used to carry the saved
  // context through the activate/deactivate round trip.
  exec_place_cuda_ctx_impl(saved_tag, CUcontext saved_context)
      : driver_context_(saved_context)
  {}

protected:
  exec_place_cuda_ctx_impl(CUcontext ctx, int devid, stream_pool pool)
      : exec_place_cuda_ctx_impl(ctx, devid, mv(pool), data_place::device(devid))
  {}

  static int resolve_devid(CUcontext ctx, int devid)
  {
    ::cuda::__ensure_current_context guard{ctx};
    const int context_devid = static_cast<int>(cuda_try<cuCtxGetDevice>());
    if (devid >= 0 && devid != context_devid)
    {
      throw ::std::invalid_argument("CUcontext device ordinal does not match devid");
    }
    return context_devid;
  }

  int devid_                = -1;
  CUcontext driver_context_ = {};
  mutable stream_pool pool_;
};

inline exec_place exec_place::cuda_context(CUcontext ctx, int devid, size_t pool_size)
{
  if (ctx == nullptr)
  {
    // Reject eagerly: with an explicit devid a null context would construct a
    // place that only fails (or silently unbinds the current context) at
    // activation time.
    throw ::std::invalid_argument("exec_place::cuda_context requires a valid CUcontext");
  }
  return exec_place(::std::make_shared<exec_place_cuda_ctx_impl>(ctx, devid, pool_size));
}

#ifdef UNITTESTED_FILE
namespace
{
//! RAII holder for the primary context of device 0, used by the unittests below.
struct primary_ctx_guard
{
  primary_ctx_guard()
  {
    cuda_try<cuInit>(0);
    dev = cuda_try<cuDeviceGet>(0);
    ctx = cuda_try<cuDevicePrimaryCtxRetain>(dev);
  }

  ~primary_ctx_guard()
  {
    cuda_try(cuDevicePrimaryCtxRelease(dev));
  }

  CUdevice dev  = -1;
  CUcontext ctx = nullptr;
};
} // namespace

UNITTEST("cuda_context exec_place equality")
{
  primary_ctx_guard guard;

  auto p0a = exec_place::cuda_context(guard.ctx, 0);
  auto p0b = exec_place::cuda_context(guard.ctx, 0);

  // Same context should be equal
  EXPECT(p0a == p0b);
  EXPECT(!(p0a != p0b));

  // A cuda_context place is not a regular device place
  auto dev0 = exec_place::device(0);
  EXPECT(p0a != dev0);
  EXPECT(!(p0a == dev0));
};

UNITTEST("cuda_context exec_place derives the device ordinal")
{
  primary_ctx_guard guard;

  // devid intentionally omitted: derived from the context via cuCtxGetDevice
  auto p = exec_place::cuda_context(guard.ctx);
  EXPECT(p.is_device());
  EXPECT(p.affine_data_place() == data_place::device(0));
  EXPECT(p == exec_place::cuda_context(guard.ctx, 0));
};

UNITTEST("cuda_context exec_place rejects a null context")
{
  bool thrown = false;
  try
  {
    auto p = exec_place::cuda_context(nullptr, 0);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);
};

UNITTEST("cuda_context exec_place rejects a mismatched device ordinal")
{
  primary_ctx_guard guard;

  bool thrown = false;
  try
  {
    auto p = exec_place::cuda_context(guard.ctx, 1);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);
};

UNITTEST("cuda_context exec_place activate/deactivate round trip")
{
  primary_ctx_guard guard;

  auto p = exec_place::cuda_context(guard.ctx, 0);

  CUcontext before = cuda_try<cuCtxGetCurrent>();
  {
    exec_place_scope scope(p);
    EXPECT(cuda_try<cuCtxGetCurrent>() == guard.ctx);
  }
  EXPECT(cuda_try<cuCtxGetCurrent>() == before);
};
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::places
