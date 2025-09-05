//===----------------------------------------------------------------------===//
//
// multi_device_async_resource
//
// A memory resource that manages one device_memory_resource per CUDA device.
// Dispatches allocations and deallocations to the appropriate per-device
// resource, based on either the current device or the device associated with
// a given stream.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#if _CCCL_CUDA_COMPILER(CLANG)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include <cuda/experimental/__memory_resource/device_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/memory_resource_base.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>
#include <cuda/experimental/__memory_resource/shared_resource.cuh>
#include <cuda/experimental/__stream/stream.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cuda/std/mutex>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/vector>

namespace cuda::experimental {

// Multi-device async memory resource.
// - Holds one device_memory_resource per device id (wrapped in
// shared_resource).
// - Dispatches allocate/deallocate to the appropriate per-device resource.
// Usage notes:
// - Prefer the explicit device overloads when you know the target device.
//   in the codebase, replace __stream_device() with that helper for correct
//   behavior when the stream belongs to another device.
class multi_device_async_resource {
private:
  using device_res_t = shared_resource<device_memory_resource>;
  ::cuda::std::vector<device_res_t> __per_device_; // indexed by device id
  mutable ::cuda::std::mutex __guard_;

  static int __current_device() noexcept {
    int d = 0;
    if (::cudaGetDevice(&d) != cudaSuccess) {
      // fallback to 0 if error (rare, but keeps API consistent)
      return 0;
    }
    return d;
  }

  // For now we fallback to current device.
  static int __stream_device([[maybe_unused]] ::cuda::stream_t s) noexcept {
#if defined(CUDA_HAS_STREAM_GET_DEVICE) // placeholder if you detect API
    int dev = 0;
    if (::cudaStreamGetDevice(s, &dev) == cudaSuccess)
      return dev;
#endif
    return __current_device();
  }

  void __validate_device_index(int device_id) const {
    if (device_id < 0 ||
        static_cast<size_t>(device_id) >= __per_device_.size()) {
      ::cuda::std::__throw_out_of_range(
          "device id out of range for multi_device_async_resource");
    }
  }

public:
  // default ctor: create per-device resources for all visible devices
  multi_device_async_resource() {
    int count = 0;
    _CCCL_TRY_CUDA_API(::cudaGetDeviceCount, "cudaGetDeviceCount failed",
                       &count);
    __per_device_.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      __per_device_.emplace_back(
          ::cuda::std::in_place_type<device_memory_resource>,
          ::cuda::device_ref{i});
    }
  }

  // construct from a list of device_refs
  explicit multi_device_async_resource(
      ::cuda::std::span<const device_ref> devices) {
    __per_device_.reserve(devices.size());
    for (auto const &d : devices) {
      __per_device_.emplace_back(
          ::cuda::std::in_place_type<device_memory_resource>, d);
    }
  }

  // disallow copy to avoid accidental shared mutable state
  multi_device_async_resource(const multi_device_async_resource &) = delete;
  multi_device_async_resource &
  operator=(const multi_device_async_resource &) = delete;

  multi_device_async_resource(multi_device_async_resource &&) = default;
  multi_device_async_resource &
  operator=(multi_device_async_resource &&) = default;

  ~multi_device_async_resource() = default;

  // get number of devices managed
  [[nodiscard]] size_t device_count() const noexcept {
    return __per_device_.size();
  }

  // access raw device resource (const + mutable)
  [[nodiscard]] device_res_t &get_resource(::cuda::device_ref devref) {
    int dev = devref.get();
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)];
  }
  [[nodiscard]] const device_res_t &
  get_resource(::cuda::device_ref devref) const {
    int dev = devref.get();
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)];
  }

  // -------- Allocators --------

  // allocate on the device implied by the given stream (fallback: current host
  // device)
  [[nodiscard]] void *
  allocate(const ::cuda::stream_ref stream, size_t bytes,
           size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = __stream_device(stream.get());
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)].allocate(stream, bytes,
                                                            alignment);
  }

  // allocate on the specified device explicitly
  [[nodiscard]] void *
  allocate(::cuda::device_ref devref, const ::cuda::stream_ref stream,
           size_t bytes,
           size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = devref.get();
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)].allocate(stream, bytes,
                                                            alignment);
  }

  // synchronous allocate on current device
  [[nodiscard]] void *
  allocate_sync(size_t bytes,
                size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = __current_device();
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)].allocate_sync(bytes,
                                                                 alignment);
  }

  // synchronous allocate on specific device
  [[nodiscard]] void *
  allocate_sync(::cuda::device_ref devref, size_t bytes,
                size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = devref.get();
    __validate_device_index(dev);
    return __per_device_[static_cast<size_t>(dev)].allocate_sync(bytes,
                                                                 alignment);
  }

  // non-throwing try_allocate (returns nullptr on failure)
  [[nodiscard]] void *try_allocate(
      const ::cuda::stream_ref stream, size_t bytes,
      size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept {
    try {
      return allocate(stream, bytes, alignment);
    } catch (...) {
      return nullptr;
    }
  }

  // -------- Deallocators --------

  void
  deallocate(const ::cuda::stream_ref stream, void *ptr, size_t bytes,
             size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = __stream_device(stream.get());
    __validate_device_index(dev);
    __per_device_[static_cast<size_t>(dev)].deallocate(stream, ptr, bytes,
                                                       alignment);
  }

  void
  deallocate(::cuda::device_ref devref, const ::cuda::stream_ref stream,
             void *ptr, size_t bytes,
             size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) {
    int dev = devref.get();
    __validate_device_index(dev);
    __per_device_[static_cast<size_t>(dev)].deallocate(stream, ptr, bytes,
                                                       alignment);
  }

  void deallocate_sync(
      void *ptr, size_t bytes,
      size_t alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept {
    int dev = __current_device();
    __validate_device_index(dev);
    __per_device_[static_cast<size_t>(dev)].deallocate_sync(ptr, bytes,
                                                            alignment);
  }

  // -------- Access control --------

  void enable_access_from(::cuda::std::span<const device_ref> devices) {
    for (auto &r : __per_device_)
      r.enable_access_from(devices);
  }

  void disable_access_from(::cuda::std::span<const device_ref> devices) {
    for (auto &r : __per_device_)
      r.disable_access_from(devices);
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_RESOURCE_CUH
