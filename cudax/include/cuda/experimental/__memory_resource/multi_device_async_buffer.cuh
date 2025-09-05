//===----------------------------------------------------------------------===//
//
// libcudacxx/include/cuda/experimental/__memory_resource/multi_device_async_buffer.cuh
//
// A buffer type that allocates via multi_device_async_resource and can migrate
// its storage between devices asynchronously when switching streams.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_BUFFER_CUH
#define _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_BUFFER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

#include <cuda/experimental/__memory_resource/multi_device_async_resource.cuh>
#include <cuda/experimental/__stream/internal_streams.cuh>
#include <cuda/experimental/__stream/stream.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cuda/std/memory>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <cuda/std/vector>

namespace cuda::experimental::mr {

// internal RAII helper: pinned host staging buffer
struct __pinned_buffer {
  void *ptr = nullptr;
  size_t size = 0;

  __pinned_buffer() = default;
  explicit __pinned_buffer(size_t bytes) { allocate(bytes); }

  void allocate(size_t bytes) {
    if (ptr)
      free();
    size = bytes;
    if (bytes == 0)
      return;
    _CCCL_TRY_CUDA_API(::cudaHostAlloc, "cudaHostAlloc failed", &ptr, bytes,
                       cudaHostAllocDefault);
  }

  void free() {
    if (ptr) {
      _CCCL_TRY_CUDA_API(::cudaFreeHost, "cudaFreeHost failed", ptr);
      ptr = nullptr;
      size = 0;
    }
  }

  ~__pinned_buffer() { free(); }
};

// A movable buffer that allocates through multi_device_async_resource.
// - `allocate_bytes()` allocates on a given stream's device
// - `switch_stream()` migrates the storage asynchronously if target differs

class multi_device_async_buffer {
private:
  multi_device_async_resource *__res_ = nullptr; // non-owning
  void *__ptr_ = nullptr;
  size_t __bytes_ = 0;
  int __device_id_ = -1;
  ::cuda::stream_ref __stream_;

  struct __retire_entry {
    void *ptr;
    size_t bytes;
    int device;
    cudaEvent_t ev;
  };
  ::cuda::std::vector<__retire_entry> __retire_list_;

  void __collect_retirements() {
    for (auto it = __retire_list_.begin(); it != __retire_list_.end();) {
      cudaError_t e = ::cudaEventQuery(it->ev);
      if (e == cudaSuccess || e != cudaErrorNotReady) {
        ::cuda::ScopedDeviceScope dev_scope(it->device);
        __res_->deallocate_sync(it->ptr, it->bytes);
        ::cudaEventDestroy(it->ev);
        it = __retire_list_.erase(it);
      } else {
        ++it;
      }
    }
  }

public:
  explicit multi_device_async_buffer(
      multi_device_async_resource *res = nullptr) noexcept
      : __res_(res) {}

  ~multi_device_async_buffer() noexcept {
    __collect_retirements();
    if (__ptr_ && __device_id_ >= 0) {
      ::cuda::ScopedDeviceScope dev_scope(__device_id_);
      __res_->deallocate_sync(__ptr_, __bytes_);
    }
    for (auto &r : __retire_list_) {
      ::cudaEventDestroy(r.ev);
      ::cuda::ScopedDeviceScope dev_scope(r.device);
      __res_->deallocate_sync(r.ptr, r.bytes);
    }
  }

  void allocate_bytes(size_t bytes, ::cuda::stream_ref s) {
    if (!__res_)
      ::cuda::std::__throw_logic_error("Resource pointer null");
    __collect_retirements();

    // TODO: replace with stream->device mapping
    int dev = 0;
    (void)::cudaGetDevice(
        &dev); // ------------- TODO: replace with stream->device mapping
    __ptr_ = __res_->allocate(s, bytes);
    __bytes_ = bytes;
    __device_id_ = dev;
    __stream_ = s;
  }

  void *data() noexcept { return __ptr_; }
  size_t size() const noexcept { return __bytes_; }
  int device_id() const noexcept { return __device_id_; }

  void switch_stream(::cuda::stream_ref new_stream);
};

} // namespace cuda::experimental::mr

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDAX__MEMORY_RESOURCE_MULTI_DEVICE_ASYNC_BUFFER_CUH
