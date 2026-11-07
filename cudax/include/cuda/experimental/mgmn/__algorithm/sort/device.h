// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_DEVICE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_DEVICE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

// Generic communicators remain usable without an installed NCCL SDK.
#if defined(__CUDACC__) && __has_include(<nccl_device.h>)
#  include <nccl_device.h>
#  if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2)
#    define _CUDAX_SORT_HAS_NCCL_DEVICE 1

#    include <cuda/std/__exception/exception_macros.h>
#    include <cuda/std/__host_stdlib/stdexcept>
#    include <cuda/std/__memory/assume_aligned.h>
#    include <cuda/std/cstdint>
#    include <cuda/std/cstring>
#    include <cuda/std/limits>

#    include <cuda/experimental/mgmn/__algorithm/sort/select.h>

#    include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

// GIN uses up to 64 chunks, bounded by input bytes and resident capacity.
inline constexpr int __transfer_blocks = 64;
// LSA copies run on the SMs and need more CTAs to hide remote-memory latency.
// This is the upper bound on allocated barrier slots. The launch grid is
// bounded by the compiled kernel's residency on every participating GPU.
inline constexpr int __lsa_transfer_blocks = 256;

_CCCL_HOST_API inline void __check_nccl_device(::ncclResult_t __status)
{
  if (__status != ::ncclSuccess)
  {
    _CCCL_THROW(::std::runtime_error, ::ncclGetErrorString(__status));
  }
}

// Each rank reserves the same window layout, even when caller sizes differ.
// The caller's allocation is never registered, resized, or used for incoming puts.
struct __device_layout
{
  ::cuda::std::size_t __bytes_ = 0;
  ::cuda::std::size_t __source_, __gathered_, __local_cuts_, __cuts_, __targets_, __sorted_, __received_;

  template <class _Tp>
  _CCCL_HOST_API auto __append(::cuda::std::size_t __count) -> ::cuda::std::size_t
  {
    constexpr auto __limit = ::cuda::std::numeric_limits<::cuda::std::size_t>::max();
    const auto __alignment = alignof(_Tp) > 128 ? alignof(_Tp) : 128;
    if (__bytes_ > __limit - (__alignment - 1))
    {
      _CCCL_THROW(::std::overflow_error, "Distributed sort window size overflow");
    }
    const auto __offset = (__bytes_ + __alignment - 1) / __alignment * __alignment;
    if (__count > (__limit - __offset) / sizeof(_Tp))
    {
      _CCCL_THROW(::std::overflow_error, "Distributed sort window size overflow");
    }
    __bytes_ = __offset + __count * sizeof(_Tp);
    return __offset;
  }
};

class __device_window
{
  ::ncclComm_t __comm_;
  ::cudaStream_t __stream_;
  int __ordinal_  = 0;
  bool __created_ = false;
  void* __host_metadata_[2]{};
  ::cuda::std::size_t __host_capacity_[2]{};

  _CCCL_HOST_API void __release() noexcept
  {
    int __previous = 0;
    (void) ::cudaGetDevice(&__previous);
    (void) ::cudaSetDevice(__ordinal_);
    // All kernels and the copy back to the caller must finish before unmapping.
    (void) ::cudaStreamSynchronize(__stream_);
    if (__created_)
    {
      (void) ::ncclDevCommDestroy(__comm_, &__device_);
    }
    if (__window_)
    {
      (void) ::ncclCommWindowDeregister(__comm_, __window_);
    }
    if (__storage_)
    {
      (void) ::ncclMemFree(__storage_);
    }
    for (auto* __buffer : __host_metadata_)
    {
      if (__buffer)
      {
        (void) ::cudaFreeHost(__buffer);
      }
    }
    (void) ::cudaSetDevice(__previous);
  }

public:
  char* __storage_         = nullptr;
  ::ncclWindow_t __window_ = nullptr;
  ::ncclDevComm __device_{};
  ::ncclGinBarrierHandle __world_barrier_{};
  bool __gin_;
  ::cuda::std::size_t __capacity_ = 0;

  [[nodiscard]] _CCCL_HOST_API bool __matches(::ncclComm_t __comm, int __ordinal, bool __gin) const noexcept
  {
    return __comm_ == __comm && __ordinal_ == __ordinal && __gin_ == __gin;
  }

  _CCCL_HOST_API void __set_stream(::cudaStream_t __stream) noexcept
  {
    __stream_ = __stream;
  }

  // Pinned count and cut buffers are reused only while the context lock is held.
  // Each transfer into them is synchronized before their next use or destruction.
  [[nodiscard]] _CCCL_HOST_API void* __host_buffer(int __slot, ::cuda::std::size_t __bytes)
  {
    if (__bytes > __host_capacity_[__slot])
    {
      void* __next       = nullptr;
      const auto __error = ::cudaMallocHost(&__next, __bytes);
      if (__error != ::cudaSuccess)
      {
        _CCCL_THROW(::std::runtime_error, ::cudaGetErrorString(__error));
      }
      if (__host_metadata_[__slot])
      {
        (void) ::cudaFreeHost(__host_metadata_[__slot]);
      }
      __host_metadata_[__slot] = __next;
      __host_capacity_[__slot] = __bytes;
    }
    return __host_metadata_[__slot];
  }

  // All ranks enter together, after prior uses of the old window have completed.
  // The device communicator and its barrier epochs survive workspace growth.
  _CCCL_HOST_API void __reserve(::cuda::std::size_t __bytes)
  {
    char* __storage         = nullptr;
    ::ncclWindow_t __window = nullptr;
    ::cuda::experimental::mgmn::__detail::__sort::__check_nccl_device(
      ::ncclMemAlloc(reinterpret_cast<void**>(&__storage), __bytes));
    _CCCL_TRY
    {
      ::cuda::experimental::mgmn::__detail::__sort::__check_nccl_device(
        ::ncclCommWindowRegister(__comm_, __storage, __bytes, &__window, NCCL_WIN_COLL_SYMMETRIC));
    }
    _CCCL_CATCH_ALL
    {
      (void) ::ncclMemFree(__storage);
      _CCCL_RETHROW;
    }
    if (__window_)
    {
      (void) ::ncclCommWindowDeregister(__comm_, __window_);
      (void) ::ncclMemFree(__storage_);
    }
    __storage_  = __storage;
    __window_   = __window;
    __capacity_ = __bytes;
  }

  _CCCL_HOST_API __device_window(::ncclComm_t __comm, ::cudaStream_t __stream, ::cuda::std::size_t __bytes, bool __gin)
      : __comm_{__comm}
      , __stream_{__stream}
      , __gin_{__gin}
  {
    _CCCL_TRY
    {
      const auto __error = ::cudaGetDevice(&__ordinal_);
      if (__error != ::cudaSuccess)
      {
        _CCCL_THROW(::std::runtime_error, ::cudaGetErrorString(__error));
      }
      __reserve(__bytes);
      ::ncclDevCommRequirements __requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      ::ncclDevResourceRequirements __barrier_requirements{};
      if (__gin_)
      {
        // Use an explicitly world-scoped GIN barrier. A rail barrier alone is
        // insufficient when more than one GPU belongs to each LSA team.
        ::cuda::experimental::mgmn::__detail::__sort::__check_nccl_device(::ncclGinBarrierCreateRequirement(
          __comm_, ::ncclTeamWorld(__comm_), __transfer_blocks, &__world_barrier_, &__barrier_requirements));
        // NCCL 2.29.2 drops custom resource signal fields when copying the
        // requirements. Reserve the barrier signals explicitly after the data
        // signals, while retaining the barrier's resource buffer requirement.
        __world_barrier_.signal0                 = __transfer_blocks;
        __barrier_requirements.ginSignalCount    = 0;
        __barrier_requirements.outGinSignalStart = nullptr;

        __requirements.resourceRequirementsList = &__barrier_requirements;
        __requirements.ginForceEnable           = true;
        __requirements.ginSignalCount           = 2 * __transfer_blocks;
      }
      else
      {
        __requirements.lsaBarrierCount = __lsa_transfer_blocks;
      }
      ::cuda::experimental::mgmn::__detail::__sort::__check_nccl_device(
        ::ncclDevCommCreate(__comm_, &__requirements, &__device_));
      __created_ = true;
      if (__gin_ && __device_.ginContextCount == 0)
      {
        _CCCL_THROW(::std::runtime_error, "NCCL GPU-initiated networking is unavailable");
      }
    }
    _CCCL_CATCH_ALL
    {
      __release();
      _CCCL_RETHROW;
    }
  }
  __device_window(const __device_window&)            = delete;
  __device_window& operator=(const __device_window&) = delete;
  _CCCL_HOST_API ~__device_window()
  {
    __release();
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API auto __at(::cuda::std::size_t __offset) const noexcept -> _Tp*
  {
    return reinterpret_cast<_Tp*>(__storage_ + __offset);
  }
};

// Signal slots can retain values after a previous device communicator is destroyed.
template <class _DeviceComm>
_CCCL_KERNEL_ATTRIBUTES void __reset_gin_signals(_DeviceComm __comm)
{
  ::ncclGin __gin{__comm, static_cast<int>(blockIdx.x)};
  for (unsigned int __signal = threadIdx.x; __signal < __comm.ginSignalCount; __signal += blockDim.x)
  {
    __gin.resetSignal(__signal);
  }
}

template <bool _Gin>
class __device_session;

// Copy complete aligned words without imposing an aliasing type on the keys.
// Redistribution preserves the source alignment in the destination run. Other
// callers can have different alignments, in which case the byte path is used.
_CCCL_DEVICE_API inline void __copy_bytes(char* __dst, const char* __src, ::cuda::std::size_t __bytes)
{
  const auto __src_alignment = reinterpret_cast<::cuda::std::uintptr_t>(__src) % 16;
  const auto __dst_alignment = reinterpret_cast<::cuda::std::uintptr_t>(__dst) % 16;
  if (__src_alignment == __dst_alignment)
  {
    const auto __padding = (16 - __src_alignment) % 16;
    const auto __head    = __bytes < __padding ? __bytes : __padding;
    if (threadIdx.x < __head)
    {
      __dst[threadIdx.x] = __src[threadIdx.x];
    }
    __src += __head;
    __dst += __head;
    __bytes -= __head;
    const auto __words = __bytes / 16;
    for (::cuda::std::size_t __i = threadIdx.x; __i < __words; __i += blockDim.x)
    {
      ::cuda::std::memcpy(
        ::cuda::std::assume_aligned<16>(__dst + 16 * __i), ::cuda::std::assume_aligned<16>(__src + 16 * __i), 16);
    }
    __src += 16 * __words;
    __dst += 16 * __words;
    __bytes -= 16 * __words;
  }
  for (::cuda::std::size_t __i = threadIdx.x; __i < __bytes; __i += blockDim.x)
  {
    __dst[__i] = __src[__i];
  }
}

template <>
class __device_session<false>
{
  ::ncclLsaBarrierSession<::ncclCoopCta> __barrier_;

public:
  _CCCL_DEVICE_API __device_session(const ::ncclDevComm& __comm, ::ncclGinBarrierHandle)
      : __barrier_{::ncclCoopCta{}, __comm, ::ncclTeamTagLsa{}, blockIdx.x}
  {}
  _CCCL_DEVICE_API void __begin()
  {
    __barrier_.sync(::ncclCoopCta{}, ::cuda::memory_order_acq_rel);
  }
  _CCCL_DEVICE_API void
  __put(::ncclWindow_t __win,
        int __peer,
        ::cuda::std::size_t __destination,
        ::cuda::std::size_t __source,
        ::cuda::std::size_t __bytes)
  {
    const auto* __src = static_cast<const char*>(::ncclGetLocalPointer(__win, __source));
    auto* __dst       = static_cast<char*>(::ncclGetLsaPointer(__win, __destination, __peer));
    ::cuda::experimental::mgmn::__detail::__sort::__copy_bytes(__dst, __src, __bytes);
  }
  _CCCL_DEVICE_API void __end()
  {
    __barrier_.sync(::ncclCoopCta{}, ::cuda::memory_order_acq_rel);
  }
};

template <>
class __device_session<true>
{
  const ::ncclDevComm& __comm_;
  ::ncclGin __gin_;
  ::ncclGinBarrierSession<::ncclCoopCta> __barrier_;
  ::cuda::std::uint64_t __expected_ = 0;

public:
  _CCCL_DEVICE_API __device_session(const ::ncclDevComm& __comm, ::ncclGinBarrierHandle __barrier)
      : __comm_{__comm}
      , __gin_{__comm, static_cast<int>(blockIdx.x)}
      , __barrier_{::ncclCoopCta{}, __gin_, ::ncclTeamWorld(__comm), __barrier, blockIdx.x}
  {}
  _CCCL_DEVICE_API void __begin()
  {
    // Sample before the rendezvous, so a fast sender cannot race the snapshot.
    __expected_ = __gin_.readSignal(blockIdx.x) + __comm_.nRanks - 1;
    __barrier_.sync(::ncclCoopCta{}, ::cuda::memory_order_acq_rel, ::ncclGinFenceLevel::Relaxed);
  }
  _CCCL_DEVICE_API void
  __put(::ncclWindow_t __win,
        int __peer,
        ::cuda::std::size_t __destination,
        ::cuda::std::size_t __source,
        ::cuda::std::size_t __bytes)
  {
    if (__peer == __comm_.rank)
    {
      const auto* __src = static_cast<const char*>(::ncclGetLocalPointer(__win, __source));
      auto* __dst       = static_cast<char*>(::ncclGetLocalPointer(__win, __destination));
      ::cuda::experimental::mgmn::__detail::__sort::__copy_bytes(__dst, __src, __bytes);
    }
    else if (threadIdx.x == 0)
    {
      if (__bytes)
      {
        __gin_.put(
          ::ncclTeamWorld(__comm_),
          __peer,
          __win,
          __destination,
          __win,
          __source,
          __bytes,
          ::ncclGin_SignalInc{blockIdx.x});
      }
      else
      {
        // Empty runs still participate in completion; receivers never wait on
        // a signal that a zero-count sender silently omitted.
        __gin_.signal(::ncclTeamWorld(__comm_), __peer, ::ncclGin_SignalInc{blockIdx.x});
      }
    }
  }
  _CCCL_DEVICE_API void __end()
  {
    __gin_.waitSignal(::ncclCoopCta{}, blockIdx.x, __expected_);
    __gin_.flush(::ncclCoopCta{});
    __syncthreads();
  }
};

template <bool _Gin>
_CCCL_DEVICE_API void __device_all_gather(
  __device_session<_Gin>& __session,
  const ::ncclDevComm& __comm,
  ::ncclWindow_t __win,
  ::cuda::std::size_t __source,
  ::cuda::std::size_t __destination,
  ::cuda::std::size_t __bytes)
{
  __session.__begin();
  for (int __peer = 0; __peer < __comm.nRanks; ++__peer)
  {
    __session.__put(__win, __peer, __destination + __comm.rank * __bytes, __source, __bytes);
  }
  __session.__end();
}

template <bool _Gin, class _Tp, class _Compare>
_CCCL_KERNEL_ATTRIBUTES void __device_select(
  ::ncclDevComm __comm,
  ::ncclGinBarrierHandle __barrier,
  ::ncclWindow_t __win,
  __device_layout __layout,
  __index_type __size,
  __selection_state<_Tp>* __states,
  bool __uniform,
  _Compare __cmp)
{
  __device_session<_Gin> __session{__comm, __barrier};
  const auto __width = static_cast<::cuda::std::size_t>(__comm.nRanks) + 1;
  const auto* __keys = static_cast<const _Tp*>(::ncclGetLocalPointer(__win, __layout.__sorted_));
  auto* __source     = static_cast<__selection_message<_Tp>*>(::ncclGetLocalPointer(__win, __layout.__source_));
  const auto* __gathered =
    static_cast<const __selection_message<_Tp>*>(::ncclGetLocalPointer(__win, __layout.__gathered_));
  const auto* __targets = static_cast<const __index_type*>(::ncclGetLocalPointer(__win, __layout.__targets_));
  auto* __cuts          = static_cast<__index_type*>(::ncclGetLocalPointer(__win, __layout.__local_cuts_));
  __shared__ int __unfinished;
  using __phase = __selection_phase;
  ::cuda::experimental::mgmn::__detail::__sort::__select_step<__phase::__initialize>(
    __keys,
    __size,
    __comm.rank,
    __comm.nRanks,
    __source,
    __gathered,
    __states,
    &__unfinished,
    __cmp,
    __targets,
    __uniform,
    __cuts);
  __syncthreads();
  ::cuda::experimental::mgmn::__detail::__sort::__device_all_gather(
    __session, __comm, __win, __layout.__source_, __layout.__gathered_, __width * sizeof(__selection_message<_Tp>));
  ::cuda::experimental::mgmn::__detail::__sort::__select_step<__phase::__first_count>(
    __keys,
    __size,
    __comm.rank,
    __comm.nRanks,
    __source,
    __gathered,
    __states,
    &__unfinished,
    __cmp,
    __targets,
    __uniform,
    __cuts);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    __source[0].__equal_ = 1;
  }
  __syncthreads();
  while (true)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__device_all_gather(
      __session, __comm, __win, __layout.__source_, __layout.__gathered_, __width * sizeof(__selection_message<_Tp>));
    if (threadIdx.x == 0)
    {
      __unfinished = 0;
      for (int __peer = 0; __peer < __comm.nRanks; ++__peer)
      {
        __unfinished |= __gathered[__peer * __width].__equal_ != 0;
      }
    }
    __syncthreads();
    if (!__unfinished)
    {
      break;
    }
    ::cuda::experimental::mgmn::__detail::__sort::__select_step<__phase::__advance>(
      __keys,
      __size,
      __comm.rank,
      __comm.nRanks,
      __source,
      __gathered,
      __states,
      &__unfinished,
      __cmp,
      __targets,
      __uniform,
      __cuts);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      __source[0].__equal_ = __unfinished;
    }
    __syncthreads();
  }
  ::cuda::experimental::mgmn::__detail::__sort::__select_step<__phase::__export_cuts>(
    __keys,
    __size,
    __comm.rank,
    __comm.nRanks,
    __source,
    __gathered,
    __states,
    &__unfinished,
    __cmp,
    __targets,
    __uniform,
    __cuts);
  __syncthreads();
  ::cuda::experimental::mgmn::__detail::__sort::__device_all_gather(
    __session, __comm, __win, __layout.__local_cuts_, __layout.__cuts_, __width * sizeof(__index_type));
}

// Keep small transfers on the original kernel; large LSA transfers benefit
// from spreading each CTA's starting peer. Count arithmetic cannot overflow.
template <class _Tp>
_CCCL_HOST_API constexpr bool __use_rotated_redistribution(::cuda::std::uint64_t __count)
{
  constexpr ::cuda::std::uint64_t __min_bytes = ::cuda::std::uint64_t{1} << 30;
  return __count >= (__min_bytes / sizeof(_Tp) + (__min_bytes % sizeof(_Tp) != 0));
}

template <bool _Gin, class _Tp>
_CCCL_KERNEL_ATTRIBUTES void __device_redistribute(
  ::ncclDevComm __comm, ::ncclGinBarrierHandle __barrier, ::ncclWindow_t __win, __device_layout __layout)
{
  __device_session<_Gin> __session{__comm, __barrier};
  const auto __width = static_cast<::cuda::std::size_t>(__comm.nRanks) + 1;
  const auto* __cuts = static_cast<const __index_type*>(::ncclGetLocalPointer(__win, __layout.__cuts_));
  __session.__begin();
  for (int __peer = 0; __peer < __comm.nRanks; ++__peer)
  {
    const auto __begin    = __cuts[__comm.rank * __width + __peer];
    const auto __count    = __cuts[__comm.rank * __width + __peer + 1] - __begin;
    __index_type __cursor = 0, __offset = 0;
    for (int __source = 0; __source <= __comm.rank; ++__source)
    {
      const auto __first  = __cuts[__source * __width + __peer];
      const auto __length = __cuts[__source * __width + __peer + 1] - __first;
      __offset            = __cursor;
      if (__length)
      {
        __offset += __first % 16;
        __cursor = (__offset + __length + 15) / 16 * 16;
      }
    }
    const auto __first =
      __count / gridDim.x * blockIdx.x + (__count % gridDim.x < blockIdx.x ? __count % gridDim.x : blockIdx.x);
    const auto __length = __count / gridDim.x + (blockIdx.x < __count % gridDim.x);
    __session.__put(
      __win,
      __peer,
      __layout.__received_ + (__offset + __first) * sizeof(_Tp),
      __layout.__sorted_ + (__begin + __first) * sizeof(_Tp),
      __length * sizeof(_Tp));
  }
  __session.__end();
}

template <class _Tp>
_CCCL_KERNEL_ATTRIBUTES void __device_redistribute_rotated(
  ::ncclDevComm __comm, ::ncclGinBarrierHandle __barrier, ::ncclWindow_t __win, __device_layout __layout)
{
  __device_session<false> __session{__comm, __barrier};
  const auto __width = static_cast<::cuda::std::size_t>(__comm.nRanks) + 1;
  const auto* __cuts = static_cast<const __index_type*>(::ncclGetLocalPointer(__win, __layout.__cuts_));
  __session.__begin();
  // Spread large transfers across peers without a remainder in each iteration.
  const int __block_peer = static_cast<int>(blockIdx.x % __comm.nRanks);
  const int __first_peer =
    __block_peer < __comm.nRanks - __comm.rank
      ? __block_peer + __comm.rank
      : __block_peer - (__comm.nRanks - __comm.rank);
  for (int __step = 0; __step < __comm.nRanks; ++__step)
  {
    const int __peer =
      __step < __comm.nRanks - __first_peer ? __step + __first_peer : __step - (__comm.nRanks - __first_peer);
    const auto __begin    = __cuts[__comm.rank * __width + __peer];
    const auto __count    = __cuts[__comm.rank * __width + __peer + 1] - __begin;
    __index_type __cursor = 0, __offset = 0;
    for (int __source = 0; __source <= __comm.rank; ++__source)
    {
      const auto __first  = __cuts[__source * __width + __peer];
      const auto __length = __cuts[__source * __width + __peer + 1] - __first;
      __offset            = __cursor;
      if (__length)
      {
        __offset += __first % 16;
        __cursor = (__offset + __length + 15) / 16 * 16;
      }
    }
    const auto __first =
      __count / gridDim.x * blockIdx.x + (__count % gridDim.x < blockIdx.x ? __count % gridDim.x : blockIdx.x);
    const auto __length = __count / gridDim.x + (blockIdx.x < __count % gridDim.x);
    __session.__put(
      __win,
      __peer,
      __layout.__received_ + (__offset + __first) * sizeof(_Tp),
      __layout.__sorted_ + (__begin + __first) * sizeof(_Tp),
      __length * sizeof(_Tp));
  }
  __session.__end();
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)
#    include <cuda/std/__cccl/epilogue.h>
#  endif
#endif
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_DEVICE_H
