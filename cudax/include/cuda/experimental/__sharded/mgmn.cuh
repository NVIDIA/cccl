//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief The MGMN bridge: run the multi-GPU multi-node algorithms of
 *        `cuda/experimental/__multi_gpu/algorithm/` directly on sharded
 *        arrays, over the shared address space.
 *
 * Two pieces:
 *
 * 1. `places_communicator` — a communicator (in the sense of the MGMN
 *    `__communicator` concept) whose P ranks are the P places of one process
 *    and whose "network" is the shared address space: point-to-point and
 *    collective operations become `cudaMemcpyAsync` / tiny kernels ordered by
 *    events between the ranks' streams. Modeled on NCCL groups: operations
 *    issued while a group guard is alive are RECORDED and EXECUTED when the
 *    outermost guard is destroyed (group end), which is what lets a single
 *    host loop issue each rank's half of a send/recv pair or of a collective
 *    one after the other. Everything the communicator enqueues is
 *    capture-legal (event record / stream wait / memcpy / kernel launch; no
 *    host synchronization, no allocation).
 *
 * 2. The adapter — `make_communicators`, `mgmn_envs` — turns a sharded view
 *    and its per-shard environments into the five lockstep ranges the MGMN
 *    multi-local-rank overloads consume (communicators, environments, input
 *    iterators, sizes, output iterators), and the `mgmn::` verbs
 *    (`inclusive_scan`, `exclusive_scan`, `reduce_into_lanes`, `reduce`,
 *    `transform`) wrap that plumbing behind the sharded vocabulary.
 *
 * Contract of the `mgmn::` verbs: asynchronous and lane-ordered (the
 * composition contract of `composition.cuh`) — each shard's work is enqueued
 * on its environment's stream, cross-shard steps are event edges between
 * those streams, results are ready in stream order, no host synchronization
 * ever happens (so they capture into CUDA graphs). A call environment
 * carrying `composition::bracketed` seals the call against the call stream.
 *
 * The existing `sharded::` algorithms are untouched: they remain the
 * reference the MGMN path is checked against.
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

#include <cuda/__functional/operator_properties.h> // identity_element
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/algorithm/scan/scan.h>
#include <cuda/experimental/__multi_gpu/algorithm/transform/transform.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/reduce.cuh> // __partial_slots, __max_fold_shards
#include <cuda/experimental/__sharded/stream_scope.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Element-wise reduction of P rank buffers into one output, in rank
//! order (deterministic): `out[i] = op(...op(op(p_0[i], p_1[i]), p_2[i])...)`.
//! One launch per destination rank of an `all_reduce`, reading every rank's
//! send buffer over the shared address space.
template <typename _Tp, typename _ReduceOp>
__global__ void __mgmn_all_reduce_kernel(
  __partial_slots<_Tp> __slots, unsigned __num_ranks, ::std::size_t __count, _ReduceOp __op, _Tp* __out)
{
  const ::std::size_t __stride = static_cast<::std::size_t>(gridDim.x) * blockDim.x;
  for (::std::size_t __i = static_cast<::std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; __i < __count;
       __i += __stride)
  {
    _Tp __acc = __slots.__p[0][__i];
    for (unsigned __r = 1; __r < __num_ranks; ++__r)
    {
      __acc = __op(__acc, __slots.__p[__r][__i]);
    }
    __out[__i] = __acc;
  }
}
} // namespace reserved

// ============================================================================
// places_communicator: the shared-address-space communicator
// ============================================================================

/**
 * @brief A communicator over the P places of one process, satisfying the
 * MGMN `__communicator` concept (with `all_gather` and `all_reduce`).
 *
 * The P rank handles of one communicator group are values sharing one
 * state (`create(streams)` returns all P at once). Operations take a group
 * guard obtained from any rank (`group_guard()`) and are recorded; the
 * outermost guard's destruction executes them:
 *
 * - `send`/`recv`: each send is paired with the first unmatched receive of
 *   the same (source, destination) pair, in issue order (FIFO per pair).
 *   Edges: event on the sender's stream, waited by the receiver's stream;
 *   `cudaMemcpyAsync` device-to-device on the receiver's stream; event on
 *   the receiver's stream, waited by the sender's stream (the sender cannot
 *   reuse its buffer early). The byte count is fixed at issue time.
 * - `all_gather`: once every rank has issued its half, every destination
 *   stream waits for every source's event, copies each source's buffer
 *   into its slot (`recv + source * count`), records a completion event,
 *   and every source stream waits for every destination's completion. The
 *   in-place form `sendbuff == recvbuff + rank * count` skips the self copy.
 * - `all_reduce`: same edges; instead of copies, every destination launches
 *   one kernel reading all P send buffers in rank order and writing its
 *   receive buffer. Not in place: a receive buffer must not overlap any send
 *   buffer (`std::invalid_argument`). At most 64 ranks.
 *
 * Streams given to rank r's operations must live on rank r's device (the
 * device of the stream `create` was given for r): events are pre-created per
 * rank and recorded on those streams. Only stream operations are enqueued —
 * event record, stream wait, memcpy, kernel launch — so a group issued
 * inside a CUDA graph capture captures. Not thread-safe: one group is one
 * host thread's sequence.
 */
class places_communicator
{
  struct __state;

public:
  //! @brief The identity of the communicator group (all ranks share it).
  using native_handle_type = const void*;

  //! @brief RAII group: constructed by `group_guard()`, executes the recorded
  //! operations of the group on destruction of the outermost guard. If the
  //! guard is destroyed by stack unwinding, the pending operations are
  //! discarded instead of executed (nothing is enqueued).
  class group_guard
  {
  public:
    explicit group_guard(::std::shared_ptr<__state> __state)
        : __state_(::std::move(__state))
        , __uncaught_on_entry_(::std::uncaught_exceptions())
    {
      ++__state_->__depth;
    }

    group_guard(const group_guard&)            = delete;
    group_guard& operator=(const group_guard&) = delete;
    group_guard(group_guard&&)                 = delete;
    group_guard& operator=(group_guard&&)      = delete;

    ~group_guard() noexcept(false)
    {
      if (--__state_->__depth != 0)
      {
        return;
      }
      if (::std::uncaught_exceptions() > __uncaught_on_entry_)
      {
        __state_->__discard();
        return;
      }
      __state_->__flush();
    }

  private:
    ::std::shared_ptr<__state> __state_;
    int __uncaught_on_entry_;
  };

  using group_guard_type = group_guard;

  places_communicator() = default;

  /**
   * @brief Create the P ranks of one communicator group.
   *
   * @param home_streams One stream per rank; rank r's events are created on
   *        the device of `home_streams[r]`. (The streams are not retained.)
   * @return The P rank handles, `result[r].rank() == r`.
   */
  [[nodiscard]] static ::std::vector<places_communicator> create(const ::std::vector<::cuda::stream_ref>& home_streams)
  {
    const ::std::size_t __n = home_streams.size();
    if (__n > reserved::__max_fold_shards)
    {
      _CCCL_THROW(::std::invalid_argument, "places_communicator::create: more than 64 ranks not supported");
    }
    auto __st    = ::std::make_shared<__state>();
    __st->__size = static_cast<::cuda::std::int32_t>(__n);
    __st->__devices.reserve(__n);
    for (const auto& __s : home_streams)
    {
      __st->__devices.push_back(places::get_device_from_stream(__s.get()));
    }
    __st->__events.resize(__n);
    // Two events per rank per collective (source-ready, destination-done):
    // pre-create a few so a group issued under capture needs no creation.
    for (::std::size_t __r = 0; __r < __n; ++__r)
    {
      for (int __k = 0; __k < 4; ++__k)
      {
        __st->__events[__r].push_back(__state::__create_event(__st->__devices[__r]));
      }
    }
    ::std::vector<places_communicator> __result;
    __result.reserve(__n);
    for (::std::size_t __r = 0; __r < __n; ++__r)
    {
      __result.push_back(places_communicator{__st, static_cast<::cuda::std::int32_t>(__r)});
    }
    return __result;
  }

  //! @brief The rank of this handle in its group, in `[0, size())`.
  [[nodiscard]] ::cuda::std::int32_t rank() const noexcept
  {
    return __rank_;
  }

  //! @brief The number of ranks of the group.
  [[nodiscard]] ::cuda::std::int32_t size() const noexcept
  {
    return __state_ ? __state_->__size : 0;
  }

  //! @brief The identity of the group (equal for all its ranks).
  [[nodiscard]] native_handle_type native_handle() const noexcept
  {
    return __state_.get();
  }

  //! @brief Open a group (see the class documentation). Any rank may open
  //! it; the guard covers every rank of the group.
  [[nodiscard]] group_guard_type group_guard() const
  {
    __require_state("group_guard");
    return group_guard_type{__state_};
  }

  //! @brief Record a send of @p count elements of @p buf to rank @p peer,
  //! ordered on @p stream (a stream of this rank's device).
  template <class _Tp>
  void send(
    group_guard_type&, const _Tp* buf, ::std::size_t count, ::cuda::std::int32_t peer, ::cuda::stream_ref stream) const
  {
    __require_open("send");
    __check_peer(peer, "send");
    __state_->__p2p.push_back(__state::__p2p_op{
      __rank_,
      peer,
      /*__is_send=*/true,
      const_cast<void*>(static_cast<const void*>(buf)),
      count * sizeof(_Tp),
      stream.get()});
  }

  //! @brief Record a receive of @p count elements from rank @p peer into
  //! @p buf, ordered on @p stream (a stream of this rank's device).
  template <class _Tp>
  void recv(group_guard_type&, _Tp* buf, ::std::size_t count, ::cuda::std::int32_t peer, ::cuda::stream_ref stream) const
  {
    __require_open("recv");
    __check_peer(peer, "recv");
    __state_->__p2p.push_back(__state::__p2p_op{
      __rank_, peer, /*__is_send=*/false, static_cast<void*>(buf), count * sizeof(_Tp), stream.get()});
  }

  //! @brief Record this rank's half of an all-gather: after the group ends,
  //! `recvbuff[r * count, (r + 1) * count)` holds rank r's @p sendbuff for
  //! every rank r, on every rank. In place when `sendbuff == recvbuff + rank() * count`.
  template <class _Tp>
  void all_gather(
    group_guard_type&, const _Tp* sendbuff, _Tp* recvbuff, ::std::size_t count, ::cuda::stream_ref stream) const
  {
    __require_open("all_gather");
    auto& __coll = __state_->__slot(__state::__collective_kind::__all_gather, __rank_);
    __coll.__slots[static_cast<::std::size_t>(__rank_)] = __state::__collective_slot{
      static_cast<const void*>(sendbuff), static_cast<void*>(recvbuff), count * sizeof(_Tp), stream.get(), true};
    if (!__coll.__execute)
    {
      __coll.__execute = [](__state& __st, __state::__collective& __c) {
        __st.__run_all_gather(__c);
      };
    }
  }

  //! @brief Record this rank's half of an all-reduce: after the group ends,
  //! every rank's @p recvbuff holds the element-wise @p op fold of all ranks'
  //! @p sendbuff, folded in rank order. Not in place.
  template <class _Tp, class _ReduceOp>
  void all_reduce(
    group_guard_type&, const _Tp* sendbuff, _Tp* recvbuff, ::std::size_t count, _ReduceOp op, ::cuda::stream_ref stream)
    const
  {
    __require_open("all_reduce");
    auto& __coll = __state_->__slot(__state::__collective_kind::__all_reduce, __rank_);
    __coll.__slots[static_cast<::std::size_t>(__rank_)] = __state::__collective_slot{
      static_cast<const void*>(sendbuff), static_cast<void*>(recvbuff), count * sizeof(_Tp), stream.get(), true};
    if (!__coll.__execute)
    {
      __coll.__execute = [op, count](__state& __st, __state::__collective& __c) {
        __st.__run_all_reduce<_Tp>(__c, count, op);
      };
    }
  }

  [[nodiscard]] friend bool operator==(const places_communicator& __a, const places_communicator& __b) noexcept
  {
    return __a.__state_ == __b.__state_ && __a.__rank_ == __b.__rank_;
  }
  [[nodiscard]] friend bool operator!=(const places_communicator& __a, const places_communicator& __b) noexcept
  {
    return !(__a == __b);
  }

private:
  places_communicator(::std::shared_ptr<__state> __st, ::cuda::std::int32_t __rank)
      : __state_(::std::move(__st))
      , __rank_(__rank)
  {}

  void __require_state(const char* __what) const
  {
    if (!__state_)
    {
      _CCCL_THROW(::std::logic_error, ::std::string("places_communicator::") + __what + ": default-constructed handle");
    }
  }

  void __require_open(const char* __what) const
  {
    __require_state(__what);
    if (__state_->__depth == 0)
    {
      _CCCL_THROW(::std::logic_error, ::std::string("places_communicator::") + __what + ": no group is open");
    }
  }

  void __check_peer(::cuda::std::int32_t __peer, const char* __what) const
  {
    if (__peer < 0 || __peer >= __state_->__size)
    {
      _CCCL_THROW(::std::invalid_argument, ::std::string("places_communicator::") + __what + ": peer out of range");
    }
  }

  struct __state
  {
    struct __p2p_op
    {
      ::cuda::std::int32_t __rank;
      ::cuda::std::int32_t __peer;
      bool __is_send;
      void* __buf;
      ::std::size_t __bytes;
      cudaStream_t __stream;
    };

    struct __collective_slot
    {
      const void* __send    = nullptr;
      void* __recv          = nullptr;
      ::std::size_t __bytes = 0; // count * sizeof(T)
      cudaStream_t __stream = nullptr;
      bool __present        = false;
    };

    enum class __collective_kind
    {
      __all_gather,
      __all_reduce
    };

    struct __collective
    {
      __collective_kind __kind;
      ::std::vector<__collective_slot> __slots;
      ::std::function<void(__state&, __collective&)> __execute;
    };

    ::cuda::std::int32_t __size = 0;
    int __depth                 = 0;
    ::std::vector<int> __devices; // per rank
    ::std::vector<::std::vector<cudaEvent_t>> __events; // per rank pool
    ::std::vector<::std::size_t> __cursor; // per rank, valid during a flush
    ::std::vector<__p2p_op> __p2p;
    ::std::vector<__collective> __collectives;

    __state()                          = default;
    __state(const __state&)            = delete;
    __state& operator=(const __state&) = delete;

    ~__state()
    {
      for (auto& __pool : __events)
      {
        for (auto __ev : __pool)
        {
          (void) cudaEventDestroy(__ev);
        }
      }
    }

    static cudaEvent_t __create_event(int __device)
    {
      int __prev = -1;
      cuda_safe_call(cudaGetDevice(&__prev));
      if (__prev != __device)
      {
        cuda_safe_call(cudaSetDevice(__device));
      }
      cudaEvent_t __ev           = nullptr;
      const cudaError_t __status = cudaEventCreateWithFlags(&__ev, cudaEventDisableTiming);
      if (__prev != __device)
      {
        cuda_safe_call(cudaSetDevice(__prev));
      }
      cuda_safe_call(__status);
      return __ev;
    }

    //! @brief Next event of rank @p __rank's pool (grows on demand).
    cudaEvent_t __next_event(::cuda::std::int32_t __rank)
    {
      auto& __pool = __events[static_cast<::std::size_t>(__rank)];
      auto& __cur  = __cursor[static_cast<::std::size_t>(__rank)];
      if (__cur == __pool.size())
      {
        __pool.push_back(__create_event(__devices[static_cast<::std::size_t>(__rank)]));
      }
      return __pool[__cur++];
    }

    //! @brief Record an event on @p __stream (a stream of rank @p __rank).
    cudaEvent_t __record(::cuda::std::int32_t __rank, cudaStream_t __stream)
    {
      const cudaEvent_t __ev = __next_event(__rank);
      cuda_safe_call(cudaEventRecord(__ev, __stream));
      return __ev;
    }

    static void __wait(cudaStream_t __stream, cudaEvent_t __ev)
    {
      cuda_safe_call(cudaStreamWaitEvent(__stream, __ev, 0));
    }

    //! @brief The first collective of @p __kind (in issue order) in which
    //! rank @p __rank has not yet taken part; a new one when none.
    __collective& __slot(__collective_kind __kind, ::cuda::std::int32_t __rank)
    {
      for (auto& __c : __collectives)
      {
        if (__c.__kind == __kind && !__c.__slots[static_cast<::std::size_t>(__rank)].__present)
        {
          return __c;
        }
      }
      __collectives.push_back(
        __collective{__kind, ::std::vector<__collective_slot>(static_cast<::std::size_t>(__size)), {}});
      return __collectives.back();
    }

    void __discard() noexcept
    {
      __p2p.clear();
      __collectives.clear();
    }

    //! @brief Group end: execute every recorded operation.
    void __flush()
    {
      __cursor.assign(static_cast<::std::size_t>(__size), 0);
      // Take the queues first so a throwing execution leaves the state clean.
      auto __colls = ::std::move(__collectives);
      auto __ops   = ::std::move(__p2p);
      __collectives.clear();
      __p2p.clear();

      for (auto& __c : __colls)
      {
        for (const auto& __s : __c.__slots)
        {
          if (!__s.__present)
          {
            _CCCL_THROW(::std::logic_error,
                        "places_communicator: group ended with a collective some ranks did not join");
          }
        }
        __c.__execute(*this, __c);
      }
      __run_p2p(__ops);
    }

    //! @brief Pair every send with the first unmatched receive of the same
    //! (source, destination) pair, in issue order, and run the transfers.
    void __run_p2p(::std::vector<__p2p_op>& __ops)
    {
      ::std::vector<bool> __done(__ops.size(), false);
      for (::std::size_t __i = 0; __i < __ops.size(); ++__i)
      {
        if (!__ops[__i].__is_send)
        {
          continue;
        }
        const auto& __snd = __ops[__i];
        ::std::size_t __j = 0;
        for (; __j < __ops.size(); ++__j)
        {
          const auto& __rcv = __ops[__j];
          if (!__done[__j] && !__rcv.__is_send && __rcv.__rank == __snd.__peer && __rcv.__peer == __snd.__rank)
          {
            break;
          }
        }
        if (__j == __ops.size())
        {
          _CCCL_THROW(::std::logic_error, "places_communicator: group ended with an unmatched send");
        }
        const auto& __rcv = __ops[__j];
        if (__rcv.__bytes != __snd.__bytes)
        {
          _CCCL_THROW(::std::invalid_argument, "places_communicator: send/recv byte counts differ");
        }
        __done[__i] = __done[__j] = true;
        __copy(__snd.__rank, __snd.__stream, __snd.__buf, __rcv.__rank, __rcv.__stream, __rcv.__buf, __snd.__bytes);
      }
      for (::std::size_t __j = 0; __j < __ops.size(); ++__j)
      {
        if (!__done[__j])
        {
          _CCCL_THROW(::std::logic_error, "places_communicator: group ended with an unmatched recv");
        }
      }
    }

    //! @brief One paired transfer: `dst <- src` on the destination stream,
    //! bracketed by the two event edges.
    void __copy(::cuda::std::int32_t __src_rank,
                cudaStream_t __src_stream,
                const void* __src,
                ::cuda::std::int32_t __dst_rank,
                cudaStream_t __dst_stream,
                void* __dst,
                ::std::size_t __bytes)
    {
      const bool __same = __src_stream == __dst_stream;
      if (!__same)
      {
        __wait(__dst_stream, __record(__src_rank, __src_stream));
      }
      if (__bytes != 0)
      {
        cuda_safe_call(cudaMemcpyAsync(__dst, __src, __bytes, cudaMemcpyDefault, __dst_stream));
      }
      if (!__same)
      {
        __wait(__src_stream, __record(__dst_rank, __dst_stream));
      }
    }

    //! @brief Fan-in edges of a collective: every destination waits for
    //! every other source's buffer to be ready.
    void __fan_in(const __collective& __c)
    {
      const ::std::size_t __n = __c.__slots.size();
      if (__n < 2)
      {
        return;
      }
      ::std::vector<cudaEvent_t> __ready(__n, nullptr);
      for (::std::size_t __s = 0; __s < __n; ++__s)
      {
        __ready[__s] = __record(static_cast<::cuda::std::int32_t>(__s), __c.__slots[__s].__stream);
      }
      for (::std::size_t __d = 0; __d < __n; ++__d)
      {
        for (::std::size_t __s = 0; __s < __n; ++__s)
        {
          if (__s != __d && __c.__slots[__s].__stream != __c.__slots[__d].__stream)
          {
            __wait(__c.__slots[__d].__stream, __ready[__s]);
          }
        }
      }
    }

    //! @brief Fan-out edges of a collective: every source waits for every
    //! other destination to be done reading its buffer.
    void __fan_out(const __collective& __c)
    {
      const ::std::size_t __n = __c.__slots.size();
      if (__n < 2)
      {
        return;
      }
      ::std::vector<cudaEvent_t> __done(__n, nullptr);
      for (::std::size_t __d = 0; __d < __n; ++__d)
      {
        __done[__d] = __record(static_cast<::cuda::std::int32_t>(__d), __c.__slots[__d].__stream);
      }
      for (::std::size_t __s = 0; __s < __n; ++__s)
      {
        for (::std::size_t __d = 0; __d < __n; ++__d)
        {
          if (__s != __d && __c.__slots[__s].__stream != __c.__slots[__d].__stream)
          {
            __wait(__c.__slots[__s].__stream, __done[__d]);
          }
        }
      }
    }

    void __run_all_gather(__collective& __c)
    {
      const ::std::size_t __n     = __c.__slots.size();
      const ::std::size_t __bytes = __c.__slots[0].__bytes;
      for (const auto& __s : __c.__slots)
      {
        if (__s.__bytes != __bytes)
        {
          _CCCL_THROW(::std::invalid_argument, "places_communicator::all_gather: counts differ across ranks");
        }
      }
      __fan_in(__c);
      for (::std::size_t __d = 0; __d < __n; ++__d)
      {
        const auto& __dst = __c.__slots[__d];
        for (::std::size_t __s = 0; __s < __n; ++__s)
        {
          void* const __to = static_cast<char*>(__dst.__recv) + __s * __bytes;
          if (__bytes == 0 || (__s == __d && __to == __c.__slots[__s].__send))
          {
            continue; // in place: the self slot is already there
          }
          cuda_safe_call(cudaMemcpyAsync(__to, __c.__slots[__s].__send, __bytes, cudaMemcpyDefault, __dst.__stream));
        }
      }
      __fan_out(__c);
    }

    template <class _Tp, class _ReduceOp>
    void __run_all_reduce(__collective& __c, ::std::size_t __count, const _ReduceOp& __op)
    {
      const ::std::size_t __n     = __c.__slots.size();
      const ::std::size_t __bytes = __count * sizeof(_Tp);
      reserved::__partial_slots<_Tp> __ptrs{};
      for (::std::size_t __s = 0; __s < __n; ++__s)
      {
        if (__c.__slots[__s].__bytes != __bytes)
        {
          _CCCL_THROW(::std::invalid_argument, "places_communicator::all_reduce: counts differ across ranks");
        }
        __ptrs.__p[__s] = static_cast<const _Tp*>(__c.__slots[__s].__send);
      }
      // Not in place: a destination must not overlap any source (the P
      // destination kernels read every source concurrently).
      for (::std::size_t __d = 0; __d < __n; ++__d)
      {
        const char* const __rb = static_cast<const char*>(__c.__slots[__d].__recv);
        for (::std::size_t __s = 0; __s < __n; ++__s)
        {
          const char* const __sb = static_cast<const char*>(__c.__slots[__s].__send);
          if (__bytes != 0 && __rb < __sb + __bytes && __sb < __rb + __bytes)
          {
            _CCCL_THROW(::std::invalid_argument, "places_communicator::all_reduce: in-place reduction not supported");
          }
        }
      }
      __fan_in(__c);
      if (__count != 0)
      {
        const unsigned __threads            = 256;
        const ::std::size_t __blocks_needed = (__count + __threads - 1) / __threads;
        const unsigned __blocks             = static_cast<unsigned>(__blocks_needed < 1024 ? __blocks_needed : 1024);
        for (::std::size_t __d = 0; __d < __n; ++__d)
        {
          const auto& __dst = __c.__slots[__d];
          stream_scope __scope(__dst.__stream);
          reserved::__mgmn_all_reduce_kernel<_Tp, _ReduceOp><<<__blocks, __threads, 0, __dst.__stream>>>(
            __ptrs, static_cast<unsigned>(__n), __count, __op, static_cast<_Tp*>(__dst.__recv));
          cuda_safe_call(cudaGetLastError());
        }
      }
      __fan_out(__c);
    }
  };

  ::std::shared_ptr<__state> __state_;
  ::cuda::std::int32_t __rank_ = 0;
};

static_assert(::cuda::experimental::__communicator<places_communicator>);
static_assert(::cuda::experimental::__has_all_gather<places_communicator>);
static_assert(::cuda::experimental::__has_all_reduce<places_communicator>);

// ============================================================================
// The adapter: sharded environments -> MGMN environments and communicators
// ============================================================================

namespace reserved
{
/**
 * @brief Give a stream-ordered memory resource the `device_accessible`
 * property and the `default_queries` the MGMN algorithms need to build
 * their `cuda::buffer` temporaries (`place_memory_resource` allocates at a
 * device place but does not advertise it). Every allocation still goes
 * through the wrapped resource: the environment's resource is the one that
 * allocates.
 */
template <class _Resource>
class __device_accessible_adapter
{
public:
  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

  explicit __device_accessible_adapter(_Resource __mr)
      : __mr_(::std::move(__mr))
  {}

  [[nodiscard]] void* allocate(::cuda::stream_ref __stream, ::std::size_t __bytes, ::std::size_t __alignment)
  {
    return __mr_.allocate(__stream, __bytes, __alignment);
  }
  [[nodiscard]] void* allocate(::cuda::stream_ref __stream, ::std::size_t __bytes)
  {
    return __mr_.allocate(__stream, __bytes);
  }
  void deallocate(::cuda::stream_ref __stream, void* __ptr, ::std::size_t __bytes, ::std::size_t __alignment)
  {
    __mr_.deallocate(__stream, __ptr, __bytes, __alignment);
  }
  void deallocate(::cuda::stream_ref __stream, void* __ptr, ::std::size_t __bytes)
  {
    __mr_.deallocate(__stream, __ptr, __bytes);
  }
  [[nodiscard]] void* allocate_sync(::std::size_t __bytes, ::std::size_t __alignment)
  {
    return __mr_.allocate_sync(__bytes, __alignment);
  }
  [[nodiscard]] void* allocate_sync(::std::size_t __bytes)
  {
    return __mr_.allocate_sync(__bytes);
  }
  void deallocate_sync(void* __ptr, ::std::size_t __bytes, ::std::size_t __alignment)
  {
    __mr_.deallocate_sync(__ptr, __bytes, __alignment);
  }
  void deallocate_sync(void* __ptr, ::std::size_t __bytes)
  {
    __mr_.deallocate_sync(__ptr, __bytes);
  }

  //! @brief The wrapped resource.
  [[nodiscard]] const _Resource& resource() const noexcept
  {
    return __mr_;
  }

  [[nodiscard]] friend bool
  operator==(const __device_accessible_adapter& __a, const __device_accessible_adapter& __b) noexcept
  {
    return __a.__mr_ == __b.__mr_;
  }
  [[nodiscard]] friend bool
  operator!=(const __device_accessible_adapter& __a, const __device_accessible_adapter& __b) noexcept
  {
    return !(__a == __b);
  }

  friend constexpr void get_property(const __device_accessible_adapter&, ::cuda::mr::device_accessible) noexcept {}

private:
  _Resource __mr_;
};

//! @brief The environment's resource as the MGMN algorithms can consume it:
//! unchanged when it already advertises `default_queries`, otherwise wrapped.
template <class _Env>
[[nodiscard]] auto __mgmn_resource(const _Env& __env)
{
  using __raw_t = ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(__env))>;
  if constexpr (::cuda::mr::__has_default_queries<__raw_t>)
  {
    return __raw_t{::cuda::mr::get_memory_resource(__env)};
  }
  else
  {
    return __device_accessible_adapter<__raw_t>{::cuda::mr::get_memory_resource(__env)};
  }
}

//! @brief One MGMN environment from one sharded environment: its stream and
//! its (adapted) memory resource, nothing else.
template <class _Env>
[[nodiscard]] auto __mgmn_env(const _Env& __env)
{
  using __mr_t = decltype(__mgmn_resource(__env));
  return ::cuda::std::execution::env<::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>,
                                     ::cuda::std::execution::prop<::cuda::mr::get_memory_resource_t, __mr_t>>{
    ::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>{
      ::cuda::get_stream, ::cuda::stream_ref{::cuda::get_stream(__env)}},
    ::cuda::std::execution::prop<::cuda::mr::get_memory_resource_t, __mr_t>{
      ::cuda::mr::get_memory_resource, __mgmn_resource(__env)}};
}
} // namespace reserved

//! @brief The MGMN environment type derived from the environments of an
//! allocating environment range.
template <class _Envs>
using mgmn_env_t =
  decltype(reserved::__mgmn_env(::cuda::std::declval<const ::cuda::std::remove_cvref_t<_Envs>&>()[::std::size_t{0}]));

/**
 * @brief The first @p count environments of @p envs as MGMN environments
 * (stream + memory resource), the range the MGMN algorithms iterate in
 * lockstep with the communicators.
 */
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<mgmn_env_t<_Envs>> mgmn_envs(const _Envs& envs, ::std::size_t count)
{
  if (reserved::__env_count(envs) < count)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn_envs: fewer environments than requested");
  }
  ::std::vector<mgmn_env_t<_Envs>> __result;
  __result.reserve(count);
  for (const auto __i : each(count))
  {
    __result.push_back(reserved::__mgmn_env(envs[__i]));
  }
  return __result;
}

/// @brief All environments of @p envs as MGMN environments.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<mgmn_env_t<_Envs>> mgmn_envs(const _Envs& envs)
{
  return sharded::mgmn_envs(envs, reserved::__env_count(envs));
}

/**
 * @brief One communicator group over the first @p count environments of
 * @p envs: rank i is environment i (its stream's device).
 */
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<places_communicator> make_communicators(const _Envs& envs, ::std::size_t count)
{
  if (reserved::__env_count(envs) < count)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::make_communicators: fewer environments than requested");
  }
  ::std::vector<::cuda::stream_ref> __streams;
  __streams.reserve(count);
  for (const auto __i : each(count))
  {
    __streams.push_back(::cuda::stream_ref{::cuda::get_stream(envs[__i])});
  }
  return places_communicator::create(__streams);
}

/// @brief One communicator group over all environments of @p envs.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<places_communicator> make_communicators(const _Envs& envs)
{
  return sharded::make_communicators(envs, reserved::__env_count(envs));
}

namespace reserved
{
//! @brief Per-shard data pointers of a view, as the MGMN iterator range.
template <class _Ptr, class _S>
[[nodiscard]] ::std::vector<_Ptr> __mgmn_pointers(const _S& __s)
{
  const ::std::size_t __n = __shard_count(__s);
  ::std::vector<_Ptr> __result;
  __result.reserve(__n);
  for (const auto __g : each(__n))
  {
    __result.push_back(static_cast<_Ptr>(__s.shard(__g).data));
  }
  return __result;
}

//! @brief Per-shard sizes of a view, as the MGMN size range.
template <class _S>
[[nodiscard]] ::std::vector<::std::size_t> __mgmn_sizes(const _S& __s)
{
  const ::std::size_t __n = __shard_count(__s);
  ::std::vector<::std::size_t> __result;
  __result.reserve(__n);
  for (const auto __g : each(__n))
  {
    __result.push_back(static_cast<::std::size_t>(__s.shard(__g).size));
  }
  return __result;
}

//! @brief The bracket of a `composition::bracketed` call environment: fork
//! the lanes from the call stream on entry, join them into it on exit.
//! Lane-ordered call environments (the default) bracket nothing.
template <class _Envs, class _CallEnv>
class __mgmn_bracket
{
public:
  __mgmn_bracket(const _Envs& __envs, ::std::size_t __count, const _CallEnv& __call_env)
      : __envs_(__envs)
      , __count_(__count)
  {
    if constexpr (async_call_env<_CallEnv>)
    {
      if (query_composition(__call_env) == composition::bracketed)
      {
        __stream_ = ::cuda::get_stream(__call_env).get();
        for (const auto __g : each(__count_))
        {
          __detail::__wait_stream_on(::cuda::get_stream(__envs_[__g]).get(), __stream_);
        }
      }
    }
  }

  __mgmn_bracket(const __mgmn_bracket&)            = delete;
  __mgmn_bracket& operator=(const __mgmn_bracket&) = delete;

  ~__mgmn_bracket()
  {
    if (__stream_ != nullptr)
    {
      for (const auto __g : each(__count_))
      {
        __detail::__wait_stream_on(__stream_, ::cuda::get_stream(__envs_[__g]).get());
      }
    }
  }

private:
  const _Envs& __envs_;
  ::std::size_t __count_;
  cudaStream_t __stream_ = nullptr;
};

//! @brief Shared driver of the MGMN scans over sharded views.
template <bool _Inclusive, class _SIn, class _SOut, class _Envs, class _ScanOp, class _Tp, class _CallEnv>
_CCCL_HOST_API void __mgmn_scan(
  const _SIn& __in,
  _SOut&& __out,
  const _Envs& __envs,
  _ScanOp __op,
  _Tp __init,
  _Tp __identity,
  const _CallEnv& __call_env,
  const char* __what)
{
  const ::std::size_t __n = __shard_count(__in);
  if (__env_count(__envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }
  __check_copartitioned(__in, __out, __what);
  if (__n == 0)
  {
    return;
  }
  const __mgmn_bracket<_Envs, _CallEnv> __bracket(__envs, __n, __call_env);

  const auto __comms   = sharded::make_communicators(__envs, __n);
  const auto __menvs   = sharded::mgmn_envs(__envs, __n);
  const auto __inputs  = __mgmn_pointers<const _Tp*>(__in);
  const auto __sizes   = __mgmn_sizes(__in);
  const auto __outputs = __mgmn_pointers<_Tp*>(__out);

  if constexpr (_Inclusive)
  {
    ::cuda::experimental::inclusive_scan(
      ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
  }
  else
  {
    ::cuda::experimental::exclusive_scan(
      ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
  }
}
} // namespace reserved

// ============================================================================
// The mgmn:: verbs: MGMN algorithms behind the sharded vocabulary
// ============================================================================

namespace mgmn
{
/**
 * @brief Inclusive scan `out[i] = fold(in[0..i])` over the global index
 * space, through the MGMN scan (reduce, all-gather of the P partials, device
 * prefix, seeded per-shard `cub::DeviceScan`). Asynchronous, lane-ordered,
 * capturable. @p out must be co-partitioned with @p in; it may be @p in.
 *
 * @p identity is the operator's identity element, defaulted where
 * `cuda::identity_element` knows the operator; custom operators supply it.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  const _SIn& in,
  _SOut&& out,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  reserved::__mgmn_scan<true>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief In-place inclusive scan (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<true>(data, data, envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief Inclusive scan into @p out (self-bound: environments of @p in).
_CCCL_TEMPLATE(class _SIn, class _SOut, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  const _SIn& in,
  _SOut&& out,
  _ScanOp scan_op,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  const auto envs = default_envs(in);
  reserved::__mgmn_scan<true>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief In-place inclusive scan (self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<true>(data, data, envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/**
 * @brief Exclusive scan `out[i] = fold(init, in[0..i-1])` over the global
 * index space — the global semantics, init entering the fold exactly once —
 * through the MGMN scan. Asynchronous, lane-ordered, capturable. @p out must
 * be co-partitioned with @p in; it may be @p in.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  const _SIn& in,
  _SOut&& out,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_SIn> init_value,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  reserved::__mgmn_scan<false>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place exclusive scan (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<false>(
    data, data, envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief Exclusive scan into @p out (self-bound: environments of @p in).
_CCCL_TEMPLATE(class _SIn, class _SOut, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  const _SIn& in,
  _SOut&& out,
  _ScanOp scan_op,
  view_element_t<_SIn> init_value,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  const auto envs = default_envs(in);
  reserved::__mgmn_scan<false>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place exclusive scan (self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<false>(
    data, data, envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place inclusive sum (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_sum(_S&& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  mgmn::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place inclusive sum (self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
_CCCL_HOST_API void inclusive_sum(_S&& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  mgmn::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
exclusive_sum(_S&& data, const _Envs& envs, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  mgmn::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
_CCCL_HOST_API void exclusive_sum(_S&& data, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  mgmn::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}

/**
 * @brief Reduction through the MGMN reduce (per-shard `cub::DeviceReduce`,
 * then `all_reduce` of the P partials): `init (+) fold(all elements)`, the
 * init entering exactly once, written to EVERY lane's output — `outs[g]` on
 * lane g's stream (the MGMN "broadcasted" result). Asynchronous,
 * lane-ordered, capturable.
 *
 * @param outs Indexable range of P device pointers (or output iterators),
 *        one per shard.
 * @param identity The operator's identity element (the partial of an empty
 *        shard); defaulted where `cuda::identity_element` knows the operator.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Outs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void reduce_into_lanes(
  const _S& data,
  const _Envs& envs,
  const _Outs& outs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const ::std::size_t __n = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn::reduce_into_lanes: fewer environments than shards");
  }
  if (static_cast<::std::size_t>(outs.size()) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn::reduce_into_lanes: fewer outputs than shards");
  }
  if (__n == 0)
  {
    return;
  }
  const reserved::__mgmn_bracket<_Envs, _CallEnv> __bracket(envs, __n, call_env);

  using __out_t       = ::cuda::std::remove_cvref_t<decltype(outs[::std::size_t{0}])>;
  const auto __comms  = sharded::make_communicators(envs, __n);
  const auto __menvs  = sharded::mgmn_envs(envs, __n);
  const auto __inputs = reserved::__mgmn_pointers<const _Tp*>(data);
  const auto __sizes  = reserved::__mgmn_sizes(data);
  ::std::vector<__out_t> __outputs;
  __outputs.reserve(__n);
  for (const auto __g : each(__n))
  {
    __outputs.push_back(outs[__g]);
  }

  ::cuda::experimental::reduce(
    ::cuda::experimental::broadcasted, __comms, __menvs, __inputs, __sizes, __outputs, init_value, reduce_op, identity);
}

/// @brief As above, self-bound (environments of @p data).
_CCCL_TEMPLATE(class _S, class _Outs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Outs>>))
_CCCL_HOST_API void reduce_into_lanes(
  const _S& data,
  const _Outs& outs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  mgmn::reduce_into_lanes(data, envs, outs, reduce_op, init_value, identity, call_env);
}

/**
 * @brief Synchronous reduction through the MGMN reduce, returning the value:
 * `init (+) fold(all elements)`. Convenience over `reduce_into_lanes` (one
 * scratch scalar per lane from the lane's resource, lane 0's copied back);
 * refuses under `sync_policy::forbid` and under capture.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API _Tp reduce(
  const _S& data,
  const _Envs& envs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  constexpr const char* __what = "sharded::mgmn::reduce";
  const ::std::size_t __n      = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }
  require_sync_allowed(call_env, __what);
  places::check_not_capturing(nullptr, __what);
  for (const auto __g : each(__n))
  {
    places::check_not_capturing(::cuda::get_stream(envs[__g]).get(), __what);
  }
  if (__n == 0)
  {
    return init_value;
  }

  ::std::vector<_Tp*> __outs(__n, nullptr);
  for (const auto __g : each(__n))
  {
    auto __mr   = ::cuda::mr::get_memory_resource(envs[__g]);
    __outs[__g] = static_cast<_Tp*>(__mr.allocate(::cuda::get_stream(envs[__g]), sizeof(_Tp), alignof(_Tp)));
  }
  mgmn::reduce_into_lanes(data, envs, __outs, reduce_op, init_value, identity);

  _Tp __result                  = init_value;
  const ::cuda::stream_ref __s0 = ::cuda::get_stream(envs[0]);
  cuda_safe_call(cudaMemcpyAsync(&__result, __outs[0], sizeof(_Tp), cudaMemcpyDefault, __s0.get()));
  cuda_safe_call(cudaStreamSynchronize(__s0.get()));
  for (const auto __g : each(__n))
  {
    auto __mr = ::cuda::mr::get_memory_resource(envs[__g]);
    __mr.deallocate(::cuda::get_stream(envs[__g]), __outs[__g], sizeof(_Tp), alignof(_Tp));
  }
  return __result;
}

/// @brief As above, self-bound (environments of @p data).
_CCCL_TEMPLATE(class _S, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ReduceOp>>))
[[nodiscard]] _CCCL_HOST_API _Tp reduce(
  const _S& data,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return mgmn::reduce(data, envs, reduce_op, init_value, identity, call_env);
}

/**
 * @brief Unary transform `out[i] = op(in[i])` through the MGMN transform
 * (rank-local `cub::DeviceTransform`). Asynchronous, lane-ordered,
 * capturable. @p out must be co-partitioned with @p in; it may be @p in.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void transform(const _SIn& in, _SOut&& out, const _Envs& envs, _UnaryOp op, const _CallEnv& call_env = {})
{
  constexpr const char* __what = "sharded::mgmn::transform";
  using __in_t                 = view_element_t<_SIn>;
  using __out_t                = view_element_t<_SOut>;
  const ::std::size_t __n      = reserved::__shard_count(in);
  if (reserved::__env_count(envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }
  reserved::__check_copartitioned(in, out, __what);
  if (__n == 0)
  {
    return;
  }
  const reserved::__mgmn_bracket<_Envs, _CallEnv> __bracket(envs, __n, call_env);

  const auto __comms   = sharded::make_communicators(envs, __n);
  const auto __inputs  = reserved::__mgmn_pointers<const __in_t*>(in);
  const auto __sizes   = reserved::__mgmn_sizes(in);
  const auto __outputs = reserved::__mgmn_pointers<__out_t*>(out);

  // The MGMN transform allocates nothing: the environments' streams suffice
  // (a non-allocating environment range is accepted, as for the map family).
  ::std::vector<::cuda::std::execution::env<::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>>>
    __menvs;
  __menvs.reserve(__n);
  for (const auto __g : each(__n))
  {
    __menvs.push_back(
      ::cuda::std::execution::env<::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>>{
        ::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>{
          ::cuda::get_stream, ::cuda::stream_ref{::cuda::get_stream(envs[__g])}}});
  }

  ::cuda::experimental::transform(::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, op);
}

/// @brief In-place unary transform (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void transform(_S&& data, const _Envs& envs, _UnaryOp op, const _CallEnv& call_env = {})
{
  mgmn::transform(data, data, envs, op, call_env);
}

/// @brief Unary transform into @p out (self-bound: environments of @p in).
_CCCL_TEMPLATE(class _SIn, class _SOut, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_UnaryOp>>))
_CCCL_HOST_API void transform(const _SIn& in, _SOut&& out, _UnaryOp op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(in);
  mgmn::transform(in, ::cuda::std::forward<_SOut>(out), envs, op, call_env);
}

/// @brief In-place unary transform (self-bound).
_CCCL_TEMPLATE(class _S, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_UnaryOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_UnaryOp>>))
_CCCL_HOST_API void transform(_S&& data, _UnaryOp op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  mgmn::transform(data, data, envs, op, call_env);
}
} // namespace mgmn
} // namespace cuda::experimental::sharded
