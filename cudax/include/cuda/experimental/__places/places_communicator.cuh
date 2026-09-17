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
 * @brief `places_communicator`: a communicator (in the sense of the MGMN
 *        `__communicator` concept of `cuda/experimental/__multi_gpu/`) whose
 *        P ranks are P places of one process and whose "network" is the
 *        shared address space.
 *
 * Point-to-point and collective operations become `cudaMemcpyAsync` / tiny
 * kernels ordered by events between the ranks' streams. Modeled on NCCL
 * groups: operations issued while a group guard is alive are RECORDED and
 * EXECUTED when the outermost guard is destroyed (group end), which is what
 * lets a single host loop issue each rank's half of a send/recv pair or of a
 * collective one after the other. Everything the communicator enqueues is
 * capture-legal (event record / stream wait / memcpy / kernel launch; no
 * host synchronization, no allocation).
 *
 * The events a group records are pooled per rank and live as long as the
 * group state does: a `place_group` owns one group per lane
 * (`place_group::communicators`), so the algorithms driven through it pay
 * for event creation once, not per call. This header depends only on the
 * places vocabulary and the vendor-free MGMN concepts, so `place_group.cuh`
 * can include it.
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

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__places/stream_pool.cuh> // get_device_from_stream
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
namespace reserved
{
//! @brief Maximum rank count of one communicator group (the by-value slot
//! array of the all-reduce kernel).
inline constexpr unsigned __max_fold_shards = 64;

//! @brief The P source pointers of an all-reduce, passed to the fold kernel
//! by value (one per rank).
template <typename _Tp>
struct __partial_slots
{
  const _Tp* __p[__max_fold_shards];
};

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

//! @brief RAII device scope derived from a stream: the stream's device is
//! current for the scope's lifetime (a launch needs the current device to
//! match the stream's), the previous device restored on exit.
class __stream_device_scope
{
public:
  explicit __stream_device_scope(cudaStream_t __stream)
      : __prev_(::cuda::experimental::stf::cuda_try<cudaGetDevice>())
  {
    const int __target = get_device_from_stream(__stream);
    if (__target != __prev_)
    {
      ::cuda::experimental::stf::cuda_safe_call(cudaSetDevice(__target));
      __switched_ = true;
    }
  }

  __stream_device_scope(const __stream_device_scope&)            = delete;
  __stream_device_scope& operator=(const __stream_device_scope&) = delete;

  ~__stream_device_scope()
  {
    if (__switched_)
    {
      (void) cudaSetDevice(__prev_);
    }
  }

private:
  int __prev_;
  bool __switched_ = false;
};
} // namespace reserved

/**
 * @brief A communicator over the P places of one process, satisfying the
 * MGMN `__communicator` concept (with `all_gather` and `all_reduce`).
 *
 * The P rank handles of one communicator group are values sharing one
 * state (`create(streams)` returns all P at once; copies of a handle share
 * it too). Operations take a group guard obtained from any rank
 * (`group_guard()`) and are recorded; the outermost guard's destruction
 * executes them:
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
 * device of the stream `create` was given for r): events are pooled per
 * rank, created on first use and kept for the life of the group state, and
 * recorded on those streams. Only stream operations are enqueued — event
 * record, stream wait, memcpy, kernel launch — so a group issued inside a
 * CUDA graph capture captures. Not thread-safe: one group is one host
 * thread's sequence (a `place_group` keeps one group per lane, which is
 * exactly the isolation two concurrent lanes need).
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
   * @throws std::invalid_argument on more than 64 ranks.
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
      __st->__devices.push_back(get_device_from_stream(__s.get()));
    }
    // The per-rank event pools start empty and grow on demand (two events
    // per rank per collective: source-ready, destination-done); event
    // creation is capture-legal, and a group that issues no collective —
    // a rank-local algorithm such as `transform` — then creates none.
    __st->__events.resize(__n);
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
      ::cuda::experimental::stf::cuda_safe_call(cudaGetDevice(&__prev));
      if (__prev != __device)
      {
        ::cuda::experimental::stf::cuda_safe_call(cudaSetDevice(__device));
      }
      cudaEvent_t __ev           = nullptr;
      const cudaError_t __status = cudaEventCreateWithFlags(&__ev, cudaEventDisableTiming);
      if (__prev != __device)
      {
        ::cuda::experimental::stf::cuda_safe_call(cudaSetDevice(__prev));
      }
      ::cuda::experimental::stf::cuda_safe_call(__status);
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
      ::cuda::experimental::stf::cuda_safe_call(cudaEventRecord(__ev, __stream));
      return __ev;
    }

    static void __wait(cudaStream_t __stream, cudaEvent_t __ev)
    {
      ::cuda::experimental::stf::cuda_safe_call(cudaStreamWaitEvent(__stream, __ev, 0));
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
        ::cuda::experimental::stf::cuda_safe_call(
          cudaMemcpyAsync(__dst, __src, __bytes, cudaMemcpyDefault, __dst_stream));
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
          ::cuda::experimental::stf::cuda_safe_call(
            cudaMemcpyAsync(__to, __c.__slots[__s].__send, __bytes, cudaMemcpyDefault, __dst.__stream));
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
          reserved::__stream_device_scope __scope(__dst.__stream);
          reserved::__mgmn_all_reduce_kernel<_Tp, _ReduceOp><<<__blocks, __threads, 0, __dst.__stream>>>(
            __ptrs, static_cast<unsigned>(__n), __count, __op, static_cast<_Tp*>(__dst.__recv));
          ::cuda::experimental::stf::cuda_safe_call(cudaGetLastError());
        }
      }
      __fan_out(__c);
    }
  };

  ::std::shared_ptr<__state> __state_;
  ::cuda::std::int32_t __rank_ = 0;
};

static_assert(::cuda::experimental::mgmn::__communicator<places_communicator>);
static_assert(::cuda::experimental::mgmn::__has_all_gather<places_communicator>);
static_assert(::cuda::experimental::mgmn::__has_all_reduce<places_communicator>);
} // namespace cuda::experimental::places
