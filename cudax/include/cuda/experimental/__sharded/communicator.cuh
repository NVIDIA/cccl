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
 * @brief A places-backed model of the multi-GPU communicator concept, plus
 *        the adapter that lets sharded containers drive the MGMN algorithms.
 *
 * On the cooperation-scope ladder, the ranks rung (`__multi_gpu`) combines
 * through message passing because its ranks share nothing; the places rung
 * shares one virtual address space. A `places_communicator` treats each place
 * of a `place_group` as a rank of the `__multi_gpu` communicator concept, so
 * the MGMN constructs (`cuda::experimental::reduce`, `inclusive_scan`,
 * `sort`, ...) run unmodified over in-process places. Because every rank can
 * address every other rank's memory, the communicator verbs lower to plain
 * device-to-device copies and — for `all_reduce` — a single fold kernel that
 * reads every rank's partial in fixed rank order (bit-identical results run
 * to run for a fixed place list).
 *
 * The two tiers of the sharded design meet here: the container tier
 * (`place_group`, `sharded_array`) owns placement and resources; this header
 * manufactures what the engine tier consumes — communicators, environments
 * (stream + per-place memory resource, so engine temporaries land on the
 * place that runs the work), per-shard iterators and sizes. See
 * `bind_engine`.
 *
 * Two communicator variants are provided so both MGMN combine paths stay
 * exercised: `places_communicator` (with `all_reduce`, the direct combine)
 * and `basic_places_communicator` (without, selecting the
 * all_gather-plus-local-combine path). Conformance to the `__multi_gpu`
 * concepts is stated in code by the static_asserts at the end of this header
 * and verified end to end by the bridge tests.
 *
 * Scope and assumptions:
 *  - all places of one group must live on ONE device (the shared-VA
 *    transport); groups spanning devices belong to a multi-device transport
 *    tier and are refused with a diagnostic;
 *  - the stream a rank passes to a collective must belong to that rank's
 *    place context (streams drawn from the group's per-place pools satisfy
 *    this by construction);
 *  - one host thread drives all ranks of a group, exactly like the MGMN
 *    range algorithms do;
 *  - at most `comm_max_ranks` ranks per group.
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

#include <cuda/devices>
#include <cuda/std/functional>
#include <cuda/stream_ref>

// The concept layer of __multi_gpu is header-only and vendor-free; the
// NCCL-backed communicator is NOT included here.
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/// @brief Maximum ranks per communicator group (the fold-kernel argument
/// block is passed by value and fixed size).
inline constexpr int comm_max_ranks = 8;

/**
 * @brief A place-backed model of the "where does this run" pair the MGMN
 * algorithms program against: an execution context to activate and a device
 * to select resources on.
 *
 * Returned by `logical_device()` on the communicators below; the MGMN
 * algorithms use exactly two members, `.context()` and
 * `.underlying_device()`.
 */
class place_device_model
{
public:
  place_device_model() = default;
  place_device_model(CUcontext ctx, int devid)
      : ctx_(ctx)
      , devid_(devid)
  {}

  /// @brief Context making this place current.
  CUcontext context() const noexcept
  {
    return ctx_;
  }

  /// @brief Device the place lives on (used for resource selection).
  ::cuda::device_ref underlying_device() const noexcept
  {
    return ::cuda::device_ref{devid_};
  }

private:
  CUcontext ctx_{};
  int devid_ = -1;
};

namespace reserved
{
//! RAII: make a driver context current (push/pop).
class ctx_scope
{
public:
  explicit ctx_scope(CUcontext ctx)
  {
    cuda_safe_call(cuCtxPushCurrent(ctx));
  }
  ~ctx_scope()
  {
    CUcontext out{};
    cuCtxPopCurrent(&out); // best effort in a destructor
  }
  ctx_scope(const ctx_scope&)            = delete;
  ctx_scope& operator=(const ctx_scope&) = delete;
};

//! Per-rank immutable state + reusable sync events (created in the rank's
//! place context, so cuEventRecord's same-context requirement holds).
struct comm_rank_state
{
  exec_place place;
  CUcontext ctx = nullptr;
  int devid     = -1;
  place_device_model model;
  cudaEvent_t ev_pre  = nullptr; //!< fences this rank's stream into the combine
  cudaEvent_t ev_post = nullptr; //!< rank 0 only: broadcasts combine completion
};

struct comm_group_state;

//! One in-flight collective, rendezvous-batched across ranks.
//!
//! The `_v` variants carry per-rank count/displacement arrays as HOST
//! pointers, matching the MGMN call sites: the caller must have synced
//! whatever produced them before contributing, and the arrays must stay
//! valid until the size-th contribution flushes the batch (the values are
//! read on the host at enqueue time to size the copies).
struct pending_collective
{
  enum class op_kind
  {
    none,
    all_reduce,
    all_gather,
    all_gather_v,
    all_to_all,
    all_to_all_v
  };

  op_kind kind = op_kind::none;
  ::std::vector<const void*> send;
  ::std::vector<void*> recv;
  ::std::vector<cudaStream_t> stream;
  ::std::vector<char> present;
  size_t count     = 0;
  size_t elem_size = 0;
  int contributed  = 0;
  //! Typed combine launcher, set by the first all_reduce contribution
  //! (captures T and Op; the fold itself runs on device in fixed rank order).
  ::std::function<void(comm_group_state&, pending_collective&)> launch;
  //! Extras for the _v variants (host pointers, see above).
  ::std::vector<size_t> send_count; //!< all_gather_v: per-rank send count
  ::std::vector<const size_t*> send_counts; //!< all_to_all_v
  ::std::vector<const size_t*> send_displs; //!< all_to_all_v
  ::std::vector<const size_t*> recv_counts; //!< all_gather_v + all_to_all_v
  ::std::vector<const size_t*> recv_displs; //!< all_gather_v (displs) + all_to_all_v

  void reset(size_t nranks)
  {
    kind = op_kind::none;
    send.assign(nranks, nullptr);
    recv.assign(nranks, nullptr);
    stream.assign(nranks, nullptr);
    present.assign(nranks, 0);
    count       = 0;
    elem_size   = 0;
    contributed = 0;
    launch      = nullptr;
    send_count.assign(nranks, 0);
    send_counts.assign(nranks, nullptr);
    send_displs.assign(nranks, nullptr);
    recv_counts.assign(nranks, nullptr);
    recv_displs.assign(nranks, nullptr);
  }
};

//! A posted point-to-point half, waiting for its match.
struct pending_p2p
{
  const void* src = nullptr; //!< send side
  void* dst       = nullptr; //!< recv side
  size_t bytes    = 0;
  int rank        = -1; //!< the rank that posted this half
  int peer        = -1; //!< the rank it is addressed to / expected from
  cudaStream_t stream = nullptr;
};

//! State shared by all communicators of one group.
struct comm_group_state
{
  ::std::vector<comm_rank_state> ranks;
  pending_collective coll;
  ::std::vector<pending_p2p> posted_sends;
  ::std::vector<pending_p2p> posted_recvs;

  ~comm_group_state()
  {
    for (auto& r : ranks)
    {
      if (r.ctx != nullptr)
      {
        CUcontext out{};
        if (cuCtxPushCurrent(r.ctx) == CUDA_SUCCESS)
        {
          if (r.ev_pre)
          {
            cudaEventDestroy(r.ev_pre);
          }
          if (r.ev_post)
          {
            cudaEventDestroy(r.ev_post);
          }
          cuCtxPopCurrent(&out);
        }
      }
    }
  }
};

//! Fold-kernel argument block (passed by value; nranks <= comm_max_ranks).
template <typename _Tp>
struct fold_args
{
  const _Tp* send[comm_max_ranks];
  _Tp* recv[comm_max_ranks];
};

//! The shared-VA all_reduce: ONE kernel reads every rank's partial and
//! writes every rank's result. The fold is sequential in rank order
//! (0, 1, ..., n-1), so for a fixed rank-to-place mapping the result is
//! BIT-IDENTICAL across runs.
template <typename _Tp, typename _Op>
__global__ void fold_kernel(fold_args<_Tp> a, int nranks, size_t count, _Op op)
{
  size_t i            = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  const size_t stride = gridDim.x * size_t(blockDim.x);
  for (; i < count; i += stride)
  {
    _Tp acc = a.send[0][i];
    for (int r = 1; r < nranks; ++r)
    {
      acc = op(acc, a.send[r][i]); // fixed order: deterministic
    }
    for (int r = 0; r < nranks; ++r)
    {
      a.recv[r][i] = acc;
    }
  }
}

//! Record one rank's contribution to the current batched collective.
inline void contribute(
  comm_group_state& st,
  pending_collective::op_kind kind,
  int rank,
  const void* send,
  void* recv,
  size_t count,
  size_t elem_size,
  cudaStream_t stream)
{
  auto& c           = st.coll;
  const auto nranks = st.ranks.size();
  if (c.kind == pending_collective::op_kind::none)
  {
    c.reset(nranks);
    c.kind      = kind;
    c.count     = count;
    c.elem_size = elem_size;
  }
  if (c.kind != kind || c.count != count || c.elem_size != elem_size)
  {
    _CCCL_THROW(::std::runtime_error, "sharded places_communicator: mismatched collective contributions in one group");
  }
  if (rank < 0 || static_cast<size_t>(rank) >= nranks || c.present[rank])
  {
    _CCCL_THROW(::std::runtime_error, "sharded places_communicator: duplicate or out-of-range rank contribution");
  }
  c.send[rank]    = send;
  c.recv[rank]    = recv;
  c.stream[rank]  = stream;
  c.present[rank] = 1;
  ++c.contributed;
}

//! Flush a complete collective batch:
//!   1. fence every rank's stream into rank 0's stream (events),
//!   2. combine on rank 0's stream (fold kernel / gather copies),
//!   3. broadcast a completion event back to every rank's stream.
inline void flush_collective(comm_group_state& st)
{
  auto& c     = st.coll;
  const int n = static_cast<int>(st.ranks.size());

  // 1. fence: rank r's stream -> event -> rank 0's stream waits
  for (int r = 1; r < n; ++r)
  {
    ctx_scope g{st.ranks[r].ctx};
    cuda_safe_call(cudaEventRecord(st.ranks[r].ev_pre, c.stream[r]));
  }
  {
    ctx_scope g{st.ranks[0].ctx};
    for (int r = 1; r < n; ++r)
    {
      cuda_safe_call(cudaStreamWaitEvent(c.stream[0], st.ranks[r].ev_pre, 0));
    }

    // 2. combine on rank 0's stream, in fixed rank order. Everything below
    // the all_reduce fold is plain device-to-device copies: on the
    // shared-address rung, exchange lowers to addressing + memcpy, no
    // transport. Copies to self (src == dst) are skipped.
    switch (c.kind)
    {
      case pending_collective::op_kind::all_reduce:
        c.launch(st, c);
        break;
      case pending_collective::op_kind::all_gather:
        // Every rank's slot to every rank's buffer: recv[j] + i*count <- send[i].
        for (int j = 0; j < n; ++j)
        {
          for (int i = 0; i < n; ++i)
          {
            char* dst       = static_cast<char*>(c.recv[j]) + static_cast<size_t>(i) * c.count * c.elem_size;
            const char* src = static_cast<const char*>(c.send[i]);
            if (dst == src)
            {
              continue; // in-place own slot: pure addressing
            }
            cuda_safe_call(cudaMemcpyAsync(dst, src, c.count * c.elem_size, cudaMemcpyDeviceToDevice, c.stream[0]));
          }
        }
        break;
      case pending_collective::op_kind::all_gather_v:
        // Rank i's block (send_count[i] elements) lands at recv[j] + displs[i]
        // on every rank j; counts/displs are receiver-side HOST arrays.
        for (int j = 0; j < n; ++j)
        {
          for (int i = 0; i < n; ++i)
          {
            const size_t cnt = c.recv_counts[j][i];
            if (cnt != c.send_count[i])
            {
              _CCCL_THROW(::std::runtime_error,
                          "sharded places_communicator: all_gather_v recv count does not match the "
                          "sender's send count");
            }
            char* dst       = static_cast<char*>(c.recv[j]) + c.recv_displs[j][i] * c.elem_size;
            const char* src = static_cast<const char*>(c.send[i]);
            if (cnt == 0 || dst == src)
            {
              continue;
            }
            cuda_safe_call(cudaMemcpyAsync(dst, src, cnt * c.elem_size, cudaMemcpyDeviceToDevice, c.stream[0]));
          }
        }
        break;
      case pending_collective::op_kind::all_to_all:
        // recv[i] block j <- send[j] block i (count elements per block).
        for (int i = 0; i < n; ++i)
        {
          for (int j = 0; j < n; ++j)
          {
            char* dst       = static_cast<char*>(c.recv[i]) + static_cast<size_t>(j) * c.count * c.elem_size;
            const char* src = static_cast<const char*>(c.send[j]) + static_cast<size_t>(i) * c.count * c.elem_size;
            if (dst == src)
            {
              continue;
            }
            cuda_safe_call(cudaMemcpyAsync(dst, src, c.count * c.elem_size, cudaMemcpyDeviceToDevice, c.stream[0]));
          }
        }
        break;
      case pending_collective::op_kind::all_to_all_v:
        // recv[i] + recv_displs[i][j] <- send[j] + send_displs[j][i],
        // send_counts[j][i] elements; counts/displs are HOST arrays.
        for (int i = 0; i < n; ++i)
        {
          for (int j = 0; j < n; ++j)
          {
            const size_t cnt = c.send_counts[j][i];
            if (c.recv_counts[i][j] != cnt)
            {
              _CCCL_THROW(::std::runtime_error,
                          "sharded places_communicator: all_to_all_v recv count does not match the "
                          "sender's send count");
            }
            char* dst       = static_cast<char*>(c.recv[i]) + c.recv_displs[i][j] * c.elem_size;
            const char* src = static_cast<const char*>(c.send[j]) + c.send_displs[j][i] * c.elem_size;
            if (cnt == 0 || dst == src)
            {
              continue;
            }
            cuda_safe_call(cudaMemcpyAsync(dst, src, cnt * c.elem_size, cudaMemcpyDeviceToDevice, c.stream[0]));
          }
        }
        break;
      case pending_collective::op_kind::none:
        return;
    }

    // 3. broadcast completion
    cuda_safe_call(cudaEventRecord(st.ranks[0].ev_post, c.stream[0]));
  }
  for (int r = 1; r < n; ++r)
  {
    ctx_scope g{st.ranks[r].ctx};
    cuda_safe_call(cudaStreamWaitEvent(c.stream[r], st.ranks[0].ev_post, 0));
  }

  c.reset(st.ranks.size());
}

//! Try to match a posted send with a posted recv; on a match, run the copy:
//! sender stream -> event -> receiver stream waits -> cudaMemcpyAsync on the
//! receiver stream -> event -> sender stream waits (both sides complete).
inline void match_p2p(comm_group_state& st)
{
  for (auto s = st.posted_sends.begin(); s != st.posted_sends.end();)
  {
    bool matched = false;
    for (auto r = st.posted_recvs.begin(); r != st.posted_recvs.end(); ++r)
    {
      if (s->peer == r->rank && r->peer == s->rank && s->bytes == r->bytes)
      {
        cudaEvent_t ev_send{}, ev_copy{};
        {
          ctx_scope g{st.ranks[s->rank].ctx};
          cuda_safe_call(cudaEventCreateWithFlags(&ev_send, cudaEventDisableTiming));
          cuda_safe_call(cudaEventRecord(ev_send, s->stream));
        }
        {
          ctx_scope g{st.ranks[r->rank].ctx};
          cuda_safe_call(cudaStreamWaitEvent(r->stream, ev_send, 0));
          cuda_safe_call(cudaMemcpyAsync(r->dst, s->src, s->bytes, cudaMemcpyDeviceToDevice, r->stream));
          cuda_safe_call(cudaEventCreateWithFlags(&ev_copy, cudaEventDisableTiming));
          cuda_safe_call(cudaEventRecord(ev_copy, r->stream));
        }
        {
          ctx_scope g{st.ranks[s->rank].ctx};
          cuda_safe_call(cudaStreamWaitEvent(s->stream, ev_copy, 0));
        }
        // Destruction is deferred by CUDA until the recorded work completes.
        {
          ctx_scope g{st.ranks[s->rank].ctx};
          cudaEventDestroy(ev_send);
        }
        {
          ctx_scope g{st.ranks[r->rank].ctx};
          cudaEventDestroy(ev_copy);
        }
        st.posted_recvs.erase(r);
        matched = true;
        break;
      }
    }
    s = matched ? st.posted_sends.erase(s) : s + 1;
  }
}
} // namespace reserved

/**
 * @brief Group-semantics guard: collective calls made while a guard is alive
 * are batched per group; the batch flushes when every rank has contributed
 * (rendezvous). Destruction verifies nothing partial was left behind.
 *
 * Flushing at rendezvous rather than at guard destruction is semantically
 * equivalent for stream-ordered operations and keeps the guard trivially
 * destructible; the destructor only verifies no PARTIAL batch remains (a
 * missing-rank protocol error).
 */
class comm_group_guard
{
public:
  explicit comm_group_guard(::std::shared_ptr<reserved::comm_group_state> st)
      : state_(::std::move(st))
  {}

  comm_group_guard(const comm_group_guard&)            = delete;
  comm_group_guard& operator=(const comm_group_guard&) = delete;
  comm_group_guard(comm_group_guard&&)                 = default;

  ~comm_group_guard()
  {
    if (state_ == nullptr)
    {
      return;
    }
    if (state_->coll.contributed != 0)
    {
      ::std::fprintf(stderr,
                     "sharded places_communicator: group guard destroyed with a PARTIAL collective "
                     "(%d/%zu contributions) -- a rank never called in; aborting\n",
                     state_->coll.contributed,
                     state_->ranks.size());
      ::std::abort();
    }
    if (!state_->posted_sends.empty() || !state_->posted_recvs.empty())
    {
      ::std::fprintf(stderr,
                     "sharded places_communicator: group guard destroyed with unmatched send/recv "
                     "(%zu sends, %zu recvs) -- aborting\n",
                     state_->posted_sends.size(),
                     state_->posted_recvs.size());
      ::std::abort();
    }
  }

private:
  ::std::shared_ptr<reserved::comm_group_state> state_;
};

/**
 * @brief Places communicator WITHOUT `all_reduce`: models the core
 * `__multi_gpu` communicator concept (native_handle, rank, size, group_guard,
 * send, recv) plus `all_gather[_v]` and `all_to_all[_v]`.
 *
 * Feeding this variant to the MGMN algorithms selects their
 * all_gather-plus-local-combine path, keeping both combine paths testable.
 */
class basic_places_communicator
{
public:
  using native_handle_type = exec_place;
  using group_guard_type   = comm_group_guard;

  basic_places_communicator(::std::shared_ptr<reserved::comm_group_state> st, int rank)
      : state_(::std::move(st))
      , rank_(rank)
  {}

  /// @brief The native handle of a places rank IS the place.
  native_handle_type native_handle() const noexcept
  {
    return state_->ranks[rank_].place;
  }

  ::std::int32_t rank() const noexcept
  {
    return rank_;
  }

  ::std::int32_t size() const noexcept
  {
    return static_cast<::std::int32_t>(state_->ranks.size());
  }

  group_guard_type group_guard() const
  {
    return group_guard_type{state_};
  }

  /// @brief The place-backed model of {context(), underlying_device()} the
  /// MGMN algorithms activate and select resources through.
  const place_device_model& logical_device() const noexcept
  {
    return state_->ranks[rank_].model;
  }

  /**
   * @brief Point-to-point send: rendezvous with the matching recv, then one
   * device-to-device copy over the shared address space. `nbytes` is a byte
   * count (the concept's point-to-point buffers are untyped).
   */
  void send(group_guard_type&, const void* buf, size_t nbytes, ::std::int32_t peer, ::cuda::stream_ref stream) const
  {
    state_->posted_sends.push_back(reserved::pending_p2p{buf, nullptr, nbytes, rank_, peer, stream.get()});
    reserved::match_p2p(*state_);
  }

  void recv(group_guard_type&, void* buf, size_t nbytes, ::std::int32_t peer, ::cuda::stream_ref stream) const
  {
    state_->posted_recvs.push_back(reserved::pending_p2p{nullptr, buf, nbytes, rank_, peer, stream.get()});
    reserved::match_p2p(*state_);
  }

  /// @brief all_gather: every rank's `count` elements become addressable in
  /// every rank's receive buffer (device-to-device copies).
  template <typename _Tp>
  void all_gather(group_guard_type&, const _Tp* sendbuff, _Tp* recvbuff, size_t count, ::cuda::stream_ref stream) const
  {
    reserved::contribute(
      *state_,
      reserved::pending_collective::op_kind::all_gather,
      rank_,
      sendbuff,
      recvbuff,
      count,
      sizeof(_Tp),
      stream.get());
    if (state_->coll.contributed == size())
    {
      reserved::flush_collective(*state_);
    }
  }

  /**
   * @brief all_gather_v: variable-count gather. `recv_counts`/`displs` are
   * HOST arrays of `size()` entries (matching the MGMN call sites); they are
   * read at flush time to size the copies and must stay valid until every
   * rank has contributed.
   */
  template <typename _Tp>
  void all_gather_v(
    group_guard_type&,
    const _Tp* sendbuff,
    size_t send_count,
    _Tp* recvbuff,
    const size_t* recv_counts,
    const size_t* displs,
    ::cuda::stream_ref stream) const
  {
    reserved::contribute(
      *state_,
      reserved::pending_collective::op_kind::all_gather_v,
      rank_,
      sendbuff,
      recvbuff,
      /*count=*/0,
      sizeof(_Tp),
      stream.get());
    auto& c              = state_->coll;
    c.send_count[rank_]  = send_count;
    c.recv_counts[rank_] = recv_counts;
    c.recv_displs[rank_] = displs;
    if (c.contributed == size())
    {
      reserved::flush_collective(*state_);
    }
  }

  /// @brief all_to_all: rank i's receive block j is rank j's send block i
  /// (`count` elements per block).
  template <typename _Tp>
  void all_to_all(group_guard_type&, const _Tp* sendbuff, _Tp* recvbuff, size_t count, ::cuda::stream_ref stream) const
  {
    reserved::contribute(
      *state_,
      reserved::pending_collective::op_kind::all_to_all,
      rank_,
      sendbuff,
      recvbuff,
      count,
      sizeof(_Tp),
      stream.get());
    if (state_->coll.contributed == size())
    {
      reserved::flush_collective(*state_);
    }
  }

  /**
   * @brief all_to_all_v: the variable-count exchange. All four
   * count/displacement arrays are HOST arrays of `size()` entries, read at
   * flush time; same validity rules as `all_gather_v`.
   */
  template <typename _Tp>
  void all_to_all_v(
    group_guard_type&,
    const _Tp* sendbuff,
    const size_t* send_counts,
    const size_t* send_displs,
    _Tp* recvbuff,
    const size_t* recv_counts,
    const size_t* recv_displs,
    ::cuda::stream_ref stream) const
  {
    reserved::contribute(
      *state_,
      reserved::pending_collective::op_kind::all_to_all_v,
      rank_,
      sendbuff,
      recvbuff,
      /*count=*/0,
      sizeof(_Tp),
      stream.get());
    auto& c              = state_->coll;
    c.send_counts[rank_] = send_counts;
    c.send_displs[rank_] = send_displs;
    c.recv_counts[rank_] = recv_counts;
    c.recv_displs[rank_] = recv_displs;
    if (c.contributed == size())
    {
      reserved::flush_collective(*state_);
    }
  }

protected:
  ::std::shared_ptr<reserved::comm_group_state> state_;
  ::std::int32_t rank_ = -1;
};

/**
 * @brief Full places communicator: adds the shared-VA `all_reduce`, so the
 * MGMN algorithms take their direct combine path.
 */
class places_communicator : public basic_places_communicator
{
public:
  using basic_places_communicator::basic_places_communicator;

  /// @brief all_reduce with a FIXED fold order (bit-identical results run to
  /// run; see `reserved::fold_kernel`).
  template <typename _Tp, typename _Op>
  void all_reduce(
    group_guard_type&, const _Tp* sendbuff, _Tp* recvbuff, size_t count, _Op op, ::cuda::stream_ref stream) const
  {
    reserved::contribute(
      *state_,
      reserved::pending_collective::op_kind::all_reduce,
      rank_,
      sendbuff,
      recvbuff,
      count,
      sizeof(_Tp),
      stream.get());
    auto& c = state_->coll;
    if (!c.launch)
    {
      c.launch = [op](reserved::comm_group_state& st, reserved::pending_collective& cc) {
        reserved::fold_args<_Tp> a{};
        const int n = static_cast<int>(st.ranks.size());
        for (int r = 0; r < n; ++r)
        {
          a.send[r] = static_cast<const _Tp*>(cc.send[r]);
          a.recv[r] = static_cast<_Tp*>(cc.recv[r]);
        }
        const int threads = 256;
        const int blocks  = static_cast<int>(::std::min<size_t>((cc.count + threads - 1) / threads, size_t{1024}));
        // rank 0's context is current here (flush_collective holds it)
        reserved::fold_kernel<_Tp, _Op><<<blocks, threads, 0, cc.stream[0]>>>(a, n, cc.count, op);
        cuda_safe_call(cudaGetLastError());
      };
    }
    if (c.contributed == size())
    {
      reserved::flush_collective(*state_);
    }
  }
};

/**
 * @brief Build one communicator per place of an explicit place list, all
 * sharing one group state. Rank i = position i in the list; size = list
 * length.
 *
 * @throws std::invalid_argument on an empty or too-long place list, or when
 *         the places span more than one device (the shared-VA transport
 *         requires one device; multi-device groups belong to a hierarchical
 *         transport tier that composes this rung with the ranks rung).
 */
template <class _Comm = places_communicator>
::std::vector<_Comm> make_communicators(::std::vector<exec_place> places)
{
  if (places.empty() || places.size() > static_cast<size_t>(comm_max_ranks))
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::make_communicators: need 1.." + ::std::to_string(comm_max_ranks) + " places");
  }

  auto st = ::std::make_shared<reserved::comm_group_state>();
  st->ranks.reserve(places.size());
  for (auto& p : places)
  {
    reserved::comm_rank_state r;
    r.place = p;
    {
      exec_place_scope scope(p);
      // Force runtime-context binding for device-rung places (green-context
      // and locality-domain places set a driver context directly; this is a
      // no-op there).
      cuda_safe_call(cudaFree(nullptr));
      cuda_safe_call(cuCtxGetCurrent(&r.ctx));
      CUdevice dev{};
      cuda_safe_call(cuCtxGetDevice(&dev));
      r.devid = static_cast<int>(dev);
      cuda_safe_call(cudaEventCreateWithFlags(&r.ev_pre, cudaEventDisableTiming));
      cuda_safe_call(cudaEventCreateWithFlags(&r.ev_post, cudaEventDisableTiming));
    }
    r.model = place_device_model{r.ctx, r.devid};
    st->ranks.push_back(::std::move(r));
  }

  for (const auto& r : st->ranks)
  {
    if (r.devid != st->ranks.front().devid)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded::make_communicators: places span devices " + ::std::to_string(st->ranks.front().devid)
                    + " and " + ::std::to_string(r.devid)
                    + "; the shared-address transport requires one device. Cross-device groups belong to a "
                      "hierarchical (places x ranks) transport tier -- not implemented.");
    }
  }
  st->coll.reset(st->ranks.size());

  ::std::vector<_Comm> comms;
  comms.reserve(places.size());
  for (size_t i = 0; i < places.size(); ++i)
  {
    comms.emplace_back(st, static_cast<int>(i));
  }
  return comms;
}

/**
 * @brief Build one communicator per place of a `place_group`: rank i = place
 * index i, size = `group.size()`.
 *
 * The group names WHERE (its places); the communicators supply HOW TO
 * COOPERATE at that rung, in the vocabulary the MGMN algorithms consume.
 */
template <class _Comm = places_communicator>
::std::vector<_Comm> make_communicators(const place_group& group)
{
  return make_communicators<_Comm>(group.places());
}

// ============================================================================
// Concept conformance, stated in code
// ============================================================================

static_assert(::cuda::experimental::__communicator<places_communicator>,
              "places_communicator must model the multi-GPU communicator concept");
static_assert(::cuda::experimental::__communicator<basic_places_communicator>,
              "basic_places_communicator must model the multi-GPU communicator concept");
static_assert(::cuda::experimental::__has_all_reduce<places_communicator, float*, ::cuda::std::plus<>>,
              "places_communicator must expose all_reduce (direct combine path)");
static_assert(!::cuda::experimental::__has_all_reduce<basic_places_communicator, float*, ::cuda::std::plus<>>,
              "basic_places_communicator must NOT expose all_reduce (gather combine path)");
static_assert(::cuda::experimental::__has_all_gather<basic_places_communicator, float*>,
              "basic_places_communicator must expose all_gather");
static_assert(::cuda::experimental::__has_all_gather_v<places_communicator, float*>,
              "places_communicator must expose all_gather_v");
static_assert(::cuda::experimental::__has_all_to_all<places_communicator, float*>,
              "places_communicator must expose all_to_all");
static_assert(::cuda::experimental::__has_all_to_all_v<places_communicator, float*>,
              "places_communicator must expose all_to_all_v");

// ============================================================================
// Engine bindings: what the container tier manufactures for the engine tier
// ============================================================================

/// @brief Environment type produced by `place_group::env` (stream + per-place
/// memory resource), consumable by the MGMN algorithms and by CUB.
using place_env_type = decltype(place_group::env(::cuda::std::declval<const data_place&>(), cudaStream_t{}));

/**
 * @brief The ranges an MGMN algorithm consumes, manufactured from a sharded
 * container: one communicator, environment, iterator and size per shard.
 *
 * All four members are index-aligned with the shards (and with the places of
 * the group the bindings were built from).
 */
template <typename _Tp, class _Comm = places_communicator>
struct engine_bindings
{
  ::std::vector<_Comm> comms; //!< one communicator per shard (rank = shard index)
  ::std::vector<place_env_type> envs; //!< stream + memory resource per shard
  ::std::vector<_Tp*> shard_data; //!< per-shard iterators
  ::std::vector<size_t> shard_sizes; //!< per-shard element counts
};

/**
 * @brief Manufacture the MGMN-facing ranges from a sharded array and its
 * place group: the two-tier seam.
 *
 * The container tier owns placement and resources; the engine tier owns
 * cross-place choreography. This adapter is the boundary: each shard becomes
 * one rank, its data pointer the rank's iterator, its reference stream and
 * its place's memory resource the rank's environment (so engine temporaries
 * are placed where the rank's work runs).
 *
 * @code
 * auto b = bind_engine(group, data);
 * cuda::experimental::reduce(
 *   cuda::experimental::broadcasted, b.comms, b.envs, b.shard_data, b.shard_sizes, out_iters);
 * @endcode
 *
 * @throws std::invalid_argument when the array does not have one shard per
 *         group place, or via `make_communicators` (see there).
 */
template <class _Comm = places_communicator, typename _Tp>
engine_bindings<_Tp, _Comm> bind_engine(place_group& group, sharded_array<_Tp>& data)
{
  check_places(data, group, "sharded::bind_engine");

  engine_bindings<_Tp, _Comm> b;
  b.comms = make_communicators<_Comm>(group);
  b.envs.reserve(data.num_shards());
  b.shard_data.reserve(data.num_shards());
  b.shard_sizes.reserve(data.num_shards());
  for (size_t i = 0; i < data.num_shards(); ++i)
  {
    auto& s                   = data.shard(i);
    const cudaStream_t stream = s.stream ? s.stream : group.get_stream(i);
    b.envs.push_back(place_group::env(s.place, stream));
    b.shard_data.push_back(s.data);
    b.shard_sizes.push_back(s.size);
  }
  return b;
}
} // namespace cuda::experimental::sharded
