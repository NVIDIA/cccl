//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_LANE_SCHEDULER
#define __CUDAX_EXECUTION_LANE_SCHEDULER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file lane_scheduler.cuh
//! @brief `lane_scheduler`: a "stream first" scheduler for stream-ordered work.
//!
//! A `lane_scheduler` stores one `cudaStream_t` and nothing else. Its `schedule()`
//! sender completes on the host before `start()` returns; `then` bodies run on
//! the host and enqueue work (kernel launches, CUB calls with an env carrying
//! the stream, ...) onto the lane's stream. `set_value` therefore means
//! "stream-ordered and enqueued", not "finished" -- the convention stream-ordered
//! CUDA code already follows. Because the P2300 identity is host-side, the
//! generic `then`, `when_all`, `let_value`, ... adaptors apply unmodified.
//!
//! The scheduler has its own execution domain, `lane_domain`, which customizes
//! exactly two algorithms through the standard hooks:
//!
//!  * `continues_on(sndr, lane)`: when the predecessor completes, record one event
//!    on every upstream stream that differs from the target lane's stream and make
//!    the target wait on it. Same stream: no event. Upstream streams are found by
//!    walking the sender tree (`when_all` children, `then` chains) at compile time,
//!    so `when_all(a, b) | continues_on(a)` issues exactly one event (b -> a),
//!    lazily, at the point where the continuation moves onto a stream. Under
//!    stream capture the join becomes a graph edge.
//!  * `sync_wait(sndr)`: the generic host completion, then `cudaStreamSynchronize`
//!    on every lane the sender completes on.
//!
//! Each join is the usual non-blocking stream-to-stream dependency
//! (`cuda::stream_ref::wait(stream_ref)`): a timing-disabled event created in the
//! upstream stream's context, recorded there, waited on by the target, and
//! destroyed right away -- `cudaStreamWaitEvent` captures the event's state at
//! call time, and the runtime defers the release. No shared state, and correct
//! when lanes live on different devices.
//!
//! The receiver's environment may carry a `get_lane_join_observer` query: a
//! callable invoked as `observer(from_stream, to_stream)` once per event join
//! issued. It is a forwarding query, so `sndr | write_env(env{prop{
//! get_lane_join_observer, fn}})` or `sync_wait(sndr, env)` reaches every join in
//! the chain. Tests use it to assert how many events a composition issues.
//!
//! `when_all` over two different lanes has no completion scheduler by design: a
//! continuation after it must `continues_on(some_lane)` before enqueuing
//! stream work.
//!
//! The completion behaviour is reported as `synchronous` (completes before
//! `start()` returns) rather than `inline_completion`: the latter would make a
//! `when_all` of lanes inherit the environment's scheduler (e.g. `sync_wait`'s
//! run loop) as its completion scheduler, whose domain is not `lane_domain`.
//! No call in this header blocks the host except `sync_wait`.

#include <cuda/__utility/immovable.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/completion_behavior.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__execution/when_all.cuh>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! @brief Environment query for an optional observer of lane event joins.
//!
//! If the receiver's environment answers this query, the result is invoked as
//! `observer(cudaStream_t from, cudaStream_t to)` each time `continues_on` onto a
//! `lane_scheduler` records an event on `from` and makes `to` wait on it.
struct get_lane_join_observer_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_lane_join_observer_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_lane_join_observer_t>
  {
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};
_CCCL_GLOBAL_CONSTANT get_lane_join_observer_t get_lane_join_observer{};

//! @brief The fork point of the enclosing `when_all`: an event recorded on the
//! lane the `when_all` started on (`origin`), before any child was started.
//!
//! A `when_all` under a lane records it once and hands it to its children through
//! their environment (this is a forwarding query). A child that begins on another
//! lane -- `schedule(lane)`, or `schedule(origin) | continues_on(lane)` -- makes
//! that lane wait on this event instead of on the origin's tail at the time the
//! child happens to start, which would include the siblings' work.
struct lane_fork_point
{
  cudaStream_t origin{nullptr};
  cudaEvent_t event{nullptr};
};

struct get_lane_fork_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_lane_fork_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_lane_fork_t>
  {
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};
_CCCL_GLOBAL_CONSTANT get_lane_fork_t get_lane_fork{};

//! @brief What a `lane_split` shares with the join machinery: the event recorded
//! at its completion point (the ready point), its lane, and the lanes that
//! waited on the ready point (its consumers). Before the split's state -- and
//! the values it holds -- dies, those lanes are joined back into the split's
//! lane, so anything the values release (a scoped buffer's free) is ordered
//! after every consumer, in eager runs and under capture alike.
struct lane_split_handle
{
  cudaEvent_t ready{nullptr};
  cudaStream_t lane{nullptr};
  static constexpr int cap = 16;
  cudaStream_t consumers[cap]{};
  int n_consumers = 0;
  void add_consumer(cudaStream_t s) noexcept
  {
    for (int i = 0; i < n_consumers; ++i)
    {
      if (consumers[i] == s)
      {
        return;
      }
    }
    if (n_consumers < cap)
    {
      consumers[n_consumers++] = s;
    }
  }
};

//! @brief Attribute of a `lane_split` sender: its handle (ready point, lane,
//! consumers). A `continues_on` whose upstream is a split consumer waits on the
//! ready point instead of on the lane's tail, and registers as a consumer.
struct get_lane_ready_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_lane_ready_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_lane_ready_t>
  {
    return __env.query(*this);
  }
};
_CCCL_GLOBAL_CONSTANT get_lane_ready_t get_lane_ready{};

//! @brief The set of lanes a pipeline touched, gathered as it runs.
//!
//! The lanes a pipeline uses cannot be read off the sender: `let_value` bodies
//! exist only at run time. So a `lane_tracker` is handed down through the
//! environment (`get_lane_tracker`, a forwarding query) and every lane node
//! registers its lane when it executes: `schedule(lane)` on start,
//! `continues_on(lane)` on completion, a split's child through the same nodes.
//! Once the pipeline has run (and its operation state has died, so that scoped
//! frees are enqueued too), the set is what the terminal has to wait on:
//! `sync_wait` synchronizes each lane, `lane_join_into` makes a stream wait on
//! each lane, `lane_capture` joins each lane back into the capturing stream.
struct lane_tracker
{
  static constexpr int cap = 32;
  cudaStream_t s[cap]{};
  int n = 0;
  void add(cudaStream_t x) noexcept
  {
    for (int i = 0; i < n; ++i)
    {
      if (s[i] == x)
      {
        return;
      }
    }
    if (n < cap)
    {
      s[n++] = x;
    }
  }
};

struct get_lane_tracker_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_lane_tracker_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_lane_tracker_t>
  {
    return __env.query(*this);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};
_CCCL_GLOBAL_CONSTANT get_lane_tracker_t get_lane_tracker{};

namespace __lane
{
// ------------------------------------------------------- stream set ----------
// The lanes a sender completes on. For each lane, optionally the event recorded
// at the point the sender's work on that lane completed (a `lane_split`'s ready
// point); a join waits on that event when present, else on the lane's tail.
struct stream_set
{
  static constexpr int cap = 16;
  cudaStream_t s[cap]{};
  lane_split_handle* split[cap]{};
  int n = 0;
  void add(cudaStream_t x, lane_split_handle* h = nullptr)
  {
    for (int i = 0; i < n; ++i)
    {
      if (s[i] == x)
      {
        if (split[i] == nullptr)
        {
          split[i] = h;
        }
        return;
      }
    }
    if (n < cap)
    {
      s[n]     = x;
      split[n] = h;
      ++n;
    }
  }
};

// Register a lane with the environment's tracker, if any.
template <class Env>
void track(cudaStream_t lane, [[maybe_unused]] const Env& env)
{
  if constexpr (__queryable_with<Env, get_lane_tracker_t>)
  {
    if (lane_tracker* t = get_lane_tracker(env))
    {
      t->add(lane);
    }
  }
}

// Tell the observer, if any, about one cross-lane dependency (fork or join).
template <class Env>
void notify(cudaStream_t from, cudaStream_t to, [[maybe_unused]] const Env& env)
{
  if constexpr (__queryable_with<Env, get_lane_join_observer_t>)
  {
    get_lane_join_observer(env)(from, to);
  }
}

// Make `to` wait on every stream of `from` that is not `to`. The lazy join.
// `env` is the receiver's environment; if it carries a get_lane_join_observer,
// the observer is told about each event.
template <class Env>
void join_into(const stream_set& from, cudaStream_t to, [[maybe_unused]] const Env& env)
{
  for (int i = 0; i < from.n; ++i)
  {
    if (from.s[i] == to)
    {
      continue;
    }
    if (from.split[i] != nullptr && from.split[i]->ready != nullptr)
    {
      // The upstream recorded its completion point (a lane_split): depend on
      // that, not on whatever the lane enqueued since; and let the split know
      // this lane consumed it, so that it is joined back before the split's
      // values die.
      if (auto st = cudaStreamWaitEvent(to, from.split[i]->ready, 0); st != cudaSuccess)
      {
        throw ::cuda::cuda_error(st, "lane_scheduler: cudaStreamWaitEvent on a split's ready point failed");
      }
      from.split[i]->add_consumer(to);
    }
    else
    {
      // Event created in from.s[i]'s context (timing disabled), recorded, waited
      // on by `to`, destroyed on scope exit; throws cuda::cuda_error on failure.
      ::cuda::stream_ref{to}.wait(::cuda::stream_ref{from.s[i]});
    }
    notify(from.s[i], to, env);
  }
}

// If the environment carries a fork point whose origin is another lane, make
// `lane` wait on it. Returns true if it did.
template <class Env>
bool wait_fork(cudaStream_t lane, [[maybe_unused]] const Env& env)
{
  if constexpr (__queryable_with<Env, get_lane_fork_t>)
  {
    const lane_fork_point f = get_lane_fork(env);
    if (f.event != nullptr && f.origin != lane)
    {
      if (auto st = cudaStreamWaitEvent(lane, f.event, 0); st != cudaSuccess)
      {
        throw ::cuda::cuda_error(st, "lane_scheduler: cudaStreamWaitEvent on fork point failed");
      }
      notify(f.origin, lane, env);
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- domain ----
struct domain; // fwd
struct scheduler;

// ------------------------------------------------------------- scheduler ----
struct scheduler
{
  using scheduler_concept = scheduler_t;

  cudaStream_t stream_{nullptr};

  scheduler() = default;
  explicit scheduler(cudaStream_t s) noexcept
      : stream_{s}
  {}
  explicit scheduler(::cuda::stream_ref s) noexcept
      : stream_{s.get()}
  {}

  [[nodiscard]] cudaStream_t stream() const noexcept
  {
    return stream_;
  }

  // Queries on the scheduler itself. A scheduler is its own completion
  // scheduler; answering this lets adaptors whose attrs defer to their target
  // scheduler (continues_on, ...) report the lane they complete on, which the
  // upstream walk in `collect` relies on.
  [[nodiscard]] constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> scheduler
  {
    return *this;
  }
  [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>) const noexcept -> domain;
  [[nodiscard]] auto query(::cuda::get_stream_t) const noexcept -> ::cuda::stream_ref
  {
    return ::cuda::stream_ref{stream_};
  }
  [[nodiscard]] constexpr auto query(get_forward_progress_guarantee_t) const noexcept
  {
    return forward_progress_guarantee::weakly_parallel;
  }

  struct attrs_t
  {
    cudaStream_t s_;
    [[nodiscard]] constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::synchronous;
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_scheduler_t<set_value_t>, const Env&...) const noexcept
      -> scheduler
    {
      return scheduler{s_};
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept -> domain;
    [[nodiscard]] auto query(::cuda::get_stream_t) const noexcept -> ::cuda::stream_ref
    {
      return ::cuda::stream_ref{s_};
    }
  };

  template <class Rcvr>
  struct opstate_t : ::cuda::__immovable
  {
    using operation_state_concept = operation_state_t;
    Rcvr rcvr_;
    cudaStream_t s_;
    void start() noexcept
    {
      track(s_, execution::get_env(rcvr_));
      // Beginning on this lane inside a when_all that started on another lane:
      // depend on the when_all's fork point, not on the origin's current tail.
      wait_fork(s_, execution::get_env(rcvr_));
      // Everything downstream runs on the host right now and enqueues onto this
      // lane; make the lane's device/context current for all of it.
      const ::cuda::__ensure_current_context guard{::cuda::stream_ref{s_}};
      execution::set_value(static_cast<Rcvr&&>(rcvr_));
    }
  };

  struct sndr_t
  {
    using sender_concept = sender_t;
    cudaStream_t s_;

    template <class Self, class... Env>
    [[nodiscard]] static constexpr auto get_completion_signatures() noexcept
    {
      return completion_signatures<set_value_t()>{};
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr rcvr) const noexcept -> opstate_t<Rcvr>
    {
      return {{}, static_cast<Rcvr&&>(rcvr), s_};
    }
    [[nodiscard]] constexpr auto get_env() const noexcept -> attrs_t
    {
      return {s_};
    }
  };

  [[nodiscard]] constexpr auto schedule() const noexcept -> sndr_t
  {
    return {stream_};
  }
  friend constexpr bool operator==(scheduler a, scheduler b) noexcept
  {
    return a.stream_ == b.stream_;
  }
  friend constexpr bool operator!=(scheduler a, scheduler b) noexcept
  {
    return a.stream_ != b.stream_;
  }
};

// ------------------------------------------ upstream lane discovery ----------
// collect(sndr, set): the lanes a sender's set_value completion is ordered
// on. Terminal case: the sender's attrs name a lane::scheduler as completion
// scheduler (schedule(), then-chains, lane::on...). Otherwise descend into the
// sender's children via the structured-binding visitor (when_all, ...).
template <class Sndr>
void collect(const Sndr& s, stream_set& out);

struct collect_visitor
{
  template <class Tag, class Data, class... Children>
  void operator()(stream_set& out, Tag, const Data&, const Children&... children) const
  {
    (collect_child(children, out), ...);
  }
  template <class C>
  static void collect_child(const C& c, stream_set& out)
  {
    if constexpr (sender<C>)
    {
      collect(c, out);
    }
  }
};

template <class Sndr, bool = ::cuda::experimental::__callable<get_completion_scheduler_t<set_value_t>, env_of_t<Sndr>>>
struct completes_on_lane_t : ::std::false_type
{};
template <class Sndr>
struct completes_on_lane_t<Sndr, true>
    : ::std::is_same<
        ::std::decay_t<::cuda::std::__call_result_t<get_completion_scheduler_t<set_value_t>, env_of_t<Sndr>>>,
        scheduler>
{};
template <class Sndr>
inline constexpr bool completes_on_lane = completes_on_lane_t<Sndr>::value;

template <class Sndr>
void collect(const Sndr& s, stream_set& out)
{
  if constexpr (completes_on_lane<Sndr>)
  {
    const auto attrs     = execution::get_env(s);
    lane_split_handle* h = nullptr;
    if constexpr (__queryable_with<decltype(attrs), get_lane_ready_t>)
    {
      h = get_lane_ready(attrs);
    }
    out.add(execution::get_completion_scheduler<set_value_t>(attrs).stream(), h);
  }
  else if constexpr (structured_binding_size<Sndr> >= 2)
  {
    collect_visitor v{};
    execution::visit(v, s, out);
  }
  // else: host-only sender, no lane.
}

// ------------------------------------------------------------- lane::on -----
// on(sndr, sched): complete `sndr`'s values on `sched`'s lane, inserting the
// cross-stream event join lazily (only for upstream streams != target).
struct on_tag_t
{};

struct on_t
{
  template <class Sndr>
  struct sndr_t;

  template <class Sndr, class Rcvr>
  struct state_t
  {
    Rcvr rcvr_;
    scheduler sch_;
    stream_set upstream_;
    // The child is a bare schedule(lane): no work of its own between the lane's
    // fork point and this transfer, so a fork point for that lane is the right
    // thing to wait on (a fresh event would also capture the siblings' work).
    bool bare_schedule_ = false;
  };

  template <class Sndr, class Rcvr>
  struct rcvr_t
  {
    using receiver_concept = receiver_t;
    state_t<Sndr, Rcvr>* st_;

    template <class... Ts>
    void set_value(Ts&&... ts) noexcept
    {
      const auto& env       = execution::get_env(st_->rcvr_);
      const cudaStream_t to = st_->sch_.stream();
      track(to, env);
      bool forked = false;
      if (st_->bare_schedule_ && st_->upstream_.n == 1 && st_->upstream_.s[0] != to)
      {
        if constexpr (__queryable_with<decltype(env), get_lane_fork_t>)
        {
          const lane_fork_point f = get_lane_fork(env);
          if (f.event != nullptr && f.origin == st_->upstream_.s[0])
          {
            if (auto st = cudaStreamWaitEvent(to, f.event, 0); st != cudaSuccess)
            {
              throw ::cuda::cuda_error(st, "lane_scheduler: cudaStreamWaitEvent on fork point failed");
            }
            notify(f.origin, to, env);
            forked = true;
          }
        }
      }
      if (!forked)
      {
        join_into(st_->upstream_, to, env);
      }
      const ::cuda::__ensure_current_context guard{::cuda::stream_ref{to}};
      execution::set_value(static_cast<Rcvr&&>(st_->rcvr_), static_cast<Ts&&>(ts)...);
    }
    template <class E>
    void set_error(E&& e) noexcept
    {
      execution::set_error(static_cast<Rcvr&&>(st_->rcvr_), static_cast<E&&>(e));
    }
    void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<Rcvr&&>(st_->rcvr_));
    }
    [[nodiscard]] auto get_env() const noexcept -> __fwd_env_t<env_of_t<Rcvr>>
    {
      return execution::__fwd_env(execution::get_env(st_->rcvr_));
    }
  };

  template <class CvSndr, class Rcvr>
  struct opstate_t
  {
    using operation_state_concept = operation_state_t;
    using Sndr                    = ::std::decay_t<CvSndr>;
    state_t<Sndr, Rcvr> st_;
    connect_result_t<CvSndr, rcvr_t<Sndr, Rcvr>> op_;

    opstate_t(CvSndr&& s, scheduler sch, Rcvr r)
        : st_{static_cast<Rcvr&&>(r), sch, {}, ::std::is_same_v<Sndr, scheduler::sndr_t>}
        , op_{execution::connect((collect(s, st_.upstream_), static_cast<CvSndr&&>(s)), rcvr_t<Sndr, Rcvr>{&st_})}
    {}
    opstate_t(opstate_t&&) = delete;
    void start() noexcept
    {
      execution::start(op_);
    }
  };

  template <class Sndr>
  struct attrs_t
  {
    const sndr_t<Sndr>* self_;
    [[nodiscard]] constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::synchronous;
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_scheduler_t<set_value_t>, const Env&...) const noexcept
      -> scheduler
    {
      return self_->sch_;
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept -> domain;
    [[nodiscard]] auto query(::cuda::get_stream_t) const noexcept -> ::cuda::stream_ref
    {
      return ::cuda::stream_ref{self_->sch_.stream()};
    }
  };

  template <class Sndr>
  struct sndr_t
  {
    using sender_concept = sender_t;
    on_tag_t tag_;
    scheduler sch_;
    Sndr sndr_;

    template <class Self, class... Env>
    [[nodiscard]] static constexpr auto get_completion_signatures()
    {
      return execution::get_completion_signatures<::cuda::std::__copy_cvref_t<Self, Sndr>, __fwd_env_t<Env>...>();
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr r) && -> opstate_t<Sndr, Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), sch_, static_cast<Rcvr&&>(r)};
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr r) const& -> opstate_t<const Sndr&, Rcvr>
    {
      return {sndr_, sch_, static_cast<Rcvr&&>(r)};
    }
    [[nodiscard]] constexpr auto get_env() const noexcept -> attrs_t<Sndr>
    {
      return {this};
    }
  };

  template <class Sndr>
  [[nodiscard]] auto operator()(Sndr sndr, scheduler sch) const -> sndr_t<Sndr>
  {
    return {{}, sch, static_cast<Sndr&&>(sndr)};
  }
  struct closure_t
  {
    scheduler sch_;
    template <class Sndr>
    friend auto operator|(Sndr sndr, closure_t c)
    {
      return on_t{}(static_cast<Sndr&&>(sndr), c.sch_);
    }
  };
  [[nodiscard]] auto operator()(scheduler sch) const -> closure_t
  {
    return {sch};
  }
};
inline constexpr on_t on{};

// Is this environment the one a when_all hands its children (possibly behind
// forwarding-env layers added by then/continues_on)?
template <class Env>
struct strip_fwd_env
{
  using type = Env;
};
template <class Env>
struct strip_fwd_env<__fwd_env_<Env>> : strip_fwd_env<Env>
{};
template <class Env>
inline constexpr bool is_when_all_child_env_ = false;
template <class Rcvr>
inline constexpr bool is_when_all_child_env_<when_all_t::__env_t<Rcvr>> = true;
template <class Env>
inline constexpr bool is_when_all_child_env = is_when_all_child_env_<typename strip_fwd_env<::std::decay_t<Env>>::type>;

// ------------------------------------------------------ fork at when_all -----
// A when_all whose children run on lanes, started on a lane: record one event on
// the origin lane *before* any child starts, and give it to the children through
// their environment (get_lane_fork). The event is destroyed right after the
// children have started: every wait on it has been enqueued by then, since lane
// senders complete synchronously.
struct fork_tag_t
{};

struct fork_when_all_t
{
  // Marks the environment the inner when_all is connected with, so that the
  // domain does not wrap it a second time. Not a forwarding query: the
  // children's when_alls are wrapped with forks of their own.
  struct no_wrap_t
  {};

  // The two queries this receiver adds to its environment. Answered by value:
  // the environment object itself is a temporary built on each get_env().
  struct fork_props_t
  {
    const lane_fork_point* fork_;
    [[nodiscard]] lane_fork_point query(get_lane_fork_t) const noexcept
    {
      return *fork_;
    }
    [[nodiscard]] constexpr bool query(no_wrap_t) const noexcept
    {
      return true;
    }
  };

  template <class Rcvr>
  struct rcvr_t
  {
    using receiver_concept = receiver_t;
    Rcvr* rcvr_;
    lane_fork_point* fork_;

    template <class... Ts>
    void set_value(Ts&&... ts) noexcept
    {
      execution::set_value(static_cast<Rcvr&&>(*rcvr_), static_cast<Ts&&>(ts)...);
    }
    template <class E>
    void set_error(E&& e) noexcept
    {
      execution::set_error(static_cast<Rcvr&&>(*rcvr_), static_cast<E&&>(e));
    }
    void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<Rcvr&&>(*rcvr_));
    }
    [[nodiscard]] auto get_env() const noexcept
    {
      return env{fork_props_t{fork_}, execution::__fwd_env(execution::get_env(*rcvr_))};
    }
  };

  template <class CvSndr, class Rcvr>
  struct opstate_t
  {
    using operation_state_concept = operation_state_t;
    using Sndr                    = ::std::decay_t<CvSndr>;
    Rcvr rcvr_;
    lane_fork_point fork_{};
    cudaStream_t origin_{nullptr};
    bool needs_fork_ = false;
    bool reuse_      = false;
    connect_result_t<CvSndr, rcvr_t<Rcvr>> op_;

    opstate_t(CvSndr&& s, Rcvr r)
        : rcvr_{static_cast<Rcvr&&>(r)}
        , op_{execution::connect((prepare(s), static_cast<CvSndr&&>(s)), rcvr_t<Rcvr>{&rcvr_, &fork_})}
    {}
    opstate_t(opstate_t&&) = delete;

    // At connect: the origin lane (the environment's scheduler, if it is a lane)
    // and whether any child touches another lane. If every lane the children
    // complete on is the origin, no fork point is needed.
    //
    // A when_all forks from its predecessor's completion point. A when_all can
    // only acquire upstream lane work through a let_value body (whose entry is
    // where this wrapper records a new point) or by being a child of an outer
    // when_all, whose predecessor is the same: in that case the enclosing fork
    // point is reused rather than re-recorded at this when_all's start, which
    // would make the children depend on whatever the sibling stages started
    // earlier enqueued on the origin lane in between.
    void prepare(const Sndr& s)
    {
      const auto& env = execution::get_env(rcvr_);
      if constexpr (__callable<get_scheduler_t, decltype(env)>)
      {
        if constexpr (::std::is_same_v<::std::decay_t<decltype(get_scheduler(env))>, scheduler>)
        {
          origin_ = get_scheduler(env).stream();
          stream_set lanes{};
          collect(s, lanes);
          needs_fork_ = lanes.n == 0; // unknown lanes (e.g. inside let_value): be safe
          for (int i = 0; i < lanes.n; ++i)
          {
            needs_fork_ = needs_fork_ || lanes.s[i] != origin_;
          }
          if constexpr (is_when_all_child_env<decltype(env)> && __queryable_with<decltype(env), get_lane_fork_t>)
          {
            reuse_ = needs_fork_;
          }
        }
      }
    }

    void start() noexcept
    {
      if (reuse_)
      {
        if constexpr (__queryable_with<env_of_t<Rcvr>, get_lane_fork_t>)
        {
          fork_ = get_lane_fork(execution::get_env(rcvr_)); // the enclosing when_all's point
        }
        execution::start(op_);
        fork_ = {};
      }
      else if (needs_fork_)
      {
        const ::cuda::__ensure_current_context guard{::cuda::stream_ref{origin_}};
        cudaEvent_t e{};
        if (auto st = cudaEventCreateWithFlags(&e, cudaEventDisableTiming); st != cudaSuccess)
        {
          throw ::cuda::cuda_error(st, "lane fork: cudaEventCreateWithFlags failed");
        }
        if (auto st = cudaEventRecord(e, origin_); st != cudaSuccess)
        {
          cudaEventDestroy(e);
          throw ::cuda::cuda_error(st, "lane fork: cudaEventRecord failed");
        }
        fork_ = {origin_, e};
        execution::start(op_); // children start; their waits on the fork point are enqueued now
        fork_ = {};
        cudaEventDestroy(e);
      }
      else
      {
        execution::start(op_);
      }
    }
  };

  template <class Sndr>
  struct sndr_t
  {
    using sender_concept = sender_t;
    fork_tag_t tag_;
    ::cuda::std::__ignore_t data_;
    Sndr sndr_;

    template <class Self, class... Env>
    [[nodiscard]] static constexpr auto get_completion_signatures()
    {
      // The inner when_all's own member, not the dispatcher: the dispatcher would run
      // transform_sender on it and wrap it a second time.
      return Sndr::template get_completion_signatures<::cuda::std::__copy_cvref_t<Self, Sndr>, __fwd_env_t<Env>...>();
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr r) && -> opstate_t<Sndr, Rcvr>
    {
      return {static_cast<Sndr&&>(sndr_), static_cast<Rcvr&&>(r)};
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr r) const& -> opstate_t<const Sndr&, Rcvr>
    {
      return {sndr_, static_cast<Rcvr&&>(r)};
    }
    [[nodiscard]] decltype(auto) get_env() const noexcept
    {
      return execution::get_env(sndr_);
    }
  };
};

// ------------------------------------------------------------- lane::split ---
// One lane sender, several consumers (P2300 `split`). The child runs once,
// started by the first consumer to start; when it completes on its lane, one
// event is recorded there -- the "ready" point -- and every consumer receives
// the (decayed) values. The ready point is exposed as the sender's
// `get_lane_ready` attribute, so a `continues_on` downstream of a consumer on
// another lane waits on *that* event (`join_into`) rather than on the lane's
// tail: the consumer depends on the split's completion point, not on whatever
// the lane enqueued afterwards. Lane completion is synchronous and every
// consumer connects before the pipeline starts, so there are no races to
// handle; the shared state is heap-allocated, as in every `split`.
struct split_t
{
  struct waiter
  {
    waiter* next_              = nullptr;
    void (*complete_)(waiter*) = nullptr;
  };

  template <class Values>
  struct shared_base
  {
    lane_split_handle* handle_{nullptr}; // owned by the control block
    bool started_ = false;
    bool done_    = false;
    int kind_     = 0; // 1 value, 2 error, 3 stopped
    Values values_{};
    exception_ptr error_{};
    waiter* waiters_ = nullptr;

    virtual ~shared_base()     = default;
    virtual void start_child() = 0;

    void push(waiter* w) noexcept
    {
      w->next_ = waiters_;
      waiters_ = w;
    }
    void finish(int kind)
    {
      kind_ = kind;
      done_ = true;
      if (kind == 1)
      {
        if (auto st = cudaEventCreateWithFlags(&handle_->ready, cudaEventDisableTiming); st != cudaSuccess)
        {
          throw ::cuda::cuda_error(st, "lane_split: cudaEventCreateWithFlags failed");
        }
        if (auto st = cudaEventRecord(handle_->ready, handle_->lane); st != cudaSuccess)
        {
          throw ::cuda::cuda_error(st, "lane_split: cudaEventRecord failed");
        }
      }
      // Waiters were pushed in start order; complete them in that order.
      waiter* list = waiters_;
      waiters_     = nullptr;
      waiter* rev  = nullptr;
      while (list)
      {
        waiter* n   = list->next_;
        list->next_ = rev;
        rev         = list;
        list        = n;
      }
      for (waiter* w = rev; w;)
      {
        waiter* n = w->next_;
        w->complete_(w);
        w = n;
      }
    }
  };

  template <class Base, class Env>
  struct child_rcvr_t
  {
    using receiver_concept = receiver_t;
    Base* base_;
    Env env_;
    template <class... Ts>
    void set_value(Ts&&... ts) noexcept
    {
      base_->values_ = typename Base::values_type{static_cast<Ts&&>(ts)...};
      base_->finish(1);
    }
    void set_error(exception_ptr e) noexcept
    {
      base_->error_ = static_cast<exception_ptr&&>(e);
      base_->finish(2);
    }
    void set_stopped() noexcept
    {
      base_->finish(3);
    }
    [[nodiscard]] auto get_env() const noexcept -> const Env&
    {
      return env_;
    }
  };

  // The child is connected with the environment of the first consumer to
  // connect (forwarding queries only): that is how its `schedule(lane)` sees the
  // enclosing when_all's fork point, and so joins a capture at the right point.
  template <class Sndr, class Values, class Env>
  struct shared_impl : shared_base<Values>
  {
    using values_type = Values;
    connect_result_t<Sndr, child_rcvr_t<shared_impl, Env>> op_;
    shared_impl(Sndr&& s, Env env)
        : op_{execution::connect(static_cast<Sndr&&>(s), child_rcvr_t<shared_impl, Env>{this, static_cast<Env&&>(env)})}
    {}
    void start_child() override
    {
      execution::start(op_);
    }
  };

  template <class Sndr>
  struct sndr_t;

  template <class Sndr, class Rcvr>
  struct opstate_t : waiter
  {
    using operation_state_concept = operation_state_t;
    using values_t                = typename sndr_t<Sndr>::values_t;
    Rcvr rcvr_;
    ::std::shared_ptr<shared_base<values_t>> state_;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE opstate_t(Rcvr r, ::std::shared_ptr<shared_base<values_t>> st)
        : rcvr_{static_cast<Rcvr&&>(r)}
        , state_{::std::move(st)}
    {
      this->complete_ = [](waiter* w) {
        static_cast<opstate_t*>(w)->complete();
      };
    }
    opstate_t(opstate_t&&) = delete;
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE ~opstate_t() {}

    void start() noexcept
    {
      if (state_->done_)
      {
        complete();
        return;
      }
      state_->push(this);
      if (!state_->started_)
      {
        state_->started_ = true;
        state_->start_child(); // completes synchronously, finishing every pushed waiter
      }
    }
    void complete() noexcept
    {
      switch (state_->kind_)
      {
        case 1:
          ::cuda::std::__apply(
            [&](auto&... vs) {
              execution::set_value(static_cast<Rcvr&&>(rcvr_), vs...);
            },
            state_->values_);
          break;
        case 2:
          execution::set_error(static_cast<Rcvr&&>(rcvr_), state_->error_);
          break;
        default:
          execution::set_stopped(static_cast<Rcvr&&>(rcvr_));
      }
    }
  };

  template <class Sndr>
  struct attrs_t
  {
    scheduler sch_;
    lane_split_handle* handle_;
    [[nodiscard]] constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::synchronous;
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_scheduler_t<set_value_t>, const Env&...) const noexcept
      -> scheduler
    {
      return sch_;
    }
    template <class... Env>
    [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept -> domain;
    [[nodiscard]] auto query(::cuda::get_stream_t) const noexcept -> ::cuda::stream_ref
    {
      return ::cuda::stream_ref{sch_.stream()};
    }
    [[nodiscard]] auto query(get_lane_ready_t) const noexcept -> lane_split_handle*
    {
      return handle_;
    }
  };

  template <class... Ts>
  using set_value_sig_t = completion_signatures<set_value_t(::std::decay_t<Ts>...)>;

  template <class Sndr>
  struct sndr_t
  {
    using sender_concept = sender_t;
    static_assert(!dependent_sender<Sndr>, "lane_split: the child must know its completions without an environment");
    using completions_t = decltype(execution::get_completion_signatures<Sndr>());
    using values_t      = __value_types<completions_t, ::cuda::std::__decayed_tuple, ::cuda::std::__type_self_t>;

    // Copies share the control block. The child is held there until the first
    // connect, when it is connected into the shared state with that consumer's
    // environment. The ready event's slot lives in the control block so that
    // `get_lane_ready` can hand out its address before anything is connected.
    struct control
    {
      lane_split_handle handle_{};
      ::std::shared_ptr<shared_base<values_t>> impl_; // declared after the handle: destroyed before it
      Sndr sndr_;
      scheduler sch_;
      // The child's lane: its completion scheduler when it reports one; else
      // (let_value bodies do not) the single lane the sender tree completes on.
      static scheduler lane_of(const Sndr& s)
      {
        if constexpr (completes_on_lane<Sndr>)
        {
          return execution::get_completion_scheduler<set_value_t>(execution::get_env(s));
        }
        else
        {
          stream_set lanes{};
          collect(s, lanes);
          if (lanes.n != 1)
          {
            throw ::std::logic_error("lane_split: the child must complete on exactly one lane");
          }
          return scheduler{lanes.s[0]};
        }
      }
      explicit control(Sndr s)
          : sndr_{static_cast<Sndr&&>(s)}
          , sch_{lane_of(sndr_)}
      {
        handle_.lane = sch_.stream();
      }
      // Scope end. Every lane that consumed the ready point is joined back into
      // the split's lane first; only then is the child's operation state (and
      // whatever its values own) destroyed, by the members' destruction below.
      ~control()
      {
        for (int i = 0; i < handle_.n_consumers; ++i)
        {
          if (handle_.consumers[i] != handle_.lane)
          {
            ::cuda::stream_ref{handle_.lane}.wait(::cuda::stream_ref{handle_.consumers[i]});
          }
        }
        impl_.reset(); // the child's operation state dies here, after the joins
        if (handle_.ready)
        {
          cudaEventDestroy(handle_.ready);
        }
      }
    };
    ::std::shared_ptr<control> ctl_;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE explicit sndr_t(Sndr s)
        : ctl_{::std::make_shared<control>(static_cast<Sndr&&>(s))}
    {}
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE sndr_t(const sndr_t& o)
        : ctl_{o.ctl_}
    {}
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE sndr_t(sndr_t&& o) noexcept
        : ctl_{::std::move(o.ctl_)}
    {}
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE ~sndr_t() {}

    template <class Self, class... Env>
    [[nodiscard]] static constexpr auto get_completion_signatures()
    {
      return concat_completion_signatures(
        __value_types<completions_t, set_value_sig_t, ::cuda::std::__type_self_t>{},
        __eptr_completion(),
        completion_signatures<set_stopped_t()>{});
    }
    template <class Rcvr>
    [[nodiscard]] auto connect(Rcvr r) const -> opstate_t<Sndr, Rcvr>
    {
      if (!ctl_->impl_)
      {
        using env_t = __fwd_env_t<env_of_t<Rcvr>>;
        auto impl   = ::std::make_shared<shared_impl<Sndr, values_t, env_t>>(
          static_cast<Sndr&&>(ctl_->sndr_), execution::__fwd_env(execution::get_env(r)));
        impl->handle_ = &ctl_->handle_;
        ctl_->impl_   = impl;
      }
      return {static_cast<Rcvr&&>(r), ctl_->impl_};
    }
    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_HOST_DEVICE auto get_env() const noexcept -> attrs_t<Sndr>
    {
      return {ctl_->sch_, &ctl_->handle_};
    }
  };

  template <class Sndr>
  [[nodiscard]] auto operator()(Sndr sndr) const -> sndr_t<Sndr>
  {
    return sndr_t<Sndr>{static_cast<Sndr&&>(sndr)};
  }
};
inline constexpr split_t split{};

template <class S>
inline constexpr bool is_when_all = false;
template <class... Children>
inline constexpr bool is_when_all<when_all_t::__sndr_t<Children...>> = true;

// ---------------------------------------------------------------- domain ----
template <class S>
inline constexpr bool is_continues_on_to_lane = false;
template <class Child>
inline constexpr bool is_continues_on_to_lane<continues_on_t::__sndr_t<scheduler, Child>> = true;

// sync_wait with a tracker added to the environment (or a fresh env).
template <class Sndr>
auto sync_wait_tracked(Sndr&& sndr, lane_tracker& t)
{
  return sync_wait.apply_sender(static_cast<Sndr&&>(sndr), env{prop{get_lane_tracker, &t}});
}
template <class Sndr, class Env>
auto sync_wait_tracked(Sndr&& sndr, lane_tracker& t, Env&& e)
{
  return sync_wait.apply_sender(static_cast<Sndr&&>(sndr), env{prop{get_lane_tracker, &t}, static_cast<Env&&>(e)});
}

struct domain
{
  // sync_wait: host completion, then synchronize every lane the sender
  // completes on. Everything else: the tag's own apply_sender.
  template <class Tag, class Sndr, class... Args>
  static auto apply_sender(Tag, Sndr&& sndr, Args&&... args)
  {
    if constexpr (::std::is_same_v<Tag, sync_wait_t>)
    {
      // Host completion, with a tracker in the environment; then synchronize
      // every lane the pipeline touched (the operation state has died inside
      // the generic sync_wait by then, so scoped frees are enqueued too).
      lane_tracker tracked{};
      {
        stream_set completion_lanes{}; // seed with the completion lanes, in case nothing registers
        collect(sndr, completion_lanes);
        for (int i = 0; i < completion_lanes.n; ++i)
        {
          tracked.add(completion_lanes.s[i]);
        }
      }
      auto result = sync_wait_tracked(static_cast<Sndr&&>(sndr), tracked, static_cast<Args&&>(args)...);
      for (int i = 0; i < tracked.n; ++i)
      {
        if (auto st = cudaStreamSynchronize(tracked.s[i]); st != cudaSuccess)
        {
          throw ::cuda::cuda_error(st, "lane::sync_wait: cudaStreamSynchronize failed");
        }
      }
      return result;
    }
    else
    {
      return Tag{}.apply_sender(static_cast<Sndr&&>(sndr), static_cast<Args&&>(args)...);
    }
  }

  // continues_on(sndr, lane) -> lane::on(sndr, lane). Everything else is
  // default_domain behaviour. continues_on eagerly wraps its child in a
  // schedule_from sender; we unwrap that so the upstream walk sees the real
  // predecessor (when_all, then-chain...).
  template <class Child>
  static auto unwrap_schedule_from(Child&& child)
  {
    if constexpr (sender_for<Child, schedule_from_t>)
    {
      auto&& [tag, data, inner] = static_cast<Child&&>(child);
      return static_cast<decltype(inner)&&>(inner);
    }
    else
    {
      return static_cast<Child&&>(child);
    }
  }

  template <class OpTag, class Sndr, class Env>
  static auto transform_sender(OpTag, Sndr&& sndr, const Env& env)
  {
    if constexpr (is_continues_on_to_lane<::std::decay_t<Sndr>>)
    {
      auto&& [tag, sch, child] = static_cast<Sndr&&>(sndr);
      return on_t{}(unwrap_schedule_from(static_cast<decltype(child)&&>(child)), sch);
    }
    else if constexpr (is_when_all<::std::decay_t<Sndr>> && !__queryable_with<Env, fork_when_all_t::no_wrap_t>)
    {
      // when_all under a lane: fork point recorded before the children start.
      return fork_when_all_t::sndr_t<::std::decay_t<Sndr>>{{}, {}, static_cast<Sndr&&>(sndr)};
    }
    else
    {
      return default_domain{}.transform_sender(OpTag{}, static_cast<Sndr&&>(sndr), env);
    }
  }
};

inline constexpr auto scheduler::query(get_completion_domain_t<set_value_t>) const noexcept -> domain
{
  return {};
}
template <class... Env>
inline constexpr auto scheduler::attrs_t::query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept
  -> domain
{
  return {};
}
template <class Sndr>
template <class... Env>
inline constexpr auto on_t::attrs_t<Sndr>::query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept
  -> domain
{
  return {};
}
template <class Sndr>
template <class... Env>
inline constexpr auto split_t::attrs_t<Sndr>::query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept
  -> domain
{
  return {};
}
} // namespace __lane

//! The public names.
using lane_scheduler = __lane::scheduler;
using lane_domain    = __lane::domain;

//! @brief Make `to` depend on every lane in `lanes` (one event per lane other
//! than `to` itself). The stream-ordered terminal of a lane pipeline: the
//! counterpart of `sync_wait`'s host wait for a stream the caller owns, legal
//! under capture (the waits become graph edges). Call it after the pipeline's
//! operation state has died, so that scoped frees are included.
inline void lane_join_into(::cuda::stream_ref to, const lane_tracker& lanes)
{
  for (int i = 0; i < lanes.n; ++i)
  {
    if (lanes.s[i] != to.get())
    {
      to.wait(::cuda::stream_ref{lanes.s[i]});
    }
  }
}

//! @brief Add a lane tracker to an environment.
template <class Env = env<>>
[[nodiscard]] auto lane_tracked(lane_tracker& t, Env e = {})
{
  return env{prop{get_lane_tracker, &t}, static_cast<Env&&>(e)};
}

namespace __lane
{
template <class Env>
struct capture_rcvr_t
{
  using receiver_concept = receiver_t;
  Env env_;
  bool* failed_;
  template <class... Ts>
  void set_value(Ts&&...) noexcept
  {}
  template <class E>
  void set_error(E&&) noexcept
  {
    *failed_ = true;
  }
  void set_stopped() noexcept
  {
    *failed_ = true;
  }
  [[nodiscard]] auto get_env() const noexcept -> const Env&
  {
    return env_;
  }
};
} // namespace __lane

//! @brief Run `sndr` inside a stream capture that begins and ends on `origin`'s
//! lane, and return the graph. Every lane the pipeline touched is joined back
//! into the origin before the capture ends, after the operation state has died
//! (so scoped frees are captured too). Nothing a caller could not write with
//! `connect`/`start` and `lane_join_into`; this is the convenience.
template <class Sndr, class Env = env<>>
[[nodiscard]] cudaGraph_t lane_capture(lane_scheduler origin, Sndr&& sndr, Env e = {})
{
  lane_tracker lanes{};
  lanes.add(origin.stream());
  if (auto st = cudaStreamBeginCapture(origin.stream(), cudaStreamCaptureModeThreadLocal); st != cudaSuccess)
  {
    throw ::cuda::cuda_error(st, "lane_capture: cudaStreamBeginCapture failed");
  }
  bool failed = false;
  {
    auto op = execution::connect(static_cast<Sndr&&>(sndr),
                                 __lane::capture_rcvr_t<decltype(lane_tracked(lanes, static_cast<Env&&>(e)))>{
                                   lane_tracked(lanes, static_cast<Env&&>(e)), &failed});
    execution::start(op);
  } // the operation state dies here: scoped frees are enqueued, inside the capture
  lane_join_into(::cuda::stream_ref{origin.stream()}, lanes);
  cudaGraph_t graph{};
  if (auto st = cudaStreamEndCapture(origin.stream(), &graph); st != cudaSuccess)
  {
    throw ::cuda::cuda_error(st, "lane_capture: cudaStreamEndCapture failed");
  }
  if (failed)
  {
    cudaGraphDestroy(graph);
    throw ::std::runtime_error("lane_capture: the pipeline did not complete with a value");
  }
  return graph;
}
//! `lane_split(sndr)`: one lane sender, several consumers; see `__lane::split_t`.
inline constexpr auto& lane_split = __lane::split;

template <class Sndr>
inline constexpr int structured_binding_size<__lane::on_t::sndr_t<Sndr>> = 3;
template <class Sndr>
inline constexpr int structured_binding_size<__lane::fork_when_all_t::sndr_t<Sndr>> = 3;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_LANE_SCHEDULER
