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
//! Events come from a small pool created once (timing disabled); an event is
//! returned to the pool right after the wait is enqueued, since
//! `cudaStreamWaitEvent` captures the event's state at call time.
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
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda_runtime_api.h>

#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

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

namespace __lane
{

// ---------------------------------------------------------------- events ----
struct event_pool
{
  static event_pool& instance()
  {
    static event_pool p;
    return p;
  }
  cudaEvent_t get()
  {
    ::std::lock_guard<::std::mutex> g{mu_};
    if (!free_.empty())
    {
      auto e = free_.back();
      free_.pop_back();
      return e;
    }
    cudaEvent_t e{};
    if (auto st = cudaEventCreateWithFlags(&e, cudaEventDisableTiming); st != cudaSuccess)
    {
      throw ::cuda::cuda_error(st, "lane_scheduler: cudaEventCreateWithFlags failed");
    }
    return e;
  }
  void put(cudaEvent_t e)
  {
    ::std::lock_guard<::std::mutex> g{mu_};
    free_.push_back(e);
  }

private:
  ::std::mutex mu_;
  ::std::vector<cudaEvent_t> free_;
};

// ------------------------------------------------------- stream set ----------
struct stream_set
{
  static constexpr int cap = 16;
  cudaStream_t s[cap]{};
  int n = 0;
  void add(cudaStream_t x)
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

// Record an event on every stream of `from` that is not `to`, and make `to`
// wait on it. The lazy join. `env` is the receiver's environment; if it carries
// a get_lane_join_observer, the observer is told about each event.
template <class Env>
void join_into(const stream_set& from, cudaStream_t to, [[maybe_unused]] const Env& env)
{
  auto& pool = event_pool::instance();
  for (int i = 0; i < from.n; ++i)
  {
    if (from.s[i] == to)
    {
      continue;
    }
    cudaEvent_t e = pool.get();
    if (auto st = cudaEventRecord(e, from.s[i]); st != cudaSuccess)
    {
      throw ::cuda::cuda_error(st, "lane_scheduler: cudaEventRecord failed");
    }
    if (auto st = cudaStreamWaitEvent(to, e, 0); st != cudaSuccess)
    {
      throw ::cuda::cuda_error(st, "lane_scheduler: cudaStreamWaitEvent failed");
    }
    pool.put(e);
    if constexpr (__queryable_with<Env, get_lane_join_observer_t>)
    {
      get_lane_join_observer(env)(from.s[i], to);
    }
  }
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
    [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept
      -> domain;
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
    void start() noexcept
    {
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
      return {{}, static_cast<Rcvr&&>(rcvr)};
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
    out.add(execution::get_completion_scheduler<set_value_t>(execution::get_env(s)).stream());
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
  };

  template <class Sndr, class Rcvr>
  struct rcvr_t
  {
    using receiver_concept = receiver_t;
    state_t<Sndr, Rcvr>* st_;

    template <class... Ts>
    void set_value(Ts&&... ts) noexcept
    {
      join_into(st_->upstream_, st_->sch_.stream(), execution::get_env(st_->rcvr_));
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
        : st_{static_cast<Rcvr&&>(r), sch, {}}
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
    [[nodiscard]] constexpr auto query(get_completion_domain_t<set_value_t>, const Env&...) const noexcept
      -> domain;
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

// ---------------------------------------------------------------- domain ----
template <class S>
inline constexpr bool is_continues_on_to_lane = false;
template <class Child>
inline constexpr bool is_continues_on_to_lane<continues_on_t::__sndr_t<scheduler, Child>> = true;

struct domain
{
  // sync_wait: host completion, then synchronize every lane the sender
  // completes on. Everything else: the tag's own apply_sender.
  template <class Tag, class Sndr, class... Args>
  static auto apply_sender(Tag, Sndr&& sndr, Args&&... args)
  {
    if constexpr (::std::is_same_v<Tag, sync_wait_t>)
    {
      stream_set lanes{};
      collect(sndr, lanes);
      auto result = sync_wait.apply_sender(static_cast<Sndr&&>(sndr), static_cast<Args&&>(args)...);
      for (int i = 0; i < lanes.n; ++i)
      {
        if (auto st = cudaStreamSynchronize(lanes.s[i]); st != cudaSuccess)
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

} // namespace __lane

//! The public names.
using lane_scheduler = __lane::scheduler;
using lane_domain    = __lane::domain;

template <class Sndr>
inline constexpr int structured_binding_size<__lane::on_t::sndr_t<Sndr>> = 3;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_LANE_SCHEDULER
