//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/internal/async_resources_handle.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/utility/getenv_cache.cuh>
#include <cuda/experimental/__stf/utility/memory.cuh>
#include <cuda/experimental/__stf/utility/stream_to_dev.cuh>
#include <cuda/experimental/__stf/utility/unstable_unique.cuh>

#include <mutex>

namespace cuda::experimental::stf
{
class stream_and_event;
namespace reserved
{
inline event join_with_stream(
  backend_ctx_untyped& bctx, decorated_stream dstream, event_list& prereq_in, ::std::string string, bool record_event);

using stream_and_event_vector = small_vector<reserved::handle<stream_and_event>, 7>;
} // namespace reserved

/* This event type allows to synchronize a CUDA stream with a CUDA event in
 * another stream, possibly on another device. */
class stream_and_event : public event_impl
{
protected:
  stream_and_event()                                   = default;
  stream_and_event(const stream_and_event&)            = delete;
  stream_and_event& operator=(const stream_and_event&) = delete;

  stream_and_event(stream_and_event&&) = delete;

  ~stream_and_event() override
  {
    // fprintf(stderr, "DESTROY EVENT %s (%d) - cudaEvent %p\n", this->get_symbol().c_str(),
    // int(this->unique_prereq_id), cudaEvent);
    if (cudaEvent)
    {
      cuda_safe_call(cudaEventDestroy(cudaEvent));

      //            fprintf(stderr, "DESTROY EVENT %p #%d (created %d)\n", event, ++destroyed_event_cnt,
      //            event_cnt);
    }
  }

  stream_and_event(const decorated_stream& dstream, bool do_insert_event)
      : dstream(dstream)
  {
    // fprintf(stderr, "stream_and_event %s (ID %d)\n", this->get_symbol().c_str(), int(this->unique_prereq_id));
    if (do_insert_event)
    {
      insert_event();
    }
  }

public:
  // @@@@TODO@@@@ remove this function so that we have a mere
  // insert_dep call, and add the initial test to check whether both
  // streams are the same here too.
  // Insert a dependency from stream s1 over s2 (s1 waits for s2)
  static void insert_dependency(cudaStream_t s1, cudaStream_t s2)
  {
    // Create a dependency between the last stream and the current stream

    // event and stream must be on the same device
    // if the stream does not belong to the current device, we
    // therefore have to find in which device the stream was created,
    // record the event, and restore the current device to its original
    // value.

    // Find the stream structure in the driver API
    CUstream s2_driver = CUstream(s2);
    CUcontext ctx;
    cuda_safe_call(cuStreamGetCtx(s2_driver, &ctx));

    // Query the context associated with a stream by using the underlying driver API
    CUdevice s2_dev;
    cuda_safe_call(cuCtxPushCurrent(ctx));
    cuda_safe_call(cuCtxGetDevice(&s2_dev));
    cuda_safe_call(cuCtxPopCurrent(&ctx));

    // ::std::cout << "STREAM DEVICE = " << s2_dev << ::std::endl;

    exec_place::device(s2_dev)->*[&] {
      // Disable timing to avoid implicit barriers
      cudaEvent_t sync_event;
      cuda_safe_call(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));
      cuda_safe_call(cudaEventRecord(sync_event, s2));

      // According to documentation "event may be from a different device than stream."
      cuda_safe_call(cudaStreamWaitEvent(s1, sync_event, 0));

      // Asynchronously destroy event to avoid a memleak
      cuda_safe_call(cudaEventDestroy(sync_event));
    };
  }

  void insert_event()
  {
    // If needed, compute the underlying device
    if (dstream.dev_id == -1)
    {
      dstream.dev_id = get_device_from_stream(dstream.stream);
    }

    // Save the current device
    exec_place::device(dstream.dev_id)->*[&] {
      // Disable timing to avoid implicit barriers
      cuda_safe_call(cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming));
      // fprintf(stderr, "CREATE EVENT %p %s\n", cudaEvent, get_symbol().c_str());
      assert(cudaEvent);
      cuda_safe_call(cudaEventRecord(cudaEvent, dstream.stream));
    };
  }

  void insert_dep(async_resources_handle& async_resources, const stream_and_event& from)
  {
    // Otherwise streams will enforce dependencies
    if (dstream.stream != from.dstream.stream)
    {
      bool skip = async_resources.validate_sync_and_update(dstream.id, from.dstream.id, int(from.unique_prereq_id));
      if (!skip)
      {
        cuda_safe_call(cudaStreamWaitEvent(dstream.stream, from.cudaEvent, 0));
      }
    }
  }

  /**
   * @brief Remove implicit dependencies already induced by more recent events using the same stream.
   */
  bool factorize(backend_ctx_untyped&, reserved::event_vector& events) override
  {
    assert(events.size() >= 2);
    assert([&] {
      for (const auto& e : events)
      {
        assert(dynamic_cast<const stream_and_event*>(e.operator->()));
      }
      return true;
    }());

    static_assert(sizeof(reserved::event_vector) == sizeof(reserved::stream_and_event_vector));
    auto& proxy = reinterpret_cast<reserved::stream_and_event_vector&>(events);

    if (events.size() == 2)
    {
      // Specialize for two entries
      auto& a = proxy[0];
      auto& b = proxy[1];

      if (a->dstream.stream == b->dstream.stream)
      {
        // Must remove one element, keep the one with the largest id
        if (a->unique_prereq_id < b->unique_prereq_id)
        {
          a = mv(b);
        }
        proxy.pop_back();
      }

      return true;
    }

    // Sort events on stream (ascending) then id (descending)
    ::std::sort(proxy.begin(), proxy.end(), [](const auto& a, const auto& b) {
      return a->dstream.stream != b->dstream.stream
             ? a->dstream.stream < b->dstream.stream
             : a->unique_prereq_id > b->unique_prereq_id;
    });

    // Remove duplicates. Two events are duplicates if they have the same stream.
    // Will keep the first element of each duplicate run, which is the one with the largest id.
    proxy.erase(unstable_unique(proxy.begin(),
                                proxy.end(),
                                [](const auto& a, const auto& b) {
                                  return a->dstream.stream == b->dstream.stream;
                                }),
                proxy.end());

    return true;

    // // For each CUDA stream, we only keep the most recent event. The `int` is the unique event ID.
    // ::std::unordered_map<cudaStream_t, ::std::pair<int, event*>> m;

    // // For each element, compare it with existing entries of the map, keep only the most recent ones
    // for (auto& e: events) {
    //     auto se = reserved::handle<stream_and_event>(e, reserved::use_dynamic_cast);
    //     cudaStream_t stream = se->stream;
    //     int id = se->unique_prereq_id;

    //     auto iterator = m.find(stream);
    //     if (iterator != m.end()) {
    //         int previous_id = iterator->second.first;
    //         // Equality is possible if we have redundant entries, but then
    //         // we drop redundant values
    //         if (id > previous_id) {
    //             iterator->second = ::std::make_pair(id, &e);
    //         }
    //     } else {
    //         m[stream] = ::std::make_pair(id, &e);
    //     }
    // }

    // // Create a new container with the remaining elements and swap it with the old one
    // ::std::vector<event> new_events;
    // new_events.reserve(m.size());
    // for (const auto& e: m) {
    //     new_events.push_back(mv(*e.second.second));
    // }

    // events.swap(new_events);

    // return true;
  }

  void sync_with_stream(backend_ctx_untyped& bctx, event_list& prereqs, cudaStream_t stream) const override
  {
    reserved::join_with_stream(bctx, decorated_stream(stream), prereqs, "sync", false);
  }

  cudaStream_t get_stream() const
  {
    return dstream.stream;
  }

  decorated_stream get_decorated_stream() const
  {
    return dstream;
  }

  ::std::ptrdiff_t get_stream_id() const
  {
    return dstream.id;
  }

  cudaEvent_t get_cuda_event() const
  {
    return cudaEvent;
  }

private:
  decorated_stream dstream;
  cudaEvent_t cudaEvent = nullptr;
};

/**
 * @brief This implements an asynchronous operation synchronized by the means of stream-based events
 */
class stream_async_op
{
public:
  stream_async_op() = default;

  stream_async_op(backend_ctx_untyped& bctx, decorated_stream dstream, event_list& prereq_in)
      : dstream(mv(dstream))
  {
    setup(bctx, prereq_in);
  }

  // Async operation associated to a data place, we automatically get a stream attached to that place
  stream_async_op(backend_ctx_untyped& bctx, const data_place& place, cudaStream_t* target_stream, event_list& prereq_in)
  {
    int stream_dev_id = device_ordinal(place);

    if (stream_dev_id >= 0)
    {
      // We try to look for a stream that is already on the same device as this will require less synchronization
      dstream = device_lookup_in_event_list(bctx, prereq_in, stream_dev_id);
    }

    if (dstream.stream == nullptr)
    {
      // We did not select a stream yet, so we take one in the pools in
      // the async_resource_handle object associated to the context
      dstream = place.getDataStream(bctx.async_resources());
    }

    // Note that if we had stream_dev_id = -1 (eg. host memory), the device
    // id of this decorated stream will disagree, as we have taken one
    // stream from any device (current device, in particular)
    assert(dstream.stream);

    if (target_stream)
    {
      *target_stream = dstream.stream;
    }

    setup(bctx, prereq_in);
  }

  void set_symbol(::std::string s)
  {
    symbol = mv(s);
  }

  // Make sure all operations are done in the stream, and insert an event. A CUDASTF event is returned
  event end_as_event(backend_ctx_untyped& bctx)
  {
    /* Create an event that synchronize with all pending work in the CUDA stream */
    event e = event(reserved::handle<stream_and_event>(dstream, true));

    if (!symbol.empty())
    {
      e->set_symbol(bctx, mv(symbol));
    }

    auto& dot = *bctx.get_dot();
    if (dot.is_tracing_prereqs())
    {
      for (int id : joined_ids)
      {
        dot.add_edge(id, e->unique_prereq_id, reserved::edge_type::prereqs);
      }
    }

    return e;
  }

  // Make sure all operations are done in the stream, and insert an event. A
  // list of CUDASTF events (with a single entry) is returned.
  event_list end(backend_ctx_untyped& bctx)
  {
    return event_list(end_as_event(bctx));
  }

private:
  // Initialize the async operation so that the stream waits for input events
  void setup(backend_ctx_untyped& bctx, event_list& prereq_in)
  {
    // Make sure we reduce the number of resulting stream/event synchronization
    // API calls to a minimum. If the list was already optimized, this will be a no-op
    prereq_in.optimize(bctx);

    // Do we have to keep track of dependencies in DOT ?
    bool tracing = bctx.get_dot()->is_tracing_prereqs();

    // Make sure any operation in the stream depends on prereq_in
    for (const auto& e : prereq_in)
    {
      assert(dynamic_cast<stream_and_event*>(e.operator->()));
      auto se = reserved::handle<stream_and_event>(e, reserved::use_static_cast);

      if (dstream.stream != se->get_stream())
      {
        bool skip =
          bctx.async_resources().validate_sync_and_update(dstream.id, se->get_stream_id(), se->unique_prereq_id);
        if (!skip)
        {
          cuda_safe_call(cudaStreamWaitEvent(dstream.stream, se->get_cuda_event(), 0));
        }
      }
      se->outbound_deps++;

      if (tracing)
      {
        joined_ids.push_back(se->unique_prereq_id);
      }
    }
  }

  /* Find is there is already a stream associated to that device in the
   * prereq list */
  static decorated_stream device_lookup_in_event_list(backend_ctx_untyped& /* bctx */, event_list& prereq_in, int devid)
  {
    if (reserved::cached_getenv("CUDASTF_NO_LOOKUP"))
    {
      return decorated_stream(nullptr);
    }

    for (const auto& e : prereq_in)
    {
      cudaStream_t stream;
      ::std::ptrdiff_t stream_id = -1;
      auto se   = reserved::handle<stream_and_event, reserved::handle_flags::non_null>(e, reserved::use_static_cast);
      stream    = se->get_stream();
      stream_id = se->get_stream_id();

      // Find the stream structure in the driver API
      auto stream_driver = CUstream(stream);
      CUcontext ctx;
      cuda_safe_call(cuStreamGetCtx(stream_driver, &ctx));

      CUdevice stream_dev;
      cuda_safe_call(cuCtxPushCurrent(ctx));
      cuda_safe_call(cuCtxGetDevice(&stream_dev));
      cuda_safe_call(cuCtxPopCurrent(&ctx));

      if (stream_dev == devid)
      {
        //    fprintf(stderr, "Found matching device %d with stream %p\n", devid, stream);
        return decorated_stream(stream, stream_id, devid);
      }
    }

    return decorated_stream();
  }

  decorated_stream dstream;
  ::std::string symbol;

  // Used to display dependencies in DOT
  ::std::vector<int> joined_ids;
};

namespace reserved
{
/* This creates a synchronization point between all entries of the prereq_in list, and a CUDA stream */
inline event join_with_stream(
  backend_ctx_untyped& bctx, decorated_stream dstream, event_list& prereq_in, ::std::string string, bool record_event)
{
  // Make sure we reduce the number of resulting stream/event synchronization
  // API calls to a minimum. If the list was already optimized, this will be a no-op
  prereq_in.optimize(bctx);

  auto se = reserved::handle<stream_and_event>(mv(dstream), record_event);
  se->set_symbol(bctx, mv(string));
  join(bctx, *se, prereq_in);
  return se;
}

/* Create a simple event in a CUDA stream */
inline event record_event_in_stream(const decorated_stream& dstream)
{
  return reserved::handle<stream_and_event>(dstream, true);
}

/* Overload to provide a symbol */
inline event record_event_in_stream(const decorated_stream& dstream, reserved::per_ctx_dot& dot, ::std::string symbol)
{
  event res = record_event_in_stream(dstream);
  res->set_symbol_with_dot(dot, mv(symbol));
  return res;
}
} // end namespace reserved
} // namespace cuda::experimental::stf
