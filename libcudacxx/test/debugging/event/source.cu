// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/stream>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using event_ref_alias = cuda::event_ref;

[[gnu::noinline]] void inspect_ref(const cuda::event_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_owning(const cuda::event& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_timed(const cuda::timed_event& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_alias(const event_ref_alias& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_null_ref(const cuda::event_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_no_init(const cuda::event& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_timed_no_init(const cuda::timed_event& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::event_ref, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::event_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::event_ref& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  const auto ref_handle     = reinterpret_cast<::cudaEvent_t>(0x1234);
  const auto owning_handle  = reinterpret_cast<::cudaEvent_t>(0x2345);
  const auto timed_handle   = reinterpret_cast<::cudaEvent_t>(0x3456);
  const auto alias_handle   = reinterpret_cast<::cudaEvent_t>(0x4567);
  const auto nested_handle  = reinterpret_cast<::cudaEvent_t>(0x5678);
  const auto updated_handle = reinterpret_cast<::cudaEvent_t>(0x6789);
  const cuda::event_ref event_reference{ref_handle};
  cuda::event owning_event      = cuda::event::from_native_handle(owning_handle);
  cuda::timed_event timed_event = cuda::timed_event::from_native_handle(timed_handle);
  const event_ref_alias aliased_event{alias_handle};
  const cuda::event_ref null_reference{::cudaEvent_t{}};
  const cuda::event no_init_event{cuda::no_init};
  const cuda::timed_event no_init_timed_event{cuda::no_init};
  const cuda::std::array<cuda::event_ref, 2> nested_events = {
    cuda::event_ref{nested_handle}, cuda::event_ref{::cudaEvent_t{}}};
  cuda::event_ref updated_event{::cudaEvent_t{}};

  inspect_ref(event_reference);
  inspect_owning(owning_event);
  inspect_timed(timed_event);
  inspect_alias(aliased_event);
  inspect_null_ref(null_reference);
  inspect_no_init(no_init_event);
  inspect_timed_no_init(no_init_timed_event);
  inspect_nested(nested_events);
  inspect_before_update(updated_event);
  updated_event = cuda::event_ref{updated_handle};
  inspect_after_update(updated_event);

  static_cast<void>(owning_event.release());
  static_cast<void>(timed_event.release());
  return 0;
}
