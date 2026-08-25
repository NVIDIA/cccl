// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/expected>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3. MSVC does not accept
// GNU asm syntax, so it gets the same volatile-pointer-plus-barrier technique this
// repo's own DoNotOptimize (test/support/test_macros.h) already uses for its MSVC
// host path.
#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#  define KEEP_FOR_DEBUGGER(values) \
    do \
    { \
      [[maybe_unused]] void const* volatile keep_for_debugger_ptr = &(values); \
      _ReadWriteBarrier(); \
    } while (false)
#else
#  define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")
#endif

struct parse_error
{
  int code;
};

[[gnu::noinline]] void inspect_value(const cuda::std::expected<int, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

[[gnu::noinline]] void inspect_error(const cuda::std::expected<int, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

[[gnu::noinline]] void inspect_void_value(const cuda::std::expected<void, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

[[gnu::noinline]] void inspect_void_error(const cuda::std::expected<void, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::expected<int, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::expected<int, parse_error>& result)
{
  KEEP_FOR_DEBUGGER(result);
}

int main()
{
  const cuda::std::expected<int, parse_error> value(42);
  const cuda::std::expected<int, parse_error> error(cuda::std::unexpect, parse_error{7});
  const cuda::std::expected<void, parse_error> void_value{};
  const cuda::std::expected<void, parse_error> void_error(cuda::std::unexpect, parse_error{9});
  cuda::std::expected<int, parse_error> mutation(1);

  inspect_value(value);
  inspect_error(error);
  inspect_void_value(void_value);
  inspect_void_error(void_error);
  inspect_before_update(mutation);
  mutation.value() = 99;
  inspect_after_update(mutation);
}
