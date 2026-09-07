//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// `__logical_device` is not reachable from `<cuda/devices>`, so the internal headers are included
// directly here.
#include <cuda/__device/logical_device.h>
#include <cuda/__device/logical_device_ref.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/stream>

#include <testing.cuh>

namespace driver = cuda::__driver;

void recursive_check_device_setter(int id)
{
  int cudart_id;
  cuda::__ensure_current_context setter(cuda::device_ref{id});
  CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
  auto ctx = driver::__ctxGetCurrent();
  CUDART(cudaGetDevice(&cudart_id));
  CCCLRT_REQUIRE(cudart_id == id);

  if (id != 0)
  {
    recursive_check_device_setter(id - 1);

    CCCLRT_REQUIRE(test::count_driver_stack() == cuda::devices.size() - id);
    CCCLRT_REQUIRE(ctx == driver::__ctxGetCurrent());
    CUDART(cudaGetDevice(&cudart_id));
    CCCLRT_REQUIRE(cudart_id == id);
  }
}

C2H_TEST("ensure current context", "[device]")
{
  test::empty_driver_stack();
  // If possible use something different than CUDART default 0
  int target_device = static_cast<int>(cuda::devices.size() - 1);

  SECTION("context setter")
  {
    recursive_check_device_setter(target_device);

    CCCLRT_REQUIRE(test::count_driver_stack() == 0);
  }
}

C2H_CCCLRT_TEST("ensure current context from a stream", "[device][stream]")
{
  const auto device = cuda::devices[0];

  SECTION("The pushed context is the one the stream was created on")
  {
    cuda::stream str{device};

    cuda::__ensure_current_context setter(cuda::stream_ref{str.get()});

    CCCLRT_REQUIRE(driver::__ctxGetCurrent() == device.__primary_context());
    CCCLRT_REQUIRE(test::count_driver_stack() == 1);
  }

  SECTION("The context is popped again on destruction")
  {
    cuda::stream str{device};

    {
      cuda::__ensure_current_context setter(cuda::stream_ref{str.get()});
    }

    CCCLRT_REQUIRE(test::count_driver_stack() == 0);
  }

  SECTION("A stream on a second device pushes the context of that device")
  {
    if (cuda::devices.size() > 1)
    {
      const auto second = cuda::devices[1];
      cuda::stream str{second};

      cuda::__ensure_current_context setter(cuda::stream_ref{str.get()});

      CCCLRT_REQUIRE(driver::__ctxGetCurrent() == second.__primary_context());
      CCCLRT_REQUIRE(driver::__ctxGetCurrent() != device.__primary_context());
    }
    else
    {
      SUCCEED("The system has a single device");
    }
  }
}

// Green contexts require CTK 12.5.
#if _CCCL_CTK_AT_LEAST(12, 5)

C2H_CCCLRT_TEST("ensure current context from a green context stream", "[device][stream]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  const auto device = cuda::devices[0];
  const auto gctx   = driver::__greenCtxCreate(driver::__deviceGet(device.get()));
  auto ldev         = cuda::__logical_device::from_native_handle(device, gctx);
  cuda::stream str{ldev};

  const cuda::stream_ref ref{str.get()};

  SECTION("The constructor accepts a green context stream")
  {
    cuda::__ensure_current_context setter(ref);

    CCCLRT_REQUIRE(test::count_driver_stack() == 1);
  }

  SECTION("The pushed context is the green context, not the primary context")
  {
    // cuStreamGetCtx_v2() reports the green context of the stream. The pushed context is the
    // context of that green context, which is what __logical_device_ref also stores.
    cuda::__ensure_current_context setter(ref);

    CCCLRT_REQUIRE(driver::__ctxGetCurrent() == driver::__ctxFromGreenCtx(gctx));
    CCCLRT_REQUIRE(driver::__ctxGetCurrent() != device.__primary_context());
  }

  SECTION("The pushed context matches the logical device of the stream")
  {
    cuda::__ensure_current_context setter(ref);

    CCCLRT_REQUIRE(driver::__ctxGetCurrent() == ref.__logical_device().context());
  }

  SECTION("The pushed context resolves to the device of the green context")
  {
    // A green context resolves to the device that owns it.
    cuda::__ensure_current_context setter(ref);

    CCCLRT_REQUIRE(driver::__cudevice_to_ordinal(driver::__ctxGetDevice()) == device.get());
  }

  SECTION("The context is popped again on destruction")
  {
    {
      cuda::__ensure_current_context setter(ref);
    }

    CCCLRT_REQUIRE(test::count_driver_stack() == 0);
  }

  SECTION("A green context stream reports the device that owns the green context")
  {
    // Below CTK 13.0, device() pushes __cu_context() and asks cuCtxGetDevice(). The green
    // context must therefore resolve to its owner device.
    CCCLRT_REQUIRE(ref.device() == device);
  }
}

#endif // _CCCL_CTK_AT_LEAST(12, 5)
