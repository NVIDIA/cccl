//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__functional/lazy_call_or.h>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include "testing.cuh"

enum class MyCoopAlgScratch
{
  none,
  smem,
  gmem,
  smem_gmem,
};

template <MyCoopAlgScratch Kind>
using MyCoopAlgScratchConstant = cuda::std::integral_constant<MyCoopAlgScratch, Kind>;

struct MyCoopAlgSmemScratch
{
  char data[128];
};

struct MyCoopAlgGmemScratch
{
  char data[128];
};

template <bool _False = false>
__device__ void my_coop_alg_impl(...)
{
  static_assert(_False, "This group type is unsupported by MyCoopAlg");
}

template <class Group, MyCoopAlgScratch Kind, class SmemScratch, class GmemScratch>
__device__ void my_coop_alg_impl(const Group&, MyCoopAlgScratchConstant<Kind>, SmemScratch&, GmemScratch&)
{
  // algorithm implementation
}

struct MyCoopAlg
{
  template <class Group, MyCoopAlgScratch Kind, class Env = cuda::std::execution::env<>>
  [[nodiscard]] __device__ static constexpr auto
  __get_scratch_requirements(const Group& group, MyCoopAlgScratchConstant<Kind>, Env = {}) noexcept
  {
    if constexpr (Kind == MyCoopAlgScratch::none)
    {
      return cudax::coop::__scratch_reqs<cudax::coop::__empty_smem_scratch, cudax::coop::__empty_gmem_scratch>{};
    }
    else if constexpr (Kind == MyCoopAlgScratch::smem)
    {
      return cudax::coop::__scratch_reqs<MyCoopAlgSmemScratch, cudax::coop::__empty_gmem_scratch>{};
    }
    else if constexpr (Kind == MyCoopAlgScratch::gmem)
    {
      return cudax::coop::__scratch_reqs<cudax::coop::__empty_smem_scratch, MyCoopAlgGmemScratch>{};
    }
    else
    {
      return cudax::coop::__scratch_reqs<MyCoopAlgSmemScratch, MyCoopAlgGmemScratch>{};
    }
  }

  template <class Group, MyCoopAlgScratch Kind, class Env = cuda::std::execution::env<>>
  __device__ void operator()(const Group& group, MyCoopAlgScratchConstant<Kind> kind, Env env = {}) const
  {
    // Get reference scratch requirements for this parameter combination.
    using ScratchReqs    = decltype(cudax::coop::get_scratch_requirements(MyCoopAlg{}, group, kind, env));
    using ExpSmemScratch = typename ScratchReqs::shared_memory_type;
    using ExpGmemScratch = typename ScratchReqs::global_memory_type;

    // Check that environment's smem and gmem scratch match the expected type.
    if constexpr (cuda::std::execution::__queryable_with<Env, cudax::coop::__get_smem_scratch_t>)
    {
      using QueryResult =
        cuda::std::remove_cvref_t<cuda::std::execution::__query_result_t<Env, cudax::coop::__get_smem_scratch_t>>;
      using EnvSmemScratch = typename QueryResult::type;
      static_assert(cuda::std::is_same_v<EnvSmemScratch, ExpSmemScratch>, "Invalid shared memory scratch passed");
    }
    if constexpr (cuda::std::execution::__queryable_with<Env, cudax::coop::__get_gmem_scratch_t>)
    {
      using QueryResult =
        cuda::std::remove_cvref_t<cuda::std::execution::__query_result_t<Env, cudax::coop::__get_gmem_scratch_t>>;
      using EnvGmemScratch = typename QueryResult::type;
      static_assert(cuda::std::is_same_v<EnvGmemScratch, ExpGmemScratch>, "Invalid global memory scratch passed");
    }

    // Extract environment's scratch or allocate default scratch.
    auto& smem_scratch =
      cuda::__lazy_call_or(
        cudax::coop::__get_smem_scratch,
        [&]() {
          return cudax::coop::__make_smem_scratch<ExpSmemScratch>(group, kind, env);
        },
        env)
        .get();
    auto& gmem_scratch =
      cuda::__lazy_call_or(
        cudax::coop::__get_gmem_scratch,
        [&]() {
          return cudax::coop::__make_gmem_scratch<ExpGmemScratch>(group, kind, env);
        },
        env)
        .get();

    // Pass scratch to algorithm implementation.
    my_coop_alg_impl(group, kind, smem_scratch, gmem_scratch);
  }
};

__device__ constexpr MyCoopAlg my_coop_alg;

__device__ cudax::coop::__empty_gmem_scratch my_empty_gmem_scratch;
__device__ MyCoopAlgGmemScratch my_gmem_scratch;

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    __shared__ cudax::coop::__empty_smem_scratch my_empty_smem_scratch;
    __shared__ MyCoopAlgSmemScratch my_smem_scratch;

    const cudax::this_thread group{config};

    // Test no scratch requirements.
    {
      const MyCoopAlgScratchConstant<MyCoopAlgScratch::none> kind{};
      using ScratchReqs = decltype(cudax::coop::get_scratch_requirements(my_coop_alg, group, kind));

      static_assert(cuda::std::is_same_v<typename ScratchReqs::shared_memory_type, cudax::coop::__empty_smem_scratch>);
      static_assert(cuda::std::is_same_v<typename ScratchReqs::global_memory_type, cudax::coop::__empty_gmem_scratch>);

      static_assert(!ScratchReqs::needs_shared_memory);
      static_assert(!ScratchReqs::needs_global_memory);

      static_assert(ScratchReqs::shared_memory_size == 0);
      static_assert(ScratchReqs::global_memory_size == 0);

      static_assert(ScratchReqs::shared_memory_alignment == 0);
      static_assert(ScratchReqs::global_memory_alignment == 0);

      // Test default environment.
      my_coop_alg(group, kind);

      // Test custom smem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_empty_smem_scratch)});

      // Test custom gmem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::global_memory_scratch(my_empty_gmem_scratch)});

      // Test custom smem and gmem scratch.
      my_coop_alg(group,
                  kind,
                  cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_empty_smem_scratch),
                                            cudax::coop::global_memory_scratch(my_empty_gmem_scratch)});
    }

    // Test smem scratch requirements.
    {
      const MyCoopAlgScratchConstant<MyCoopAlgScratch::smem> kind{};
      using ScratchReqs = decltype(cudax::coop::get_scratch_requirements(my_coop_alg, group, kind));

      static_assert(cuda::std::is_same_v<typename ScratchReqs::shared_memory_type, MyCoopAlgSmemScratch>);
      static_assert(cuda::std::is_same_v<typename ScratchReqs::global_memory_type, cudax::coop::__empty_gmem_scratch>);

      static_assert(ScratchReqs::needs_shared_memory);
      static_assert(!ScratchReqs::needs_global_memory);

      static_assert(ScratchReqs::shared_memory_size == sizeof(MyCoopAlgSmemScratch));
      static_assert(ScratchReqs::global_memory_size == 0);

      static_assert(ScratchReqs::shared_memory_alignment == alignof(MyCoopAlgSmemScratch));
      static_assert(ScratchReqs::global_memory_alignment == 0);

      // Test default environment.
      my_coop_alg(group, kind);

      // Test custom smem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_smem_scratch)});

      // Test custom gmem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::global_memory_scratch(my_empty_gmem_scratch)});

      // Test custom smem and gmem scratch.
      my_coop_alg(group,
                  kind,
                  cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_smem_scratch),
                                            cudax::coop::global_memory_scratch(my_empty_gmem_scratch)});
    }

    // Test gmem scratch requirements.
    {
      const MyCoopAlgScratchConstant<MyCoopAlgScratch::gmem> kind{};
      using ScratchReqs = decltype(cudax::coop::get_scratch_requirements(my_coop_alg, group, kind));

      static_assert(cuda::std::is_same_v<typename ScratchReqs::shared_memory_type, cudax::coop::__empty_smem_scratch>);
      static_assert(cuda::std::is_same_v<typename ScratchReqs::global_memory_type, MyCoopAlgGmemScratch>);

      static_assert(!ScratchReqs::needs_shared_memory);
      static_assert(ScratchReqs::needs_global_memory);

      static_assert(ScratchReqs::shared_memory_size == 0);
      static_assert(ScratchReqs::global_memory_size == sizeof(MyCoopAlgGmemScratch));

      static_assert(ScratchReqs::shared_memory_alignment == 0);
      static_assert(ScratchReqs::global_memory_alignment == alignof(MyCoopAlgGmemScratch));

      // Test default environment.
      my_coop_alg(group, kind);

      // Test custom smem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_empty_smem_scratch)});

      // Test custom gmem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::global_memory_scratch(my_gmem_scratch)});

      // Test custom smem and gmem scratch.
      my_coop_alg(group,
                  kind,
                  cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_empty_smem_scratch),
                                            cudax::coop::global_memory_scratch(my_gmem_scratch)});
    }

    // Test smem and gmem scratch requirements.
    {
      const MyCoopAlgScratchConstant<MyCoopAlgScratch::smem_gmem> kind{};
      using ScratchReqs = decltype(cudax::coop::get_scratch_requirements(my_coop_alg, group, kind));

      static_assert(cuda::std::is_same_v<typename ScratchReqs::shared_memory_type, MyCoopAlgSmemScratch>);
      static_assert(cuda::std::is_same_v<typename ScratchReqs::global_memory_type, MyCoopAlgGmemScratch>);

      static_assert(ScratchReqs::needs_shared_memory);
      static_assert(ScratchReqs::needs_global_memory);

      static_assert(ScratchReqs::shared_memory_size == sizeof(MyCoopAlgSmemScratch));
      static_assert(ScratchReqs::global_memory_size == sizeof(MyCoopAlgGmemScratch));

      static_assert(ScratchReqs::shared_memory_alignment == alignof(MyCoopAlgSmemScratch));
      static_assert(ScratchReqs::global_memory_alignment == alignof(MyCoopAlgGmemScratch));

      // Test default environment.
      my_coop_alg(group, kind);

      // Test custom smem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_smem_scratch)});

      // Test custom gmem scratch.
      my_coop_alg(group, kind, cuda::std::execution::env{cudax::coop::global_memory_scratch(my_gmem_scratch)});

      // Test custom smem and gmem scratch.
      my_coop_alg(group,
                  kind,
                  cuda::std::execution::env{cudax::coop::shared_memory_scratch(my_smem_scratch),
                                            cudax::coop::global_memory_scratch(my_gmem_scratch)});
    }
  }
};

C2H_TEST("scratch", "[scratch]")
{
  const auto device = cuda::devices[0];
  cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>());
  cuda::launch(stream, config, TestKernel{});
  stream.sync();
}
