//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include <cuda/barrier>

#include "cuda_space_selector.h"
#include "large_type.h"
#include "overrun_guard.h"

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class BarrierSelector,
    cuda::thread_scope BarrierScope,
    typename ...CompletionF
>
__host__ __device__ __noinline__
void test_fully_specialized()
{
    SourceSelector<overrun_guard<T>, constructor_initializer> source_sel;
    typename DestSelector<overrun_guard<T>, constructor_initializer>
        ::template offsetted<decltype(source_sel)::shared_offset> dest_sel;
    BarrierSelector<cuda::barrier<BarrierScope, CompletionF...>, constructor_initializer> bar_sel;

    overrun_guard<T> * source_guard = source_sel.construct(12);
    overrun_guard<T> * dest_guard = dest_sel.construct(0);
    cuda::barrier<BarrierScope, CompletionF...> * bar = bar_sel.construct(1);

    T * source = source_guard->get();
    T * dest = dest_guard->get();

    assert(*source_guard == 12);
    assert(*dest_guard == 0);

    cuda::memcpy_async(dest, source, sizeof(T), *bar);

    bar->arrive_and_wait();

    assert(*source_guard == 12);
    assert(*dest_guard == 12);

    *source = 24;

    cuda::memcpy_async(static_cast<void *>(dest), static_cast<void *>(source), sizeof(T), *bar);

    bar->arrive_and_wait();

    assert(*source_guard == 24);
    assert(*dest_guard == 24);
}

struct completion
{
    __host__ __device__
    void operator()() const {}
};

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector,
    template<typename, typename> class BarrierSelector
>
__host__ __device__ __noinline__
void test_select_scope()
{
    test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_system>();
    test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_device>();
    test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block>();
    // Test one of the scopes with a non-default completion. Testing them all would make this test take twice as much time to compile.
    // Selected block scope because the block scope barrier with the default completion has a special path, so this tests both that the
    // API entrypoints accept barriers with arbitrary completion function, and that the synchronization mechanism detects it correctly.
    test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_block, completion>();
    test_fully_specialized<T, SourceSelector, DestSelector, BarrierSelector, cuda::thread_scope_thread>();

}

template <class T,
    template<typename, typename> class SourceSelector,
    template<typename, typename> class DestSelector
>
__host__ __device__ __noinline__
void test_select_barrier()
{
    test_select_scope<T, SourceSelector, DestSelector, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,(
    test_select_scope<T, SourceSelector, DestSelector, shared_memory_selector>();
    test_select_scope<T, SourceSelector, DestSelector, global_memory_selector>();
  ))
}

template <class T,
    template<typename, typename> class SourceSelector
>
__host__ __device__ __noinline__
void test_select_destination()
{
    test_select_barrier<T, SourceSelector, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,(
    test_select_barrier<T, SourceSelector, shared_memory_selector>();
    test_select_barrier<T, SourceSelector, global_memory_selector>();
  ))
}

template <class T>
__host__ __device__ __noinline__
void test_select_source()
{
    test_select_destination<T, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,(
    test_select_destination<T, shared_memory_selector>();
    test_select_destination<T, global_memory_selector>();
  ))
}
