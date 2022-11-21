---
grand_parent: Extended API
parent: Synchronization Primitives
---

# `cuda::pipeline`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
class cuda::pipeline {
public:
  pipeline() = delete;

  __host__ __device__ ~pipeline();

  pipeline& operator=(pipeline const&) = delete;

  __host__ __device__ void producer_acquire();

  __host__ __device__ void producer_commit();

  __host__ __device__ void consumer_wait();

  template <typename Rep, typename Period>
  __host__ __device__ bool consumer_wait_for(
    cuda::std::chrono::duration<Rep, Period> const& duration);

  template <typename Clock, typename Duration>
  __host__ __device__
  bool consumer_wait_until(
    cuda::std::chrono::time_point<Clock, Duration> const& time_point);

  __host__ __device__ void consumer_release();

  __host__ __device__ bool quit();
};
```

The class template `cuda::pipeline` provides a coordination mechanism which
  can sequence [asynchronous operations], such as [`cuda::memcpy_async`], into
  stages.

A thread interacts with a _pipeline stage_ using the following pattern:
1. Acquire the pipeline stage.
2. Commit some operations to the stage.
3. Wait for the previously committed operations to complete.
4. Release the pipeline stage.

For [`cuda::thread_scope`]s other than `cuda::thread_scope_thread`, a
  [`cuda::pipeline_shared_state`] is required to coordinate the participating
  threads.

_Pipelines_ can be either _unified_ or _partitioned_.
In a _unified pipeline_, all the participating threads are both producers and
  consumers.
In a _partitioned pipeline_, each participating thread is either a producer or
  a consumer.

## Template Parameters

| `Scope` | The scope of threads participating in the _pipeline_. |

## Member Functions

| (constructor) [deleted] | `cuda::pipeline` is not constructible.                                                                                                            |
| [(destructor)]          | Destroys the `cuda::pipeline`.                                                                                                                    |
| `operator=` [deleted]   | `cuda::pipeline` is not assignable.                                                                                                               |
| [`producer_acquire`]    | Blocks the current thread until the next _pipeline stage_ is available.                                                                           |
| [`producer_commit`]     | Commits operations previously issued by the current thread to the current _pipeline stage_.                                                       |
| [`consumer_wait`]       | Blocks the current thread until all operations committed to the current _pipeline stage_ complete.                                                |
| [`consumer_wait_for`]   | Blocks the current thread until all operations committed to the current _pipeline stage_ complete or after the specified timeout duration.        |
| [`consumer_wait_until`] | Blocks the current thread until all operations committed to the current _pipeline stage_ complete or until specified time point has been reached. |
| [`consumer_release`]    | Release the current _pipeline stage_.                                                                                                             |
| [`quit`]                | Quits current thread's participation in the _pipeline_.                                                                                           |

## Notes

A thread role cannot change during the lifetime of the pipeline object.

## Example

```cuda
#include <cuda/pipeline>
#include <cooperative_groups.h>

// Disables `pipeline_shared_state` initialization warning.
#pragma diag_suppress static_var_with_dynamic_init

template <typename T>
__device__ void compute(T* ptr);

template <typename T>
__global__ void example_kernel(T* global0, T* global1, cuda::std::size_t subset_count) {
  extern __shared__ T s[];
  auto group = cooperative_groups::this_thread_block();
  T* shared[2] = { s, s + 2 * group.size() };

  // Create a pipeline.
  constexpr auto scope = cuda::thread_scope_block;
  constexpr auto stages_count = 2;
  __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);

  // Prime the pipeline.
  pipeline.producer_acquire();
  cuda::memcpy_async(group, shared[0],
                     &global0[0], sizeof(T) * group.size(), pipeline);
  cuda::memcpy_async(group, shared[0] + group.size(),
                     &global1[0], sizeof(T) * group.size(), pipeline);
  pipeline.producer_commit();

  // Pipelined copy/compute.
  for (cuda::std::size_t subset = 1; subset < subset_count; ++subset) {
    pipeline.producer_acquire();
    cuda::memcpy_async(group, shared[subset % 2],
                       &global0[subset * group.size()],
                       sizeof(T) * group.size(), pipeline);
    cuda::memcpy_async(group, shared[subset % 2] + group.size(),
                       &global1[subset * group.size()],
                       sizeof(T) * group.size(), pipeline);
    pipeline.producer_commit();
    pipeline.consumer_wait();
    compute(shared[(subset - 1) % 2]);
    pipeline.consumer_release();
  }

  // Drain the pipeline.
  pipeline.consumer_wait();
  compute(shared[(subset_count - 1) % 2]);
  pipeline.consumer_release();
}

template void __global__ example_kernel<int>(int*, int*, cuda::std::size_t);
```

[See it on Godbolt](https://godbolt.org/z/zc41bWvja){: .btn }


[asynchronous operations]: ../asynchronous_operations.md
[`cuda::memcpy_async`]: ../asynchronous_operations/memcpy_async.md

[`cuda::thread_scope`]: ../memory_model.md
[`cuda::pipeline_shared_state`]: ./pipeline_shared_state.md

[(destructor)]: ./pipeline/destructor.md
[`producer_acquire`]: ./pipeline/producer_acquire.md
[`producer_commit`]: ./pipeline/producer_commit.md
[`consumer_wait`]: ./pipeline/consumer_wait.md
[`consumer_wait_for`]: ./pipeline/consumer_wait.md
[`consumer_wait_until`]: ./pipeline/consumer_wait.md
[`consumer_release`]: ./pipeline/consumer_release.md
[`quit`]: ./pipeline/quit.md
