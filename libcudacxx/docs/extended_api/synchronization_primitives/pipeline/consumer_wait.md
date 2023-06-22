---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::consumer_wait`

Defined in header `<cuda/pipeline>`:

```cuda
// (1)
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline<Scope>::consumer_wait();

// (2)
template <cuda::thread_scope Scope>
template <typename Rep, typename Period>
__host__ __device__
bool cuda::pipeline<Scope>::consumer_wait_for(
  cuda::std::chrono::duration<Rep, Period> const& duration);

// (3)
template <cuda::thread_scope Scope>
template <typename Clock, typename Duration>
__host__ __device__
bool cuda::pipeline<Scope>::consumer_wait_until(
  cuda::std::chrono::time_point<Clock, Duration> const& time_point);
```

1. Blocks the current thread until all operations committed to the current
   _pipeline stage_ complete.
2. Blocks the current thread until all operations committed to the current
   _pipeline stage_ complete or after the specified timeout duration.
3. Blocks the current thread until all operations committed to the current
   _pipeline stage_ complete or until specified time point has been reached.

## Parameters

| `duration`   | An object of type `cuda::std::chrono::duration` representing the maximum time to spend waiting. |
| `time_point` | An object of type `cuda::std::chrono::time_point` representing the time when to stop waiting.   |

## Return Value

`false` if the _wait_ timed out, `true` otherwise.

## Notes

If the calling thread is a _producer thread_, the behavior is undefined.

The pipeline is in a _quitted state_ (see [`cuda::pipeline::quit`]), the
  behavior is undefined.


[`cuda::pipeline::quit`]: ./quit.md

