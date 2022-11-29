## Synchronization Primitives

### Atomics

| [`cuda::atomic`]             | System-wide [`cuda::std::atomic`] objects and operations. `(class template)`                                   <br/><br/> 1.0.0 / CUDA 10.2 |
| [`cuda::atomic_ref`]         | System-wide [`cuda::std::atomic_ref`] objects and operations. `(class template)`                               <br/><br/> 1.7.0 / CUDA 11.6 |

### Latches

| [`cuda::latch`]              | System-wide [`cuda::std::latch`] single-phase asynchronous thread coordination mechanism. `(class template)`   <br/><br/> 1.1.0 / CUDA 11.0 |

### Barriers

| [`cuda::barrier`]            | System-wide [`cuda::std::barrier`] multi-phase asynchronous thread coordination mechanism. `(class template)`  <br/><br/> 1.1.0 / CUDA 11.0 |

### Semaphores

| [`cuda::counting_semaphore`] | System-wide [`cuda::std::counting_semaphore`] primitive for constraining concurrent access. `(class template)` <br/><br/> 1.1.0 / CUDA 11.0 |
| [`cuda::binary_semaphore`]   | System-wide [`cuda::std::binary_semaphore`] primitive for mutual exclusion. `(class template)`                 <br/><br/> 1.1.0 / CUDA 11.0 |

### Pipelines

The pipeline library is included in the CUDA Toolkit, but is not part of the
  open source libcu++ distribution.

| [`cuda::pipeline`]                     | Coordination mechanism for sequencing asynchronous operations. `(class template)`                                       <br/><br/> CUDA 11.1 |
| [`cuda::pipeline_shared_state`]        | [`cuda::pipeline`] shared state object. `(class template)`                                                              <br/><br/> CUDA 11.1 |
| [`cuda::pipeline_role`]                | Defines producer/consumer role for a thread participating in a _pipeline_. `(enum)`                                     <br/><br/> CUDA 11.1 |
| [`cuda::make_pipeline`]                | Creates a [`cuda::pipeline`]. `(function template)`                                                                     <br/><br/> CUDA 11.1 |
| [`cuda::pipeline_consumer_wait_prior`] | Blocks the current thread until all operations committed up to a prior _pipeline stage_ complete. `(function template)` <br/><br/> CUDA 11.1 |
| [`cuda::pipeline_producer_commit`]     | Binds operations previously issued by the current thread to a [`cuda::barrier`]. `(function template)`                  <br/><br/> CUDA 11.1 |


[`cuda::std::atomic`]: https://en.cppreference.com/w/cpp/atomic/atomic
[`cuda::std::atomic_ref`]: https://en.cppreference.com/w/cpp/atomic/atomic_ref
[`cuda::std::barrier`]: https://en.cppreference.com/w/cpp/thread/barrier
[`cuda::std::latch`]: https://en.cppreference.com/w/cpp/thread/latch
[`cuda::std::counting_semaphore`]: https://en.cppreference.com/w/cpp/thread/counting_semaphore
[`cuda::std::binary_semaphore`]: https://en.cppreference.com/w/cpp/thread/binary_semaphore

[asynchronous operations]: {{ "extended_api/asynchronous_operations.html" | relative_url }}
[`cuda::memcpy_async`]: {{ "extended_api/asynchronous_operations/memcpy_async.html" | relative_url }}

[`cuda::atomic`]: {{ "extended_api/synchronization_primitives/atomic.html" | relative_url }}
[`cuda::atomic_ref`]: {{ "extended_api/synchronization_primitives/atomic_ref.html" | relative_url }}
[`cuda::barrier`]: {{ "extended_api/synchronization_primitives/barrier.html" | relative_url }}
[`cuda::latch`]: {{ "extended_api/synchronization_primitives/latch.html" | relative_url }}
[`cuda::counting_semaphore`]: {{ "extended_api/synchronization_primitives/counting_semaphore.html" | relative_url }}
[`cuda::binary_semaphore`]: {{ "extended_api/synchronization_primitives/binary_semaphore.html" | relative_url }}

[`cuda::pipeline`]: {{ "extended_api/synchronization_primitives/pipeline.html" | relative_url }}
[`cuda::pipeline_shared_state`]: {{ "extended_api/synchronization_primitives/pipeline_shared_state.html" | relative_url }}
[`cuda::pipeline_role`]: {{ "extended_api/synchronization_primitives/pipeline_role.html" | relative_url }}
[`cuda::make_pipeline`]: {{ "extended_api/synchronization_primitives/make_pipeline.html" | relative_url }}
[`cuda::pipeline_consumer_wait_prior`]: {{ "extended_api/synchronization_primitives/pipeline_consumer_wait_prior.html" | relative_url }}
[`cuda::pipeline_producer_commit`]: {{ "extended_api/synchronization_primitives/pipeline_producer_commit.html" | relative_url }}
