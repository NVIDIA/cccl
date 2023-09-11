---
grand_parent: Extended API
parent: Synchronization Primitives
nav_order: 3
---

# `cuda::barrier`

Defined in header `<cuda/barrier>`:

```cuda
template <cuda::thread_scope Scope,
          typename CompletionFunction = /* unspecified */>
class cuda::barrier;
```

The class template `cuda::barrier` is an extended form of [`cuda::std::barrier`]
  that takes an additional [`cuda::thread_scope`] argument.
It has the same interface and semantics as [`cuda::std::barrier`], with the
  following additional operations.

## Barrier Operations

| [`cuda::barrier::init`]                 | Initialize a `cuda::barrier`. `(friend function)`                       |
| [`cuda::device::barrier_native_handle`] | Get the native handle to a `cuda::barrier`. `(function)`                |
| [`cuda::device::barrier_arrive_tx`]     | Arrive on a `cuda::barrier` with transaction count update. `(function)` |

## NVCC `__shared__` Initialization Warnings

When using libcu++ with NVCC, a `__shared__` `cuda::barrier` will lead to the
  following warning because `__shared__` variables are not initialized:

```
warning: dynamic initialization is not supported for a function-scope static
__shared__ variable within a __device__/__global__ function
```

It can be silenced using `#pragma nv_diag_suppress static_var_with_dynamic_init`.

To properly initialize a `__shared__` `cuda::barrier`, use the
  [`cuda::barrier::init`] friend function.

## Concurrency Restrictions

An object of type `cuda::barrier` or `cuda::std::barrier` shall not be accessed
  concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess` property] is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported` property] is 1.

Note, for objects of scopes other than `cuda::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

Under CUDA Compute Capability 8 (Ampere) or above, when an object of type
  `cuda::barrier<thread_scope_block>` is placed in `__shared__` memory, the
  member function `arrive` performs a reduction of the arrival count among
  [coalesced threads] followed by the arrival operation in one thread.
Programs shall ensure that this transformation would not introduce errors, for
  example relative to the requirements of [thread.barrier.class paragraph 12]
  of ISO/IEC IS 14882 (the C++ Standard).

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type
  `cuda::barrier` or `cuda::std::barrier` may not be used.

## Shared memory barriers with transaction count

In addition to the arrival count, a `cuda::barrier<thread_scope_block>` object
located in shared memory supports a
[tx-count](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tracking-asynchronous-operations-by-the-mbarrier-object),
which is used for tracking the completion of some asynchronous memory operations or
transactions.
The tx-count tracks the number of asynchronous transactions, in
units specified by the asynchronous memory operation (typically bytes), that are
outstanding and yet to be complete.
This capability is exposed, starting with the Hopper architecture (CUDA Compute
Capability 9).

The tx-count of `cuda::barrier` must be set to the total amount of asynchronous
memory operations, in units as specified by the asynchronous operations, to be
tracked by the current phase. This can be achieved with the
`cuda::device::barrier_arrive_tx` function call. Upon completion of each of the
asynchronous operations, the tx-count of the `cuda::barrier` will be updated and
thus progress the `cuda::barrier` towards the completion of the current phase.
This may complete the current phase.

### Phase Completion of a `cuda::barrier` with tx-count support

The requirements for completion of the current phase are described below. Upon
completion of the current phase, the phase transitions to the subsequent phase
as described below.

**Current phase completion requirements**. A `cuda::barrier` with tx-count
support completes the current phase when all of the following conditions are
met:

1. The count of the pending arrivals has reached zero.
2. The tx-count has reached zero.

**Phase transition**. When a `cuda::barrier` with tx-count support completes the
current phase, the following actions are performed atomically:

1. The `cuda::barrier` object transitions to the next phase.
2. The pending arrival count is reinitialized to the expected arrival count.

## Implementation-Defined Behavior

For each [`cuda::thread_scope`] `S` and `CompletionFunction` `F`, the value of
  `cuda::barrier<S, F>::max()` is as follows:

| [`cuda::thread_scope`] `S`     | `CompletionFunction` `F` | `barrier<S, F>::max()`                                   |
|--------------------------------|--------------------------|----------------------------------------------------------|
| `cuda::thread_scope_block`     | Default or user-provided | `(1 << 20) - 1`                                          |
| Not `cuda::thread_scope_block` | Default                  | `cuda::std::numeric_limits<cuda::std::int32_t>::max()`   |
| Not `cuda::thread_scope_block` | User-provided            | `cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()` |

## Example

```cuda
#include <cuda/barrier>

__global__ void example_kernel() {
  // This barrier is suitable for all threads in the system.
  cuda::barrier<cuda::thread_scope_system> a(10);

  // This barrier has the same type as the previous one (`a`).
  cuda::std::barrier<> b(10);

  // This barrier is suitable for all threads on the current processor (e.g. GPU).
  cuda::barrier<cuda::thread_scope_device> c(10);

  // This barrier is suitable for all threads in the same thread block.
  cuda::barrier<cuda::thread_scope_block> d(10);
}
```

[See it on Godbolt](https://godbolt.org/z/ehdrY8Kae){: .btn }


[`cuda::thread_scope`]: ../memory_model.md

[`cuda::barrier::init`]: ./barrier/init.md
[`cuda::device::barrier_native_handle`]: ./barrier/barrier_native_handle.md
[`cuda::device::barrier_arrive_tx`]: ./barrier/barrier_arrive_tx.md

[`cuda::std::barrier`]: https://en.cppreference.com/w/cpp/thread/barrier

[thread.barrier.class paragraph 12]: https://eel.is/c++draft/thread.barrier.class#12

[coalesced threads]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg

[`concurrentManagedAccess` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b
[`hostNativeAtomicSupported` property]: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f
