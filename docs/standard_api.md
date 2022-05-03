---
has_children: true
has_toc: false
nav_order: 2
---

# Standard API

## Synchronization Library

Any Standard C++ header not listed below is omitted.

| [`<cuda/std/atomic>`]    | Atomic objects and operations (see also: [Extended API](./extended_api/synchronization_primitives/atomic.md)). <br/><br/> 1.0.0 / CUDA 10.2 |
| [`<cuda/std/latch>`]     | Single-phase asynchronous thread-coordination mechanism (see also: [Extended API](./extended_api/synchronization_primitives/latch.md)). <br/><br/> 1.1.0 / CUDA 11.0 |
| [`<cuda/std/barrier>`]   | Multi-phase asynchronous thread-coordination mechanism (see also: [Extended API](./extended_api/synchronization_primitives/barrier.md)). <br/><br/> 1.1.0 / CUDA 11.0 |
| [`<cuda/std/semaphore>`] | Primitives for constraining concurrent access (see also: [Extended API](./extended_api/synchronization_primitives/counting_semaphore.md)). <br/><br/> 1.1.0 / CUDA 11.0 |

{% include_relative standard_api/time_library.md %}

{% include_relative standard_api/numerics_library.md %}

{% include_relative standard_api/utility_library.md %}

## C Library

Any Standard C++ header not listed below is omitted.

| [`<cuda/std/cassert>`] | Lightweight assumption testing. <br/><br/> 1.0.0 / CUDA 10.2         |
| [`<cuda/std/cstddef>`] | Fundamental types. <br/><br/> 1.0.0 / CUDA 10.2 <br/> 1.4.0 (`byte`) |


[`<cuda/std/atomic>`]: https://en.cppreference.com/w/cpp/header/atomic
[`<cuda/std/latch>`]: https://en.cppreference.com/w/cpp/header/latch
[`<cuda/std/barrier>`]: https://en.cppreference.com/w/cpp/header/barrier
[`<cuda/std/semaphore>`]: https://en.cppreference.com/w/cpp/header/semaphore
[`<cuda/std/cassert>`]: https://en.cppreference.com/w/cpp/header/cassert
[`<cuda/std/cstddef>`]: https://en.cppreference.com/w/cpp/header/cstddef

