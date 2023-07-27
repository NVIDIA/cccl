---
parent: Extended API
nav_order: 1
---

# Thread Groups

```cuda
struct ThreadGroup {
  static constexpr cuda::thread_scope thread_scope;
  Integral size() const;
  Integral thread_rank() const;
  void sync() const;
};
```

The _ThreadGroup concept_ defines the requirements of a type that represents a
  group of cooperating threads.

The [CUDA Cooperative Groups Library] provides a number of types that satisfy
  this concept.

## Data Members

| `thread_scope` | The scope at which `ThreadGroup::sync()` synchronizes memory operations and thread execution. |

## Member Functions

| `size`        | Returns the number of participating threads.                                                                    |
| `thread_rank` | Returns a unique value for each participating thread (`0 <= ThreadGroup::thread_rank() < ThreadGroup::size()`). |
| `sync`        | Synchronizes the participating threads.                                                                         |

## Notes

This concept is defined for documentation purposes but is not materialized in
  the library.

## Example

```cuda
#include <cuda/atomic>
#include <cuda/std/cstddef>

struct single_thread_group {
  static constexpr cuda::thread_scope thread_scope = cuda::thread_scope::thread_scope_thread;
  cuda::std::size_t size() const { return 1; }
  cuda::std::size_t thread_rank() const { return 0; }
  void sync() const {}
};
```

[See it on Godbolt](https://godbolt.org/z/6c16KxqY7){: .btn }


[CUDA Cooperative Groups Library]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#group-collectives

