---
grand_parent: Extended API
parent: Synchronization Primitives
---

# `cuda::pipeline_shared_state`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
class cuda::pipeline_shared_state {
public:
  __host__ __device__
  pipeline_shared_state();

  ~pipeline_shared_state() = default;

  pipeline_shared_state(pipeline_shared_state const&) = delete;

  pipeline_shared_state(pipeline_shared_state&&) = delete;
};
```

The class template `cuda::pipeline_shared_state` is a storage type used to
  coordinate the threads participating in a `cuda::pipeline`.

## Template Parameters

| `Scope`       | A [`cuda::thread_scope`] denoting a scope including all the threads participating in the `cuda::pipeline`. `Scope` cannot be `cuda::thread_scope_thread`. |
| `StagesCount` | The number of stages for the _pipeline_.                                                                                                                  |

## Member Functions

| [(constructor)]                    | Constructs a `cuda::pipeline_shared_state`.      |
| (destructor) [implicitly declared] | Destroys the `cuda::pipeline_shared_state`. |
| `operator=` [deleted]              | `cuda::pipeline_shared_state` is not assignable. |

## NVCC `__shared__` Initialization Warnings

When using libcu++ with NVCC, a `__shared__` `cuda::pipeline_shared_state` will
  lead to the following warning because `__shared__` variables are not
  initialized:

```
warning: dynamic initialization is not supported for a function-scope static
__shared__ variable within a __device__/__global__ function
```

It can be silenced using `#pragma diag_suppress static_var_with_dynamic_init`.

## Example

```cuda
#include <cuda/pipeline>

// Disables `pipeline_shared_state` initialization warning.
#pragma diag_suppress static_var_with_dynamic_init

__global__ void example_kernel(char* device_buffer, char* sysmem_buffer) {
  // Allocate a 2 stage block scoped shared state in shared memory.
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss0;

  // Allocate a 2 stage block scoped shared state in device memory.
  auto* pss1 = new cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;

  // Construct a 2 stage device scoped shared state in device memory.
  auto* pss2 =
    new (device_buffer) cuda::pipeline_shared_state<cuda::thread_scope_device, 2>;

  // Construct a 2 stage system scoped shared state in system memory.
  auto* pss3 =
    new (sysmem_buffer) cuda::pipeline_shared_state<cuda::thread_scope_system, 2>;
}
```

[See it on Godbolt](https://godbolt.org/z/M9ah7r1Yx){: .btn }


[`cuda::thread_scope`]: ../memory_model.md

[(constructor)]: ./pipeline_shared_state/constructor.md
