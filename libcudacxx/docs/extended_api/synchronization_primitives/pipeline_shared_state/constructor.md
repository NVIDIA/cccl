---
grand_parent: Pipelines
parent: cuda::pipeline_shared_state
---

# `cuda::pipeline_shared_state::pipeline_shared_state`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
__host__ __device__
cuda::pipeline_shared_state();

template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
cuda::pipeline_shared_state(cuda::pipeline_shared_state const&) = delete;

template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
cuda::pipeline_shared_state(cuda::pipeline_shared_state&&) = delete;
```

Construct a [`cuda::pipeline`] _shared state_ object.

## Example

```cuda
#include <cuda/pipeline>

#pragma nv_diag_suppress static_var_with_dynamic_init

__global__ void example_kernel() {
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;
}
```

[See it on Godbolt](https://godbolt.org/z/K4vKq4vd3){: .btn }


