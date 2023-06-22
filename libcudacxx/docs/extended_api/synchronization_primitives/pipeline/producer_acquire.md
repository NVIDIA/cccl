---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::producer_acquire`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline<Scope>::producer_acquire();
```

Blocks the current thread until the next _pipeline stage_ is available.

## Notes

If the calling thread is a _consumer thread_, the behavior is undefined.

The pipeline is in a _quitted state_ (see [`cuda::pipeline::quit`]), the
  behavior is undefined.


[`cuda::pipeline::quit`]: ./quit.md

