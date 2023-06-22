---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::~pipeline`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
cuda::pipeline<Scope>::~pipeline();
```

Destructs the pipeline.
Calls [`cuda::pipeline::quit`] if it was not called by the current thread and
  destructs the pipeline.

[`cuda::pipeline::quit`]: ./quit.md

