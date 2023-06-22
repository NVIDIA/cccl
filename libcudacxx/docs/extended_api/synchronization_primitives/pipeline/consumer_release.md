---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::consumer_release`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
void cuda::pipeline<Scope>::consumer_release();
```

Releases the current _pipeline stage_.

## Notes

If the calling thread is a _producer thread_, the behavior is undefined.

The pipeline is in a _quitted state_ (see [`cuda::pipeline::quit`]), the
  behavior is undefined.


[`cuda::pipeline::quit`]: ./quit.md

