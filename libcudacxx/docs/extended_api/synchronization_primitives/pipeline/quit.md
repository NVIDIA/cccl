---
grand_parent: Pipelines
parent: cuda::pipeline
---

# `cuda::pipeline::quit`

Defined in header `<cuda/pipeline>`:

```cuda
template <cuda::thread_scope Scope>
__host__ __device__
bool cuda::pipeline<Scope>::quit();
```

Quits the current thread's participation in the collective ownership of the
  corresponding [`cuda::pipeline_shared_state`].
Ownership of [`cuda::pipeline_shared_state`] is released by the last invoking
  thread.

## Return Value

`true` if ownership of the _shared state_ was released, otherwise `false`.

## Notes

After the completion of a call to `cuda::pipeline::quit`, no other operations
  other than [`cuda::pipeline::~pipeline`] may called by the current thread.


[`cuda::pipeline::~pipeline`]: ./destructor.md

[`cuda::pipeline_shared_state`]: ../pipeline_shared_state.md

