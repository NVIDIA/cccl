---
grand_parent: Extended API
parent: Synchronization Primitives
---

# `cuda::pipeline_role`

Defined in header `<cuda/pipeline>`:

```cuda
enum class pipeline_role : /* unspecified */ {
  producer,
  consumer
};
```

`cuda::pipeline_role` specifies the role of a particular thread in a
  partitioned producer/consumer pipeline.

## Constants

| `producer` | A producer thread that generates data and issuing [asynchronous operations].                            |
| `consumer` | A consumer thread that consumes data and waiting for previously [asynchronous operations] to complete). |

## Example

See the [`cuda::make_pipeline` example].


[asynchronous operations]: ../asynchronous_operations.md

[`cuda::make_pipeline` example]: ./make_pipeline.md#example

