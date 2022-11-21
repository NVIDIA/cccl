---
has_children: true
has_toc: false
nav_order: 3
---

# Extended API

## Fundamentals

| [Thread Scopes]               | Defines the kind of threads that can synchronize using a primitive. `(enum)` <br/><br/> 1.0.0 / CUDA 10.2 |
| [Thread Groups]               | Concepts for groups of cooperating threads. `(concept)`                      <br/><br/> 1.2.0 / CUDA 11.1 |

{% include_relative extended_api/shapes.md %}

{% include_relative extended_api/synchronization_primitives.md %}

{% include_relative extended_api/asynchronous_operations.md %}

{% include_relative extended_api/memory_access_properties.md %}

{% include_relative extended_api/functional.md %}

[Thread Scopes]: ./extended_api/memory_model.md#thread-scopes
[Thread Groups]: ./extended_api/thread_groups.md

