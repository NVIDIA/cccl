.. _cub-environments:

Execution environments
================================================================================

..
    TODO(bgruber): We should generalize the std::execution::* parts into a new document in libcu++ and link to it here

Almost all CUB device-wide algorithms accept an execution environment as their last argument.
Environments are objects responding to queries, returning values.
A "query" in this case is a type naming the query.
For example, `cuda::get_stream` can query an environment for a CUDA stream:

.. code-block:: c++

    auto env = ...;
    cudaStream_t stream = cuda::get_stream(env);

The purpose of an environment is to provide properties that govern the execution of an algorithm,
like the used CUDA stream, memory resource, required determinism, provided guarantees, requested tuning, etc.


Constructing environments for CUB algorithms
--------------------------------------------------------------------------------

If the type of a value is not queryable directly,
a simple environment can be constructed using `cuda::std::execution::prop` given a value and a query.
For example:

.. code-block:: c++

    auto mr = cuda::mr::resource_ref<>{allocator};
    auto mr_env = cuda::std::execution::prop{cuda::mr::get_memory_resource, mr}

Builds an environment `env` that responds to the `cuda::mr::get_memory_resource` query with the value of `mr`.

If environments with more properties are needed,
they can be constructed from other environments using `cuda::std::execution::env`.
For example:

.. code-block:: c++

    auto mr = cuda::mr::resource_ref<>{allocator};
    auto mr_env = cuda::std::execution::prop{cuda::mr::get_memory_resource, mr}
    cudaStream_t stream = ...;
    auto stream_env = cuda::std::execution::prop(cuda::get_stream, stream);
    auto env = cuda::std::execution::env(stream_env, mr_env);

Here, `env` is constructed from two other environments, `stream_env` and `mr_env`.
`env` now responds to both the `cuda::get_stream` and `cuda::mr::get_memory_resource` queries.
Notice that wrapping environments with `cuda::std::execution::env` does not nest them.


Implicit environments
--------------------------------------------------------------------------------

Some types are already environment themselves.
For example, `cudaStream_t` or `cuda::stream_ref` are already environments responding to the `cuda::get_stream` query.
Similarly, several memory resource types are environments responding to `cuda::mr::get_memory_resource`, etc.
Objects of such types can be passed directly to a CUB device-scope algorithm as environment,
and the algorithm will query them for the stream or memory resource to use.
No wrapping of the value is necessary.

So, because `cudaStream_t` and `cuda::mr::resource_ref<>` are already environments themselves,
the above example can be simplified to:

.. code-block:: c++

    cudaStream_t stream = ...;
    cuda::mr::resource_ref<> mr = ...;
    auto env = cuda::std::execution::env(stream, mr);

`env` again responds to both, the `cuda::get_stream` and `cuda::mr::get_memory_resource` queries.


Convenience functions
--------------------------------------------------------------------------------

CUB also has a few more convenience functions for constructing environments,
like `cuda::execution::tune(...)`, `cuda::execution::require(...)`, or  `cuda::execution::guarantee(...)`
which build environments containing tuning policies, execution requirements, or guarantees to the algorithm, respectively.
These environments can be freely composed with others again:

.. code-block:: c++

    cudaStream_t stream = ...;
    cuda::mr::resource_ref<> mr = ...;
    auto env = cuda::std::execution::env(stream, mr, cuda::execution::tune(...));
    auto env2 = cuda::std::execution::env(env, cuda::execution::require(...));

Here, `env` is an environment containing a stream, a memory resource, and tuning policies.
`env2` additionally contains an execution requirement.
A shorter way to define `env2` would be:

.. code-block:: c++

    cudaStream_t stream = ...;
    cuda::mr::resource_ref<> mr = ...;
    auto env2 = cuda::std::execution::env(
      stream, mr, cuda::execution::tune(...), cuda::execution::require(...));

`env2` is equivalent to the previous definition and can be queried for
a stream, a memory resource, tuning policies, and execution requirements.
