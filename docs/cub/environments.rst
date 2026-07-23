.. _cub-environments:

Execution environments
================================================================================

..
    TODO(bgruber): We should generalize the std::execution::* parts into a new document in libcu++ and link to it here

Almost all CUB device-wide algorithms accept an execution environment as their last argument.
Environments are objects responding to queries which return values.
A "query" in this case is usually a (tag) type naming the query, for example `cuda::execution::__get_tuning_t`.
Environments can also be passed to certain customization point objects (CPOs),
which will extract values from the environment.
For example, the CPO `cuda::get_stream` will extract the stream from the given environment.
The purpose of an environment is to provide properties that govern the execution of an algorithm,
like the used CUDA stream, memory resource, required determinism, requested tuning, etc.

Some types are already environment themselves.
For example, `cudaStream_t` or `cuda::stream_ref` are environments responding to the `cuda::get_stream` CPO.
Similarly, several memory resource types are environments responding to the `cuda::mr::get_memory_resource` CPO.
Objects of such types can be passed directly to a CUB device-scope algorithm,
and the algorithm will query them for the stream or memory resource to use.
No wrapping of the value is necessary.


Constructing environments
--------------------------------------------------------------------------------

If the type of a value is not queryable directly,
a simple environment can be constructed using `cuda::std::execution::prop` given a value and a query.
For example:

.. code-block:: c++

    cudaStream_t stream = ...;
    auto env = cuda::std::execution::prop(cuda::get_stream, stream);

Builds an environment `env` that responds to the `cuda::get_stream` query with the value of `stream`.
If environments with more properties are needed,
they can be constructed from other environments using `cuda::std::execution::env`.
For example:

.. code-block:: c++

    cudaStream_t stream = ...;
    auto stream_env = cuda::std::execution::prop(cuda::get_stream, stream);
    cuda::mr::resource_ref<> mr = ...;
    auto mr_env = cuda::std::execution::prop{cuda::mr::get_memory_resource, cuda::mr::resource_ref<>{alloc}}
    auto env = cuda::std::execution::env(stream_env, mr_env);

Here, `env` is constructed from two other environments, `stream_env` and `mr_env`.
`env` now responds to both the `cuda::get_stream` and `cuda::mr::get_memory_resource` queries.
Notice that wrapping environments with `cuda::std::execution::env` does not nest them.

However, because `cudaStream_t` and `cuda::mr::resource_ref<>` are already environments themselves,
the above can be simplified to:

.. code-block:: c++

    cudaStream_t stream = ...;
    cuda::mr::resource_ref<> mr = ...;
    auto env = cuda::std::execution::env(stream, mr);

CUB also has a few more convenience functions for constructing environments,
like `cuda::execution::tune(...)` or `cuda::execution::require(...)`,
which build environments containing tuning policies and execution requirements, respectively.
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


Querying environments
--------------------------------------------------------------------------------

There are two ways to get a value out of an environment again.
Using the query (tag), we can query an environment directly using `cuda::std::execution::__query_or`:

.. code-block:: c++

    auto env = cuda::std::execution::env(stream, mr); // from above example
    cudaStream_t fallback_stream = ...;
    cudaStream_t stream = cuda::std::execution::__query_or(env, cuda::get_stream, fallback_stream);

`__query_or` will look for a value in the environment responding to the `cuda::get_stream` query.
If no such value is found, it will return the fallback value, `fallback_stream`, instead.

..
    TODO(bgruber) Does cuda::std::execution::__query_or(cudaStream_t{}, cuda::get_stream, fallback_stream); work?

...
    TODO(bgruber): Should we mention `__query_result_or_t`?

Alternatively, we can use the CPO `cuda::get_stream` to query an environment for a stream:

.. code-block:: c++

    auto env = cuda::std::execution::env(stream, mr); // from above example
    cudaStream_t stream = cuda::get_stream(env);

If `env` would not contain any value responding to the `cuda::get_stream` query, the above would not compile.
Using a CPO to query an environment is preferred over using `__query_or` in general.

CUB currently supports the following CPOs for querying environments:

 - `cuda::get_stream`
 - `cuda::mr::get_memory_resource`
 - `cuda::execution::__get_requirements_t` (internal)
 - `cuda::execution::__get_tuning_t` (internal)

While combining environments with `cuda::std::execution::env` does not nest them,
environments can contain values which are environments themselves,
and thus respond to queries again.
Execution requirements and tuning policy selectors are examples of values which are environments themselves.
For example:

.. code-block:: c++

    auto env = cuda::std::execution::env(cuda::execution::require(cuda::execution::determinism::run_to_run));
    auto requirements = cuda::__get_requirements_t(env);
    auto determinism = cuda::__get_requirements_t(requirements);
    static_assert(is_same_v<decltype(determinism), cuda::execution::determinism::run_to_run>):

..
    TODO(bgruber): Should we mention `__queryable_with`?
    TODO(bgruber): Should we mention `__call_or`?

Structure of the CUB execution environment
--------------------------------------------------------------------------------

In CUB, all device wide algorithms with an environment parameter `env` expect that:

 - If the environment contains a stream, it can be queried using `cuda::get_stream(env)`.
 - If the environment contains a memory resource, it can be queried using `cuda::mr::get_memory_resource(env)`.
 - If the environment contains execution requirements,
   they can be queried using `auto requirements = cuda::execution::__get_requirements_t(env)`.
   The value `requirements` is another environment, which can be queried for requirements,
   e.g. the requested determinism using `cuda::execution::__get_determinism_t`.
 - If the environment contains tuning policy selectors,
   they can be queried using `auto tunings = cuda::execution::__get_tuning_t(env)`.
   The value `tunings` is another environment, which can be queried for tuning policy selectors using
   `cuda::execution::__query_or` and the type of a tuning policy

See also the documentation on `determinism <cub-determinism>`_ and tuning <cub-policy-selectors>`_.


Properties of environments
--------------------------------------------------------------------------------

Since any object (of any type) can potentially be an environment,
we cannot describe the properties of environments in general.
However, in CUB we assume that environments are lightweight objects,
containing reference/pointer like values like stream or memory resource handles.
CUB therefore assumes in many places that an environment is copyable and movable.
We are currently changing this to pass environments between functions by `const&`,
to not rely on these properties.
Since CUB only queries environments, it does not mutate them, store them,
extend their lifetime in any way beyond a CUB API call,
or end their lifetime prematurely.
If an environment manages the lifetime of a contained resource,
and CUB queries and uses this resource for the execution of an algorithm,
it is the responsibility of the user to ensure the lifetime of the resource
exceeds the duration of the asynchronous CUB algorithm execution.
CUB strives for non-owning environments.
