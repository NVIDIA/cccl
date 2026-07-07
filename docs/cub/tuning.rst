..
    TODO(bgruber): rename the label below to _cub-tuning when all tuning API exposure PRs have landed

.. _cub-policy-selectors:

CUB Tunings
================================================================================

Device-scope algorithms in CUB have many knobs that significantly impact performance (without affecting correctness).
For instance, the number of threads per block and items per thread can be tuned to maximize performance for a given device and data type.
But also algorithmic choices such as the load or store algorithm, a load vectorization size,
the used block-level algorithm, or enabling the use of the tensor memory accelerator can be tweaked.
Most device-scope algorithms in CUB accept a set of such tuning parameters,
by passing a policy selector, wrapped into :code:`cuda::execution::tune(...)`,
as part of the environment of a CUB device-scope API.
In the following, we describe this process in more detail.


Policy selectors
--------------------------------------------------------------------------------

Policy selectors are the mechanism by which CUB's device algorithms and kernels select
tuning parameters for a given workload and GPU compute capability.
A policy selector is a stateless callable that maps a :code:`cuda::compute_capability` to a policy struct
containing the tuning values for that compute capability.
Each set of CUB algorithms using a common underlying implementation defines a common policy struct,
e.g. :code:`cub::ReducePolicy`, which must be returned by a policy selector passed to those algorithms.

CUB employs internal default policy selectors providing tunings for known compute capabilities and workloads,
which are not publicly accessible to users.
Users can override CUB's policy selector for a given algorithm
by passing a custom policy selector through the algorithm's environment parameter.

For a description of how policy selectors are used internally in CUB's dispatch layer and kernels
see the corresponding :ref:`developer documentation <cub-developer-guide-device-scope>`.

Defining a policy selector
--------------------------------------------------------------------------------

A policy selector is any type with a :code:`__host__`,  :code:`__device__`, :code:`constexpr`, and :code:`const` call operator
taking a ``cuda::compute_capability`` and returning the algorithm's policy struct:

.. code:: c++

    template <typename T>
    struct my_reduce_tuning {
      __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::ReducePolicy {
        // tuning for Hopper and later
        if (cc >= cuda::compute_capability(9, 0)) {
          const auto pass = cub::ReducePassPolicy{
            .threads_per_block = 512,
            .items_per_thread = std::max(64 / sizeof(T), 1), // 8 double, 16 float, 32 half_t, ...
            .vec_size = 2,
            .reduce_algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
            .load_modifier = cub::LOAD_DEFAULT
          };
          return { .multi_tile = pass, .single_tile = pass };
        }
        // fallback for older GPUs
        const auto pass = cub::ReducePassPolicy{
          .threads_per_block = 256,
          .items_per_thread = 12,
          .vec_size = 1,
          .reduce_algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
          .load_modifier = cub::LOAD_DEFAULT
        };
        return { .multi_tile = pass, .single_tile = pass };
      }
    };

The policy selector must be stateless (:code:`std::is_empty_v<T>` must be :code:`true`)
since only its type will be passed to a kernel.
Policy selectors are also freely constructed and copied where needed,
so they must also be default constructible and copyable (:code:`std::semiregular<T>` must be :code:`true`).
It can be a class template, but then only a full specialization can be passed to CUB,
e.g., :code:`my_reduce_tuning<float>`.
The implementation can use branches and helper functions arbitrarily,
as long as they can be evaluated at compile-time.
The returned tuning policy can, and probably should, contain different values for different compute capabilities or workloads.

Each CUB algorithm with an environment parameter searches the environment for a policy selector returning a matching policy struct.
If one is found, the policy selector will be used to determine the tuning values for host-side dispatch and kernel compilation.
Each CUB algorithm documents the policy struct to which it responds.
If CUB does not find a matching policy selector in the environment, it falls back to its internal default policy selector.

Multiple policy selectors returning different policy structs can be passed as part of the same environment,
and each algorithm will pick the one with the matching policy struct.
This is useful if the same environment is reused across several algorithm calls.

The policy structs themselves are simple semiregular aggregates.
They support C++20 designated initializers (i.e., the syntax :code:`{ .threads_per_block = 512, ... }`),
comparison for (in-)equality, and serialization using :code:`operator<<`.
They may occasionally contain member functions that compute derived values from the contained tuning values.
All policy structs are public types and will evolve in a non-breaking way, at least during minor releases,
by only adding new data members at the end of the struct.


Passing a policy selector to CUB device-scope algorithms
--------------------------------------------------------------------------------

Custom policy selectors are passed to CUB algorithms via the environment argument.
They first need to be wrapped by passing them to :code:`cuda::execution::tune(...)`:

.. code:: c++

    cub::DeviceReduce::Reduce(
      d_in, d_out, num_items, op, init,
      cuda::execution::tune(my_reduce_tuning<int>{}));

Multiple tunings for different algorithms can be combined in a single environment:

.. code:: c++

    auto env = cuda::execution::tune(
      my_reduce_tuning<int>{},
      my_scan_tuning{}
    );
    cub::DeviceReduce::Reduce(d_in, d_out, num_items, op, init, env);
    cub::DeviceScan::ExclusiveSum(d_in, d_out, num_items, env);

An environment can carry further properties like a stream or a memory resource.
Policy selectors can simply be added to those:

.. code:: c++

    auto env = cuda::std::execution::env{
      stream_ref,
      resource,
      cuda::execution::tune(
        my_reduce_tuning<int>{}, my_scan_tuning{})
    };
    cub::DeviceReduce::Reduce(d_in, d_out, num_items, op, init, env);

    // alternatively, if we want to extend an env `other_env`
    auto env = cuda::std::execution::env{
      other_env,
      cuda::execution::tune(my_scan_tuning{})
    };
    cub::DeviceReduce::Reduce(d_in, d_out, num_items, op, init, env);

CUB's benchmarks also make heavy use of policy selectors for tuning.
For more details on authoring benchmarks and their policy selectors for automatic tuning, see :ref:`cub-tuning-infra`.
