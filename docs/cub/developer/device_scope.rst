.. _cub-developer-guide-device-scope:

Device-scope
*************

Overview
=========

Device-scope functionality is provided by classes called ``DeviceAlgorithm``,
where ``Algorithm`` is the implemented algorithm.
These classes then contain static member functions providing corresponding API entry points.
For example, device-level reduce will look like `cub::DeviceReduce::Sum`.
Here is a generic example:

.. code-block:: c++

    struct DeviceAlgorithm {
      // two step API
      template <typename ...>
      static cudaError_t Algorithm(void *d_temp_storage, size_t &temp_storage_bytes, ..., cudaStream_t stream = 0) {
        // optional: minimal argument checking or setup to call dispatch layer
        return detail::algorithm::dispatch(d_temp_storage, temp_storage_bytes, ..., stream);
      }

      // environment API
      template <typename ..., typename Env = cuda::std::execution::env<>>
      static cudaError_t Algorithm(..., Env env = {}) {
        // optional: minimal argument checking or setup to call dispatch layer
        using default_policy_selector = detail::algorithm::policy_selector_from_types<...>;
        return dispatch_with_env_and_tuning<default_policy_selector>(
          env, [&](auto policy_selector, void* storage, size_t& bytes, auto stream) {
            return detail::algorithm::dispatch(d_temp_storage, temp_storage_bytes, ..., stream, policy_selector);
          });
      }
    };

Device-scope APIs come in two flavors, two step APIs and environment APIs,
both return ``cudaError_t`` and take algorithm specific arguments.
The two step API accepts a ``stream`` as the last parameter (``NULL`` stream by default)
and the first two parameters are always ``void *d_temp_storage, size_t &temp_storage_bytes``.
The environment API just takes an environment as the last parameter (empty environment by default).
The implementation may consist of some minimal argument checking, but should forward as soon as possible to the dispatch layer.
Device-scope algorithms are implemented in files located in `cub/device/device_***.cuh`.

The two step API is called in two phases:

  1. Temporary storage size is calculated and returned in ``size_t &temp_storage_bytes``.
  2. ``temp_storage_bytes`` of memory is expected to be allocated and ``d_temp_storage`` is expected to be the pointer to this memory.

The following example illustrates this pattern:

.. code-block:: c++

    // First call: Determine temporary device storage requirements
    std::size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(/*d_temp_storage*/ nullptr, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    thrust::device_vector<unsigned char> temp_storage(temp_storage_bytes, thrust::no_init);

    // Second call: Perform algorithm
    cub::DeviceReduce::Sum(temp_storage.data().get(), temp_storage_bytes, d_in, d_out, num_items);

.. warning::
    Even if the algorithm doesn't need temporary storage as scratch space,
    the overload with ``void *d_temp_storage, size_t &temp_storage_bytes``
    still requires one byte of memory to be allocated.

The environment overloads just require a single call, but setting up the environment may be more complex:

.. code-block:: c++

    // Setup environment, everything is optional
    auto env = cuda::std::execution::env{
      stream_ref,
      memory_resource,
      cuda::execution::require(requirements...),
      cuda::execution::tune(policy_selectors....)
    };

    // Perform algorithm
    cub::DeviceReduce::Sum(d_in, d_out, num_items, env);

The device layer handles the extraction of the:

* CUDA stream
* memory resource
* requirements
* guarantees (TODO(bgruber): are those public?)
* :ref:`tuning policy selectors <cub-policy-selectors>`

from the environment argument, or provide default values in case the environment does not contain them.
They typically use helper functions like ``dispatch_with_env`` and ``dispatch_with_env_and_tuning``.

Some CUB APIs require no temporary storage and may omit the ``void *d_temp_storage, size_t &temp_storage_bytes`` parameters.
Their environment overloads will also ignore any passed memory resource.

Dispatch layer
====================================

A dispatch function exists for each device-scope algorithm (e.g., ``detail::reduce::dispatch``),
and is located in ``cub/device/dispatch``.
Most device-scope algorithms share one dispatch function.
Only device-scope algorithms have dispatch functions, which are also referred to as the dispatch layer.

The dispatch layer follows a certain architecture.
The high-level control flow is represented by the code below.
A more precise description is given later.

..
    TODO(bgruber): consider removing the numbers below. control flow is now easier to follow

.. code-block:: c++

    // Device-scope API
    cudaError_t cub::DeviceAlgorithm::Algorithm(d_temp_storage, temp_storage_bytes, ...) {
      return detail::algorithm::dispatch(d_temp_storage, temp_storage_bytes, ...);
    }

    namespace detail::algorithm {
      cudaError_t dispatch(
          void *d_temp_storage, size_t &temp_storage_bytes,
          ...,
          cudaStream_t stream, PolicySelector policy_selector = {}) {
        cuda::compute_capability cc{};
        ptx_compute_cap(cc);
        const /*or constexpr*/ AlgorithmPolicy active_policy = policy_selector(cc);
        // host-side implementation of algorithm, calls kernels
        kernel<PolicySelector><<<grid_size, active_policy.threads_per_block>>>(...);
      }

      template <typename PolicySelector>
      __launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
      void kernel(...) {
        static constexpr auto policy = current_policy<PolicySelector>();
        using agent_policy = AgentPolicy<policy.threads_per_block, policy.items_per_thread, ...>;
        using agent = AgentAlgorithm<agent_policy, InputIteratorT, OffsetT, ...>;
        agent a{...};
        a.Process();
      }

      template <int ThreadsPerBlock, ...>
      struct AgentPolicy { // legacy
        static constexpr threads_per_block = ThreadsPerBlock;
        ...
      };

      template <typename Policy, ...>
      struct AlgorithmAgent {
        void Process() { ... }
      };
    }

Let's look at each of the building blocks closer.

The dispatch function
------------------------------------

The dispatch function is typically a simple function template called ``dispatch`` inside an algorithm-specific namespace.
It basically receives the same parameters as the public API entry point,
but they may have already been modified, extended, or generalized (to map many public APIs to the same dispatch function).
The goal of the dispatch function is to setup the execution of the algorithm on the device
by selecting the appropriate policy for the current GPU's compute capability,
preparing temporary storage and shared memory, configuring kernel launches, launching kernels, etc.

There are two style of dispatch functions, depending on whether the policy is needed as runtime or compile-time value:

.. code-block:: c++

    namespace detail::algorithm {
      // Dispatch version A - runtime policy
      cudaError_t dispatch(
          void *d_temp_storage, size_t &temp_storage_bytes,
          ...,
          cudaStream_t stream, PolicySelector policy_selector = {}) {
        cuda::compute_capability cc{};
        ptx_compute_cap(cc);
        const auto active_policy = policy_selector(cc); // runtime-time policy
        // host-side implementation of algorithm, calls kernels
        kernel<PolicySelector><<<grid_size, active_policy.threads_per_block>>>(...);
      }
    }

The dispatch function starts by querying the target compute capability for which compiled GPU code (PTX or SASS) is available,
by calling ``ptx_compute_cap``.
If the host code does not require the policy for this compute capability at compile-time (version A),
we can just pass the compute capability to the :ref:`policy selector <cub-policy-selectors>` to obtain the tuning policy at runtime.
The values from the policy are then used to setup resources and the kernel.
The kernel is then instantiated using only the type of the policy selector (not a concrete tuning policy),
so there is only one kernel instantiation across all target architectures compiled for.
More on that later.

If the host code needs the policy as a compile-time value (version B), we have to use ``dispatch_compute_cap``.

dispatch_compute_cap
------------------------------------

``dispatch_compute_cap`` maps a runtime ``cuda::compute_capability`` to a compile-time policy value
and calls the user-provided functor ``f`` with a nullary callable (a ``policy_getter``)
that returns the policy as a compile-time constant.
The policy getter is necessary to work around a C++17 limitation.

.. code-block:: c++

    namespace detail::algorithm {
      // Dispatch version B - compile-time policy
      cudaError_t dispatch(
          void *d_temp_storage, size_t &temp_storage_bytes,
          ...,
          cudaStream_t stream, PolicySelector policy_selector = {}) {
        cuda::compute_capability cc{};
        ptx_compute_cap(cc);
        return dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
          constexpr auto active_policy = policy_getter(); // compile-time policy
          static_assert(active_policy.tile_size() * sizeof(T) <= 48 * 1024, "Not enough SMEM");
          // host-side implementation of algorithm, calls kernels
          kernel<PolicySelector><<<grid_size, active_policy.threads_per_block>>>(...);
        });
      }

      template <typename PolicySelector, typename F>
      cudaError_t dispatch_compute_cap(PolicySelector, cuda::compute_capability cc, F&& f) { // (2)
        // fold over __CUDA_ARCH_LIST__, calling f with a policy_getter
        // that returns the policy for the matching arch as a compile-time value
      }
    }

Inside the lambda, ``policy_getter()`` returns the selected policy as a constant expression,
so compile-time branching (i.e. ``if constexpr``) or static assertions using policy values is possible.

Internally, ``dispatch_compute_cap`` uses ``__CUDA_ARCH_LIST__`` / ``NV_TARGET_SM_INTEGER_LIST`` (or all known compute capabilities as fallback)
to create one instantiation of ``f`` per distinct value of ``policy_selector(cc)`` (not per compute capability).
This results in one template instantiation of ``f`` per distinct policy value.
The kernel is then again only instantiated once using the type of the policy selector.


Kernels
------------------------------------

Kernels are templated on the ``PolicySelector`` type,
which is stateless and the same for all target architectures compiled for.

.. code-block:: c++

    namespace detail::algorithm {
      template <typename PolicySelector, typename InputIteratorT, typename OffsetT, /* ... */>
      __launch_bounds__(int(current_policy<PolicySelector>().reduce.threads_per_block))
      void DeviceReduceKernel(InputIteratorT d_in, OffsetT num_items, /* ... */) {
        static constexpr auto policy = current_policy<PolicySelector>();
        using agent_policy = AgentPolicy<policy.threads_per_block, policy.items_per_thread, ...>;
        using agent = AgentAlgorithm<agent_policy, InputIteratorT, OffsetT, ...>;
        agent a{...};
        a.Process();
      }

      template <typename PolicySelector>
      constexpr auto current_policy() {
        return PolicySelector{}(cuda::compute_capability{__CUDA_ARCH__ / 10}); // simplified
      }
    }

``PolicySelector`` must be stateless (``is_empty_v<PolicySelector>`` is ``true``),
so it can be default-constructed in device code where needed.
The utility function ``current_policy`` can only be called in device code.
It selects the target compute capability based on compiler macros of the current device compilation pass
and retrieves a tuning policy from the policy selector.

The kernel typically uses ``current_policy`` in two places,
to get the block size to define the launch bounds,
and inside the kernel to setup various sub algorithms (like agents).

Because C++17 does not allow to pass structs as Non-Type Template Parameters (NTTPs),
we cannot easily pass the policy around to other functions,
so it will be converted to legacy agent policies in many places.
Those are just structs with static data members holding the policy value as a type,
so they can be passed to templates.
Agent policy structs have historically been part of the public API, and should be removed in the future.

.. warning::
    The kernel gets compiled for each target architecture (N many) that was provided to the compiler.
    During each device pass, ``current_policy`` may return a different policy.
    During the host pass, version A (runtime policy) compiles a single instantiation of the dispatch logic for all target architectures.
    Version B (using ``dispatch_compute_cap``) compiles the dispatch logic for each distinct tuning policy (M many).
    If we passed the selected tuning policy instead of the policy selector as a kernel template parameter,
    the kernel template instantiation would be different for each tuning policy value and
    we would compile O(M*N) kernels for version B instead of O(N).

Many kernels are short, since the functionality is extracted into the agent layer.
All the kernel does is derive the proper policy,
unwrap it to initialize the agent and call one of its ``Consume`` / ``Process`` functions.
Agents hold kernel bodies and are intended to be reused across multiple device-scope algorithms.
However, this is not a requirement and some kernels just contain their entire implementation themselves.


The default policy selector
------------------------------------

CUB contains a default policy selector for each dispatch function.
Because many dispatch functions are also used by CCCL.C,
which compiles them without proper type information,
we have to provide them in two forms,
a typeless and a typeful version.

.. code-block:: c++

    struct AlgorithmPolicy { ... }; // unrelated to AgentPolicy

    namespace detail::algorithm {
      struct policy_selector {
        type_t accum_t;
        op_kind_t operation_t;
        int offset_size;
        int accum_size;

        constexpr auto operator()(::cuda::compute_capability cc) const -> AlgorithmPolicy {
          // parameter selection across target compute capabilities and input characteristics (possibly HUGE logic)
        }
      };

      template <typename AccumT, typename OffsetT, typename ReductionOpT>
      struct policy_selector_from_types {
        constexpr auto operator()(cuda::compute_capability cc) const -> AlgorithmPolicy {
          constexpr auto ps = policy_selector{
            classify_type<AccumT>(),
            classify_op<ReductionOpT>(),
            int{sizeof(OffsetT)},
            int{sizeof(AccumT)}
          };
          return ps(cc);
        }
      };
    }

The ``policy_selector`` is intended to be used without template-parameter-based type information
and is thus suitable to be used in CCCL.C without a JIT compiler for host code.
It contains the necessary information on the algorithm's input as data members.
This policy is only used in the host code of the dispatch function.
Because it is not stateless anymore, CCCL.C overrides the kernel launcher used by CUB,
providing the kernel from a JIT-compiled instantiation that uses a proper stateless policy selector.

The kernel, and the dispatch function when not called from CCCL.C,
will use the second version of the policy selector, ``policy_selector_from_types``,
which offers proper template parameters to pass type information.
This type is stateless and delegates to a ``constexpr`` instance of the ``policy_selector``
with compile-time values derived from the template parameters.
The policy selection logic is thus the same for CCCL.C and CUB,
the type information is just provided differently.

Each dispatch function also has an associated concept for the policy selector it expects:

.. code-block:: c++

    template <typename T, typename Policy>
    concept policy_selector = requires(T pol_sel, cuda::compute_capability cc) {
      requires std::regular<Policy>;
      { pol_sel(cc) } -> std::same_as<Policy>;
    };

    namespace detail::algorithm {
      template <typename T>
      concept algorithm_policy_selector = policy_selector<T, AlgorithmPolicy>;
    }

The concept basically checks whether the policy selector can be called with a ``cuda::compute_capability``
and returns the expected policy struct.
The default policy selectors and related concept are defined in ``cub/device/dispatch/tuning/tuning_<algorithm>.cuh``.


Backward compatibility
------------------------------------

Legacy public dispatchers (e.g. ``DispatchReduce``) are deprecated.
They continue to work by translating the ``PolicyHub`` template parameter to the new ``policy_selector``
via a ``policy_selector_from_hub`` adapter:

.. code-block:: c++

    template <typename PolicyHub>
    struct policy_selector_from_hub {
      constexpr auto operator()(cuda::compute_capability cc) const -> AlgorithmPolicy {
        // conversion logic
      }
    };

This allows existing user code that passes custom policy hubs to dispatchers to continue working.
The legacy dispatchers, policy hubs and related logic are scheduled for removal in CCCL 4.0.


Policies
====================================

Policies describe the configuration of agents or just kernels with respect to performance.
They must not change functional behavior, but affect how work is mapped to the hardware
by defining certain parameters (items per thread, block size, etc.),
or choosing between algorithms.

Policies must be plain semiregular aggregates to allow using them during constant evaluation,
and with designated initializers:

.. code-block:: c++

    struct AlgorithmPolicy {
      int threads_per_block;
      int items_per_thread;
      cub::BlockLoadAlgorithm load_algorithm;
      cub::CacheLoadModifier load_modifier;

      friend constexpr bool operator==(const AlgorithmPolicy& lhs, const AlgorithmPolicy& rhs) { ... }
      friend constexpr bool operator!=(const AlgorithmPolicy& lhs, const AlgorithmPolicy& rhs) { ... }
      friend std::ostream& operator<<(std::ostream& os, const AlgorithmPolicy& p) { ... }
    };

Tuning policies can have various complexities and contain nested structures.
Policies are defined in ``cub/device/dispatch/tuning/tuning_<algorithm>.cuh``.


Tunings
====================================

Because the values to parameterize an agent may vary a lot for different compile-time parameters,
the selection of values can involve complex logic.
Often, such tunings are found by experimentation or heuristic search.
See also :ref:`cub-tuning`.

Tunings are expressed as logic and values inside the ``constexpr operator()`` of a policy selector.
Because of the complexity of some policy selectors, nested functions may be used.
Many policy selectors also implement a fallback logic,
where they try to find a matching tuning based on the input characteristics (policy selector data members),
but if no match is found, they fall back to an older target compute capability.
Here is an example:

.. code-block:: c++

    struct sm90_tuning_values {
      int items;
      int threads;
      int items_per_vec_load;
    };

    constexpr auto get_sm90_tuning(type_t accum_t, op_kind_t op, int offset_size, int accum_size)
        -> std::optional<sm90_tuning_values> {
      // provide tunings for certain operations, accumulator or offset types
      if (op == op_kind_t::plus) {
        if (accum_t == type_t::float32 && offset_size == 4 && accum_size == 4)
          return sm90_tuning_values{16, 512, 2};
        if (offset_size == 4 && accum_size == 8)
          return sm90_tuning_values{15, 512, 2};
      }
      return {}; // no tuning available, causes fallback
    }

    struct policy_selector {
      type_t accum_t;
      op_kind_t operation_t;
      int offset_size;
      int accum_size;

      constexpr auto operator()(cuda::compute_capability cc) const -> reduce_policy {
        if (cc >= cuda::compute_capability{9, 0}) {
          if (auto tuning = get_sm90_tuning(accum_t, operation_t, offset_size, accum_size)) {
            return *tuning; // found a tuning, use it
          }
          // fall through to sm_80 if no matching tuning found
        }
        if (cc >= cuda::compute_capability{8, 0}) {
          return { /* sm_80 default policy */ };
        }
        return { /* default policy for everything else */ };
      }
    };

In general, tunings are not exhaustive and usually only apply for specific combinations
of parameter values and a single compute capability.
This is because they originate from tuning benchmarks running for specific workloads on specific target architectures.
Generic fallbacks are often just retained from earlier days of CUB to not risk regressions,
or are based on heuristics trying to provide reasonable performance based on a model of the GPU architecture or algorithm.

Tunings for CUB algorithms reside in ``cub/device/dispatch/tuning/tuning_<algorithm>.cuh``.


Temporary storage usage
====================================

It's safe to reuse storage in the stream order:

.. code-block:: c++

    cub::DeviceReduce::Sum(nullptr, storage_bytes, d_in, d_out, num_items, stream_1);
    // allocate temp storage
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_1);
    // fine not to synchronize stream
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_1);
    // illegal, should call cudaStreamSynchronize(stream)
    cub::DeviceReduce::Sum(d_storage, storage_bytes, d_in, d_out, num_items, stream_2);

Temporary storage management
====================================

Often times temporary storage for device-scope algorithms has a complex structure.
To simplify temporary storage management and make it safer,
we introduced ``cub::detail::temporary_storage::layout``:

.. code-block:: c++

    cub::detail::temporary_storage::layout<2> storage_layout;

    auto slot_1 = storage_layout.get_slot(0);
    auto slot_2 = storage_layout.get_slot(1);

    auto allocation_1 = slot_1->create_alias<int>();
    auto allocation_2 = slot_1->create_alias<double>(42);
    auto allocation_3 = slot_2->create_alias<char>(12);

    if (condition)
    {
      allocation_1.grow(num_items);
    }

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = storage_layout.get_size();
      return;
    }

    storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes);

    // different slots, safe to use simultaneously
    use(allocation_1.get(), allocation_3.get(), stream);
    // `allocation_2` alias `allocation_1`, safe to use in stream order
    use(allocation_2.get(), stream);
