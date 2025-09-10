.. _cub-developer-guide-device-scope:

Device-scope
*************

Overview
=========

Device-scope functionality is provided by classes called ``DeviceAlgorithm``,
where ``Algorithm`` is the implemented algorithm.
These classes then contain static member functions providing corresponding API entry points.

.. code-block:: c++

    struct DeviceAlgorithm {
      template <typename ...>
      CUB_RUNTIME_FUNCTION static cudaError_t Algorithm(
          void *d_temp_storage, size_t &temp_storage_bytes, ..., cudaStream_t stream = 0) {
        // optional: minimal argument checking or setup to call dispatch layer
        return DispatchAlgorithm<...>::Dispatch(d_temp_storage, temp_storage_bytes, ..., stream);
      }
    };

For example, device-level reduce will look like `cub::DeviceReduce::Sum`.
Device-scope facilities always return ``cudaError_t`` and accept ``stream`` as the last parameter (NULL stream by default)
and the first two parameters are always ``void *d_temp_storage, size_t &temp_storage_bytes``.
The implementation may consist of some minimal argument checking, but should forward as soon as possible to the dispatch layer.
Device-scope algorithms are implemented in files located in `cub/device/device_***.cuh`.

In general, the use of a CUB algorithm consists of two phases:

  1. Temporary storage size is calculated and returned in ``size_t &temp_storage_bytes``.
  2. ``temp_storage_bytes`` of memory is expected to be allocated and ``d_temp_storage`` is expected to be the pointer to this memory.

The following example illustrates this pattern:

.. code-block:: c++

    // First call: Determine temporary device storage requirements
    std::size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Second call: Perform algorithm
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

.. warning::
    Even if the algorithm doesn't need temporary storage as scratch space,
    we still require one byte of memory to be allocated.


Dispatch layer
====================================

A dispatch layer exists for each device-scope algorithms (e.g., `DispatchReduce`),
and is located in `cub/device/dispatch`.
Only device-scope algorithms have a dispatch layer.

The dispatch layer follows a certain architecture.
The high-level control flow is represented by the code below.
A more precise description is given later.

.. code-block:: c++

    // Device-scope API
    cudaError_t cub::DeviceAlgorithm::Algorithm(d_temp_storage, temp_storage_bytes, ...) {
      return DispatchAlgorithm::Dispatch(d_temp_storage, temp_storage_bytes, ...); // calls (1)
    }

    // Dispatch entry point
    static cudaError_t DispatchAlgorithm::Dispatch(...) { // (1)
      DispatchAlgorithm closure{...};
      // MaxPolicy - tail of linked list containing architecture-specific tunings
      return MaxPolicy::Invoke(get_device_ptx_version(), closure); // calls (2)
    }

    // Chained policy - linked list of tunings
    template <int PolicyPtxVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      using ActivePolicy = conditional_t<CUB_PTX_ARCH < PolicyPtxVersion, // (5)
                                        typename PrevPolicy::ActivePolicy, Policy>;

      static cudaError_t Invoke(int device_ptx_version, auto dispatch_closure) { // (2)
        if (device_ptx_version < PolicyPtxVersion) {
          return PrevPolicy::Invoke(device_ptx_version, dispatch_closure); // calls (2) of next policy
        }
        return dispatch_closure.Invoke<Policy>(); // eventually calls (3)
      }
    };

    // Dispatch object - a closure over all algorithm parameters
    template <typename Policy>
    cudaError_t DispatchAlgorithm::Invoke() { // (3)
        // host-side implementation of algorithm, calls kernels
        kernel<MaxPolicy><<<grid_size, Policy::AlgorithmPolicy::BLOCK_THREADS>>>(...); // calls (4)
    }

    template <typename ChainedPolicy>
    __launch_bounds__(ChainedPolicy::ActivePolicy::AlgorithmPolicy::BLOCK_THREADS) CUB_DETAIL_KERNEL_ATTRIBUTES
    void kernel(...) { // (4)
      using policy = ChainedPolicy::ActivePolicy; // selects policy of active device compilation pass (5)
      using agent = AgentAlgorithm<policy>; // instantiates (6)
      agent a{...};
      a.Process(); // calls (7)
    }

    template <typename Policy>
    struct AlgorithmAgent {  // (6)
      void Process() { ... } // (7)
    };

Let's look at each of the building blocks closer.

The dispatch entry point is typically represented by a static member function called ``DispatchAlgorithm::Dispatch``
that constructs an object of type ``DispatchAlgorithm``, filling it with all arguments to run the algorithm,
and passes it to the ``ChainedPolicy::Invoke`` function:

.. code-block:: c++

    template <..., // algorithm specific compile-time parameters
              typename PolicyHub>
    struct DispatchAlgorithm {
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static
      cudaError_t Dispatch(void *d_temp_storage, size_t &temp_storage_bytes, ..., cudaStream stream) {
        if (/* no items to process */) {
          if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
          }
          return cudaSuccess;
        }

        int ptx_version   = 0;
        const cudaError_t error = CubDebug(PtxVersion(ptx_version));
        if (cudaSuccess != error)
        {
          return error;
        }
        DispatchAlgorithm dispatch(..., stream);
        return CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      }
    };

For many legacy algorithms, the dispatch layer is publicly accessible and used directly by users,
since it often exposes additional performance knobs or configuration,
like choosing the index type or policies to use.
Exposing the dispatch layer also allowed users to tune algorithms for their use cases.
In the newly added algorithms, the dispatch layer should not be exposed publicly anymore.

The ``ChainedPolicy`` has two purposes.
During ``Invoke``, it converts the runtime PTX version of the current device
to the nearest lower-or-equal compile-time policy available:

.. code-block:: c++

    template <int PolicyPtxVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      using ActivePolicy = conditional_t<CUB_PTX_ARCH < PolicyPtxVersion,
                                        typename PrevPolicy::ActivePolicy, Policy>;

      template <typename Functor>
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
      static cudaError_t Invoke(int device_ptx_version, Functor dispatch_closure) {
        if (device_ptx_version < PolicyPtxVersion) {
          return PrevPolicy::Invoke(device_ptx_version, dispatch_closure);
        }
        return dispatch_closure.Invoke<Policy>();
      }
    };

The dispatch object's ``Invoke`` function is then called with the best policy for the device's PTX version:

.. code-block:: c++

    template <..., typename PolicyHub = detail::algorithm::policy_hub>
    struct DispatchAlgorithm {
      template <typename ActivePolicy>
      CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
      cudaError_t Invoke() {
        // host-side implementation of algorithm, calls kernels
        using MaxPolicy = typename DispatchSegmentedReduce::MaxPolicy;
        kernel<MaxPolicy /*(2)*/><<<grid_size, ActivePolicy::AlgorithmPolicy::BLOCK_THREADS /*(1)*/>>>(...); // calls (4)
      }
    };

This is where all the host-side work happens and kernels are eventually launched using the supplied policies.
Note how the kernel is instantiated on ``MaxPolicy`` (2) while the kernel launch configuration uses ``ActivePolicy`` (1).
This is an important optimization to reduce compilation-time:

.. code-block:: c++

    template <typename ChainedPolicy /* ... */ >
    __launch_bounds__(ChainedPolicy::ActivePolicy::AlgorithmPolicy::BLOCK_THREADS) __CUB_DETAIL_KERNEL_ATTRIBUTES
    void kernel(...) {
      using policy = ChainedPolicy::ActivePolicy::AlgorithmPolicy;
      using agent = AgentAlgorithm<policy>;

      __shared__ typename agent::TempStorage temp_storage; // allocate static shared memory for agent

      agent a{temp_storage, ...};
      a.Process();
    }

The kernel gets compiled for each PTX version (``N`` many) that was provided to the compiler.
During each device pass,
``ChainedPolicy`` compares ``CUB_PTX_ARCH`` against the template parameter ``PolicyPtxVersion``
to select an ``ActivePolicy`` type.
During the host pass,
``Invoke`` is compiled for each architecture in the tuning list (``M`` many).
If we used ``ActivePolicy`` instead of ``MaxPolicy`` as a kernel template parameter,
we would compile ``O(M*N)`` kernels instead of ``O(N)``.

The kernels in the dispatch layer shouldn't contain a lot of code.
Usually, the functionality is extracted into the agent layer.
All the kernel does is derive the proper policy type,
unwrap the policy to initialize the agent and call one of its ``Consume`` / ``Process`` functions.
Agents hold kernel bodies and are frequently reused by unrelated device-scope algorithms.

To gain a better understanding of why we use ``MaxPolicy`` instead of
``ActivePolicy`` to instantiate the kernel, consider the following example which
minimally reproduces the CUB dispatch layer.

.. code-block:: c++

    #include <cuda/std/type_traits>
    #include <cstdio>

    /// In device code, CUB_PTX_ARCH expands to the PTX version for which we are
    /// compiling. In host code, CUB_PTX_ARCH's value is implementation defined.
    #ifndef CUB_PTX_ARCH
    #  if !defined(__CUDA_ARCH__)
    #    define CUB_PTX_ARCH 0
    #  else
    #    define CUB_PTX_ARCH __CUDA_ARCH__
    #  endif
    #endif

    template <int PTXVersion, typename Policy, typename PrevPolicy>
    struct ChainedPolicy {
      static constexpr int cc = PTXVersion;
      using ActivePolicy      = ::cuda::std::conditional_t<CUB_PTX_ARCH<PTXVersion, PrevPolicy, Policy>;
      using PrevPolicyT       = PrevPolicy;
      using PolicyT           = Policy;
    };

    template <int PTXVersion, typename Policy>
    struct ChainedPolicy<PTXVersion, Policy, Policy> {
      static constexpr int cc = PTXVersion;
      using ActivePolicy      = Policy;
      using PrevPolicyT       = Policy;
      using PolicyT           = Policy;
    };

    struct sm60 : ChainedPolicy<600, sm60, sm60> { static constexpr int BLOCK_THREADS = 256; };
    struct sm70 : ChainedPolicy<700, sm70, sm60> { static constexpr int BLOCK_THREADS = 512; };
    struct sm80 : ChainedPolicy<800, sm80, sm70> { static constexpr int BLOCK_THREADS = 768; };
    struct sm90 : ChainedPolicy<900, sm90, sm80> { static constexpr int BLOCK_THREADS = 1024; };

    using MaxPolicy = sm90;

    // Kernel instantiated with ActivePolicy
    template <typename ActivePolicy>
    __launch_bounds__(ActivePolicy::BLOCK_THREADS) __global__ void bad_kernel() {
      if (threadIdx.x == 0) {
        std::printf("launch bounds %d; block threads %d\n", ActivePolicy::BLOCK_THREADS, blockDim.x);
      }
    }

    // Kernel instantiated with MaxPolicy
    template <typename MaxPolicy>
    __launch_bounds__(MaxPolicy::ActivePolicy::BLOCK_THREADS) __global__ void good_kernel() {
      if (threadIdx.x == 0) {
        std::printf("launch bounds %d; block threads %d\n", MaxPolicy::ActivePolicy::BLOCK_THREADS, blockDim.x);
      }
    }

    // Function to dispatch kernel with the correct ActivePolicy
    template <typename T>
    void invoke_with_best_policy_bad(int runtime_cc) {
      if (runtime_cc < T::cc) {
        invoke_with_best_policy_bad<typename T::PrevPolicyT>(runtime_cc);
      } else {
        bad_kernel<typename T::PolicyT><<<1, T::PolicyT::BLOCK_THREADS>>>();
      }
    }

    // Function to dispatch kernel with the correct MaxPolicy
    template <typename T>
    void invoke_with_best_policy_good(int runtime_cc) {
      if (runtime_cc < T::cc) {
        invoke_with_best_policy_good<typename T::PrevPolicyT>(runtime_cc);
      } else {
        good_kernel<MaxPolicy><<<1, T::PolicyT::BLOCK_THREADS>>>();
      }
    }

    void call_kernel() {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      int runtime_cc = deviceProp.major * 100 + deviceProp.minor * 10;
      std::printf("runtime cc %d\n", runtime_cc);

      invoke_with_best_policy_bad<MaxPolicy>(runtime_cc);
      invoke_with_best_policy_good<MaxPolicy>(runtime_cc);
    }

    int main() {
      call_kernel();
      cudaDeviceSynchronize();
    }

In this example, we define four execution policies for four GPU architectures,
``sm60``, ``sm70``, ``sm80``, and ``sm90``, with each having a different value
for ``BLOCK_THREADS``. Additionally, we define two kernels, ``bad_kernel`` and
``good_kernel`` to illustrate the effect of instantiating with the
``ActivePolicy`` and ``MaxPolicy`` respectively, as well a function to invoke
each of the kernels by selecting the policy for the appropriate architecture.
Finally, we call these two functions by specifying the current device's
architecture, which we obtain at run-time from host code. We cannot use the
``__CUDA_ARCH__`` macro here since it is not defined in host code, so it must be
extracted at run-time.

Compiling this file results in four template instantiations of ``bad_kernel``,
one for each architecture's policy, but only one template instantiation of
``good_kernel`` using ``MaxPolicy``. Assume we compiled the above file using
``nvcc policies.cu -std=c++20 -gencode arch=compute_80,code=sm_80 -o policies``,
we can inspect the generated binary to see these template instantiations using
``cuobjdump --dump-elf-symbols policies``. The relevant output is shown below.

.. code-block:: text

    symbols:
    STT_OBJECT       STB_LOCAL  STV_DEFAULT    $str
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z11good_kernelI4sm90Evv
    STT_FUNC         STB_GLOBAL STV_DEFAULT  U vprintf
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm90Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm80Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm70Evv
    STT_FUNC         STB_GLOBAL STO_ENTRY      _Z10bad_kernelI4sm60Evv

We can see that there are four symbols containing ``bad_kernel`` in their name,
which correspond to the four execution policies, but only one symbol for
``good_kernel``. This is due to the functions that invoke these kernels. For
``bad_kernel``, the template instantiation we call depends on the ``T::PolicyT``
(the ``ActivePolicy``), which depends on a run-time argument, so the compiler
must instantiate ``bad_kernel`` for every policy. For ``good_kernel``, we always
instantiate the kernel with ``MaxPolicy``. This results in only one template
instantiation compared to four template instantiations per fat binary, which
saves on compile time and binary size.

To show that both kernels' invocation is equivalent, we can run the code using
``./policies``. We obtain the following output.

.. code-block:: text

    runtime cc 890
    launch bounds 768; block threads 768
    launch bounds 768; block threads 768

Our GPU's architecture is ``sm89``, so the nearest policy less than or equal to
that, ``sm80`` in this case, is selected at run-time. Both kernels are invoked
with the same launch bounds and number of threads per block.

.. _cub-developer-policies:

Policies
====================================

Policies describe the configuration of agents wrt. to their execution.
They do not change functional behavior, but usually affect how work is mapped to the hardware by defining certain compile-time parameters (items per thread, block size, etc.).

An agent policy could look like this:

.. code-block:: c++

    template <int _BLOCK_THREADS,
              int _ITEMS_PER_THREAD,
              BlockLoadAlgorithm _LOAD_ALGORITHM,
              CacheLoadModifier _LOAD_MODIFIER>
    struct AgentAlgorithmPolicy {
      static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
      static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
      static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
      static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
      static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
    };

It's typically a collection of configuration values for the kernel launch configuration,
work distribution setting, load and store algorithms to use, as well as load instruction cache modifiers.
A CUB algorithm can have multiple agents and thus use multiple agent policies.

Since the device code of CUB algorithms is compiled for each PTX version, a different agent policy may be used.
Therefore, all agent policies of a CUB algorithm, called a policy, may be replicated for several minimum PTX versions.
A chained collection of such policies finally forms a policy hub:

.. code-block:: c++

    template <typename... TuningRelevantParams /* ... */>
    struct policy_hub {
      // TuningRelevantParams... could be used for decision making, like element types used, iterator category, etc.

      // for SM50
      struct Policy500 : ChainedPolicy<500, Policy500, Policy500> {
        using AlgorithmPolicy = AgentAlgorithmPolicy<256, 20, BLOCK_LOAD_DIRECT, LOAD_LDG>;
        // ... additional policies may exist, often one per agent
      };

      // for SM60
      struct Policy600 : ChainedPolicy<600, Policy600, Policy500> {
        using AlgorithmPolicy = AgentAlgorithmPolicy<256, 16, BLOCK_LOAD_DIRECT, LOAD_LDG>;
      };

      using MaxPolicy = Policy600; // alias where policy selection is started by ChainedPolicy
    };

The policy hub is a class template, possibly parameterized by tuning-relevant compile-time parameters,
containing a list of policies, one per minimum PTX version (i.e., SM architecture) they target.
These policies are chained by inheriting from ``ChainedPolicy``
and passing the minimum PTX version where they should be used,
as well as their own policy type and the next lower policy type.
An alias ``MaxPolicy`` serves as entry point into the chain of tuning policies.


Tunings
====================================

Because the values to parameterize an agent may vary a lot for different compile-time parameters,
the selection of values can be further delegated to tunings.
Often, such tunings are found by experimentation or heuristic search.
See also :ref:`cub-tuning`.

Tunings are usually organized as a class template, one per PTX version,
with a template specialization for each combination of the compile-time parameters,
for which better values for an agent policy are known.
An example set of tunings could look like this:

.. code-block:: c++

    template <int ValueSize, bool IsPlus>
    struct sm60_tuning { // default tuning
        static constexpr int threads = 128;
        static constexpr int items = 16;
    };

    template <>
    struct sm60_tuning<4, true> { // tuning for summing 4-byte values
        static constexpr int threads = 256;
        static constexpr int items = 20;
    };

    template <int ValueSize>
    struct sm60_tuning<ValueSize, true> { // tuning for summing values of other sizes
        static constexpr int threads = 128;
        static constexpr int items = 12;
    };

    ...

    template <typename ValueType, typename Operation>
    struct policy_hub {
      struct Policy600 : ChainedPolicy<600, Policy600, Policy500> {

        using tuning = sm60_tuning<sizeof(ValueType), is_same_v<Operation, plus>>;
        using AlgorithmPolicy = AgentAlgorithmPolicy<tuning::threads, tuning::items, BLOCK_LOAD_DIRECT, LOAD_LDG>;
      };
    };

Here, ``sm60_tuning`` provides defaults for the tuning values ``threads`` and ``items``.
``sm60_tuning`` is instantiated with the size of the value type and with a boolean indicating whether the operation is a sum.
Template specializations of ``sm60_tuning`` then provide different tuning values for summing value types of 4-byte size,
and for summing any other value types.
Notice how partial template specializations are used to pattern match the compile-time parameters.
Independent of which template specializations (or the base template) of the tuning is chose,
the agent policy is then parameterized by the nested ``threads`` and ``items`` values from this tuning.

The logic to select tunings varies, and different mechanisms are used for different algorithms.
Some algorithms provide a generic default policy if no tuning is available,
others implement a fallback logic to select the previous PTX version's agent policy,
if no tuning is available for the current PTX version.
In general, tunings are not exhaustive and usually only apply for specific combinations of parameter values and a single PTX version,
falling back to generic policies when no tuning matches.
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
