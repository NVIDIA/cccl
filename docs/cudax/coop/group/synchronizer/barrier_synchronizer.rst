.. _cudax-coop-group-synchronizer-barrier-synchronizer:

``cuda::experimental::coop::barrier_synchronizer``
==================================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template </*cuda-barrier-type*/ Barrier, ::cuda::std::size_t N>
  class barrier_synchronizer
  {
    cuda::std::span<Barrier, N> /*barriers_*/; // exposition-only

  public:
    __device__ barrier_synchronizer(cuda::std::span<Barrier, N> barriers) noexcept
      : /*barriers_*/{barriers}
    {}

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const noexcept;
  };

  template <typename Barrier, cuda::std::size_t N>
  barrier_synchronizer(cuda::std::span<Barrier, N>) -> barrier_synchronizer<Barrier, N>;

  template <typename Container>
    requires /*is-spannable*/<Container>
  barrier_synchronizer(Container&)
    -> barrier_synchronizer</*span-element-type-of*/<Container>, /*span-element-extent-of*/<Container>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::barrier_synchronizer`` is a synchronizer that uses a ``cuda::barrier`` object to synchronize the group. The type is constructible from a contiguous range of uninitialized ``cuda::barrier`` objects, which are sequentially assigned to each of the created groups during the synchronizer instance creation. It can be used with any group for which the barriers have a sufficient scope.

Users should always rely on template argument deduction and never set the template arguments themselves.

Examples
--------

.. code:: cuda

  #include <cuda/barrier>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __device__ cuda::barrier<cuda::thread_scope_device> device_barriers[128];

  __global__ void kernel()
  {
    // Barrier synchronizer that uses an array of block-scope barriers placed in shared memory.
    __shared__ cuda::barrier<cuda::thread_scope_block> block_barriers[8];
    cudax::coop::barrier_synchronizer s1{block_barriers};

    // Barrier synchronizer that uses an array of device-scope barriers placed in global memory.
    cudax::coop::barrier_synchronizer s2{device_barriers};
  }
