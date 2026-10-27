.. _cudax-coop-group-synchronizer-barrier-synchronizer:

``cuda::experimental::coop::barrier_synchronizer``
==================================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename RangeView>
  class barrier_synchronizer
  {
    RangeView /*barriers_*/; // exposition-only

  public:
    template <typename Range>
      requires cuda::std::same_as<RangeView, cuda::std::ranges::views::all_t<Range>>
    __device__ barrier_synchronizer(Range&& barriers)
      noexcept(noexcept(cuda::std::ranges::views::all(cuda::std::forward<Range>(barriers))))
      : /*barriers_*/{cuda::std::ranges::views::all(cuda::std::forward<Range>(barriers))}
    {}

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const noexcept;
  };

  template <typename Range>
    requires cuda::std::ranges::viewable_range<Range>
      && cuda::std::ranges::sized_range<Range>
      && cuda::std::ranges::forward_range<Range>
      && cuda::std::ranges::borrowed_range<Range>
      && /*is-lvalue-cuda-barrier-reference*/<cuda::std::ranges::range_reference_t<Range>>
  barrier_synchronizer(Range&&)
    -> barrier_synchronizer<cuda::std::ranges::views::all_t<Range>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::barrier_synchronizer`` is a synchronizer that uses a ``cuda::barrier`` object to synchronize the group. The type is constructible from a viewable, sized, forward and borrowed range of uninitialized ``cuda::barrier`` objects. These barriers are initialized only once the ``make_instance(...)`` method is called.

The barrier at position ``i`` in the range is assigned to the group with rank ``i`` in the mapping result. The range must provide at least as many barriers as there are groups. The barriers must be accessible to all participating threads, and their scope must be sufficient for the parent group's level.

Users should always rely on template argument deduction and never set the template arguments themselves.

Examples
--------

.. code:: cuda

  #include <cuda/barrier>
  #include <cuda/std/ranges>
  #include <cuda/std/span>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __device__ cuda::barrier<cuda::thread_scope_device> device_barriers[128];

  __global__ void kernel()
  {
    // Barrier synchronizer that uses an array of block-scope barriers placed in shared memory.
    __shared__ cuda::barrier<cuda::thread_scope_block> block_barriers[8];
    cudax::coop::barrier_synchronizer s1{block_barriers};

    // Barrier synchronizer from a span of device-scope barriers placed in global memory.
    cudax::coop::barrier_synchronizer s2{cuda::std::span{device_barriers}};

    // A non-contiguous view assigns barriers in reverse order.
    cudax::coop::barrier_synchronizer s3{cuda::std::ranges::views::reverse(block_barriers)};
  }
