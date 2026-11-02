.. _cudax-coop-group-coalesced-group:

``cuda::experimental::coop::coalesced_group``
=============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Hierarchy>
  class coalesced_group : public /*group-interface-for*/<coalesced_group>
  {
  public:
    using unit_type       = thread_level;
    using level_type      = warp_level;
    using hierarchy_type  = Hierarchy;

    coalesced_group() = delete;

    template <typename HierarchyLike>
      requires cuda::std::same_as<Hierarchy, /*hierarchy-type-of*/<HierarchyLike>>
    __device__ explicit coalesced_group(const HierarchyLike& hierarchy_like) noexcept;

    coalesced_group(const coalesced_group&) = delete;

    coalesced_group(coalesced_group&&) = delete;

    coalesced_group& operator=(const coalesced_group&) = delete;

    coalesced_group& operator=(coalesced_group&&) = delete;
  };

  template </*hierarchy-like-type*/ HierarchyLike>
  coalesced_group(const HierarchyLike&) -> coalesced_group</*hierarchy-type-of*/<HierarchyLike>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::coalesced_group`` is a predefined group that groups all currently active threads within a warp. It is useful for opportunistic programming. The mapping result is determined based on the lane mask returned by the ``__activemask()`` intrinsic. The group can be explicitly constructed from a hierarchy-like object.

A *coalesced* group is not always contiguous.

A *coalesced* group uses the :ref:`cuda::experimental::coop::lane_synchronizer <cudax-coop-group-synchronizer-lane-synchronizer>` synchronizer for synchronization.

Queries
-------

Sub-unit queries work as with any other group type; however, since the *coalesced* group doesn't have a parent group, a native hierarchy level must be used in the super-level queries as the upper bound. For example, ``cuda::experimental::coop::coalesced_group{...}.count(cuda::block)`` queries the number of coalesced groups within a block. It is equivalent to the number of warps within a block.

Examples
--------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/stream>
  #include <cuda/hierarchy>
  #include <cuda/launch>
  #include <cuda/std/cassert>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __device__ void demo(auto config)
  {
    cudax::coop::this_warp warp{config};

    // When the whole warp takes the same path, coalesced_group and this_warp contain the same threads in the same order.
    {
      cudax::coop::coalesced_group coalesced{config};

      assert(cuda::gpu_thread.count(coalesced) == cuda::gpu_thread.count(warp));
      assert(cuda::gpu_thread.rank(coalesced) == cuda::gpu_thread.rank(warp));

      coalesced.sync();
      warp.sync();
    }

    // When we split the warp in half, the coalesced group shall contain only half of the threads.
    if (cuda::gpu_thread.rank(warp) < 16)
    {
      cudax::coop::coalesced_group coalesced{config};

      assert(cuda::gpu_thread.count(coalesced) == 16);
      assert(cuda::gpu_thread.rank(coalesced) < 16);

      coalesced.sync();
    }
  }

  struct Kernel
  {
    __device__ void operator()(auto config) const
    {
      demo(config);
    }
  };

  int main()
  {
    // Select a device and create a stream for it.
    const auto device = cuda::devices[0];
    cuda::stream stream{device};

    // Launch the test kernel on the stream.
    const auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<32>());
    cuda::launch(stream, config, Kernel{});

    // Wait until the kernel finishes.
    stream.sync();
  }

`See it on Godbolt 🔗 <https://godbolt.org/z/Kn17f5hbd>`__
