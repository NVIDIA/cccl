.. _cudax-coop-group-group-view:

``cuda::experimental::coop::group_view``
========================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Unit, typename Group>
  class group_view : public /*group-interface-for*/<group_view>
  {
  public:
    using unit_type       = Unit;
    using level_type      = typename Group::level_type;
    using hierarchy_type  = typename Group::hierarchy_type;

    group_view() = delete;

    __device__ group_view(const Group& group) noexcept
      requires cuda::std::same_as<Unit, typename Group::unit_type>;

    __device__ explicit group_view(const Unit& unit, const Group& group) noexcept;

    template <typename OtherUnit>
      requires /*unit-same-as-or-below*/<Unit, OtherUnit>
    __device__ explicit group_view(const Unit& unit, const group_view<OtherUnit, Group>& other) noexcept;

    group_view(const group_view&) = default;

    group_view(group_view&&) = default;

    group_view& operator=(const group_view&) = default;

    group_view& operator=(group_view&&) = default;
  };

  template <group Group>
  group_view(const Group&) -> group_view<typename Group::unit_type, Group>;

  template <typename Unit, class Group>
  group_view(const group_view<Unit, Group>&) -> group_view<Unit, Group>;

  template </*hierarchy-level-type*/ Unit, group Group>
  group_view(const Unit&, const Group&) -> group_view<Unit, Group>;

  template </*hierarchy-level-type*/ Unit, class OtherUnit, class Group>
    requires /*unit-same-as-or-below*/<Unit, OtherUnit>
  group_view(const Unit&, const group_view<OtherUnit, Group>&) -> group_view<Unit, Group>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::group_view`` provides a non-owning view over another group that also allows you to view a group as if it had its ``unit_type`` changed. These properties can be useful when you need to pass a group by value or need more fine-grained rank addressing.

``cuda::experimental::coop::group_view`` has the same mapping-related properties as the viewed group, unless the unit type was changed. In that case, the (static) unit count and unit rank are recalculated.

``cuda::experimental::coop::group_view`` uses the viewed group's synchronization mechanism.

Queries
-------

``cuda::experimental::coop::group_view`` provides the exact same queries as the viewed group, unless the unit type was changed. In that case, the sub-unit queries are limited to the group view's unit type.

Examples
--------

.. code:: cuda

  #include <cuda/devices>
  #include <cuda/stream>
  #include <cuda/hierarchy>
  #include <cuda/launch>
  #include <cuda/std/cassert>
  #include <cuda/std/cstdint>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  // The shuffle algorithm exchanges values among units within a given group.

  // An overload for "this" groups (the unit type is the same as the level type) is basically useless,
  // because there is just 1 unit within the group.
  template <typename Group>
    requires cuda::std::same_as<typename Group::unit_type, typename Group::level_type>
  __device__ int shuffle(const Group& group, int value, cuda::std::uint32_t src_rank)
  {
    assert(src_rank == 0);
    return value;
  }

  // What we need is an overload that takes a group whose unit and level types are not the same.
  // This overload implements shuffling values among threads within a warp.
  template <typename Group>
    requires cuda::std::same_as<typename Group::unit_type, cuda::thread_level>
      && cuda::std::same_as<typename Group::level_type, cuda::warp_level>
  __device__ int shuffle(const Group& group, int value, cuda::std::uint32_t src_rank)
  {
    // Because this overload takes the group as const&, we can't use it in a constant expression
    // before C++23. This is when group_view can be handy. We can create a view over the group
    // that can be used in a constant expression.
    const cudax::coop::group_view view{group};
    constexpr auto unit_count = cuda::gpu_thread.static_count(view);

    // Allocate shared memory for data exchange.
    __shared__ int smem[unit_count];

    // Write the unit's data to the shared memory.
    smem[cuda::gpu_thread.rank(view)] = value;

    // Wait until the write is done.
    group.sync_aligned();

    // Return the result.
    return smem[src_rank];
  }

  __device__ void demo(auto config)
  {
    // Create a group that holds a warp.
    cudax::coop::this_warp g{config};

    // Using g directly in the shuffle algorithm doesn't give us what we need. Only rank 0 can
    // be passed as the source rank.
    const auto result1 = shuffle(g, /*value*/ 0xabcdef, /*src_rank*/ 0);
    assert(result1 == 0xabcdef);


    // What we need is to treat g as a group of threads. That's where group_view can save us. We
    // needn't create another group that could require new synchronization resources; we can
    // just change the unit type.
    const auto result2 = shuffle(cudax::coop::group_view{cuda::gpu_thread, g},
                  /*value*/     cuda::gpu_thread.rank(g) * 200,
                    /*src_rank*/ (cuda::gpu_thread.rank(g) + 10) % cuda::gpu_thread.count(g));
    assert(result2 == ((cuda::gpu_thread.rank(g) + 10) % cuda::gpu_thread.count(g)) * 200);
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

`See it on Godbolt 🔗 <https://godbolt.org/z/qPPb887zf>`__
