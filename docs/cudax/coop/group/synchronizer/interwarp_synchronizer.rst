.. _cudax-coop-group-synchronizer-interwarp-synchronizer:

``cuda::experimental::coop::interwarp_synchronizer``
====================================================

.. code:: cuda

  namespace cuda::experimental::coop {

  using /*interwarp-barrier-id-type*/ = int;

  template <typename RangeView>
  class interwarp_synchronizer
  {
    RangeView /*barrier_ids_*/; // exposition-only

  public:
    template <typename Range>
      requires cuda::std::same_as<RangeView, cuda::std::ranges::views::all_t<Range>>
    __device__ interwarp_synchronizer(Range&& barrier_ids)
      noexcept(noexcept(cuda::std::ranges::views::all(cuda::std::forward<Range>(barrier_ids))))
      : /*barrier_ids_*/{cuda::std::ranges::views::all(cuda::std::forward<Range>(barrier_ids))}
    {}

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const noexcept;
  };

  template <typename Range>
    requires cuda::std::ranges::viewable_range<Range>
      && cuda::std::is_convertible_v<cuda::std::ranges::range_value_t<Range>, /*interwarp-barrier-id-type*/>
  interwarp_synchronizer(Range&&)
    -> interwarp_synchronizer<cuda::std::ranges::views::all_t<Range>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::interwarp_synchronizer`` is a synchronizer that uses one of the 16 built-in warp barriers to synchronize the group. The type is constructible from a range of barrier IDs, which are sequentially assigned to each of the created groups during the synchronizer instance creation. It can only be used with groups that group warps within the block level.

Users should always rely on template argument deduction and never set the template arguments themselves. All of the barrier IDs must be less than ``16`` and can't be repeated multiple times.

.. warning:: The barrier ID ``0`` is also used by the ``__syncthreads()`` operation. It is recommended to avoid this barrier ID and start from barrier ID ``1`` instead.

Examples
--------

.. code:: cuda

  #include <cuda/std/array>
  #include <cuda/std/ranges>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Interwarp synchronizer from a list of barrier IDs.
    cuda::std::array array{1, 3, 5, 7};
    cudax::coop::interwarp_synchronizer s1{array};

    // Interwarp synchronizer from an infinite iota range.
    cudax::coop::interwarp_synchronizer s2{cuda::std::ranges::views::iota(1)};
  }
