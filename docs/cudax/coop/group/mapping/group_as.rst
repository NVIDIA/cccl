.. _cudax-coop-group-mapping-group-as:

``cuda::experimental::coop::group_as``
======================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename RangeView>
  class group_as : public /*group-mapping-interface-for*/<group_as>
  {
    RangeView /*unit_counts_*/; // exposition-only

  public:
    template <typename Range>
      requires cuda::std::is_same_v<RangeView, cuda::std::views::all_t<Range>>
    __device__ explicit group_as(Range&& unit_counts)
      noexcept(noexcept(cuda::std::views::all(cuda::std::forward<Range>(unit_counts))))
      : /*unit_counts_*/{cuda::std::views::all(cuda::std::forward<Range>(unit_counts))}
    {}
  };

  template <typename Range>
    requires (cuda::std::ranges::sized_range<Range> &&
              cuda::std::ranges::viewable_range<Range> &&
              cuda::std::is_convertible_v<cuda::std::ranges::range_value_t<Range>, cuda::std::uint32_t>)
  group_as(Range&&) -> group_as<cuda::std::views::all_t<Range>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::group_as`` is a mapping that splits every group from the previous mapping result into subgroups with individually specified unit counts. Use it when subgroups need different sizes, for example to assign different numbers of warps to different tasks within a block.

The ``unit_counts`` range contains one size per subgroup. Units are assigned in order of their rank in the previous mapping result: the first subgroup gets the first ``unit_counts[0]`` units, the second gets the next ``unit_counts[1]`` units, and so on. If the previous mapping result describes multiple groups, the same sizes are applied independently to each group, and the resulting subgroup ranks are ordered first by the previous group rank, then by the index in ``unit_counts``.

``RangeView`` must be a viewable sized range whose value type is convertible to ``cuda::std::uint32_t``. The stored view must support indexing and iteration through a const view. Counts must be nonnegative and representable as ``cuda::std::uint32_t``, and their sum and the resulting group count must also be representable in that type. All units in the same input group must agree on the sizes.

During exhaustive mapping, the sum of the sizes must equal the previous mapping result's unit count. During non-exhaustive mapping, the sum may be smaller, leaving trailing units without a group. The sum must never exceed the available unit count; non-exhaustive mapping does not truncate a requested subgroup. Violating these requirements is undefined behavior.

Examples
--------

.. code:: cuda

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Range of unit counts for the mapping.
    const cuda::std::array unit_counts{4, 12, 16};

    // Mapping that splits every group from the previous mapping result into 3 subgroups of 4, 12 and 16 units.
    cudax::coop::group_as m{unit_counts};
  }
