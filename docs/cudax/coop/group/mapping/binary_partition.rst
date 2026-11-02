.. _cudax-coop-group-mapping-binary-partition:

``cuda::experimental::coop::binary_partition``
==============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Fn>
  class binary_partition : public /*group-mapping-interface-for*/<binary_partition>
  {
    Fn /*fn_*/; // exposition-only

  public:
    __device__ explicit binary_partition(Fn fn) noexcept(cuda::std::is_nothrow_move_constructible_v<Fn>)
      : /*fn_*/(cuda::std::move(fn))
    {}
  };

  template <typename Fn>
  binary_partition(Fn) -> binary_partition<Fn>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::binary_partition`` is a mapping that splits units from the previous mapping result into 2 groups based on a boolean value. The boolean value is obtained by invoking the ``/*fn_*/`` callable with the previous mapping result. It is possible that all of the units will end up in the same group. Currently, it can only be used to group threads within the warp level.

The returned mapping result is always exhaustive.

The returned mapping result is not always contiguous.

Examples
--------

.. code:: cuda

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Creates a mapping that splits units into even/odd groups.
    cudax::coop::binary_partition m{[](const auto& prev_mapping_result){ return prev_mapping_result.unit_rank() % 2 == 1; }};
  }
