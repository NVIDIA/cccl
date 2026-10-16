import pytest

impl = pytest.importorskip("cuda.compute._device_copy_impl")


def make_plan(
    shape,
    source_strides,
    destination_strides,
    source_element_offset=0,
    destination_element_offset=0,
):
    return impl._make_device_copy_plan(
        shape,
        source_strides,
        destination_strides,
        source_element_offset,
        destination_element_offset,
    )


def test_DeviceCopyPlan_collapses_contiguous_axes():
    plan = make_plan((2, 3, 4), (12, 4, 1), (12, 4, 1))

    assert plan.original_rank == 3
    assert plan.rank == 1
    assert plan.elements == 24
    assert plan.shape == (24,)
    assert plan.source_strides == (1,)
    assert plan.destination_strides == (1,)
    assert plan.contiguous_1d


def test_DeviceCopyPlan_removes_unit_axes_before_collapsing():
    plan = make_plan((1, 2, 1, 3), (99, 3, 123, 1), (88, 3, 77, 1))

    assert plan.original_shape == (1, 2, 1, 3)
    assert plan.original_source_strides == (99, 3, 123, 1)
    assert plan.original_destination_strides == (88, 3, 77, 1)
    assert plan.rank == 1
    assert plan.shape == (6,)
    assert plan.source_strides == (1,)
    assert plan.destination_strides == (1,)


def test_DeviceCopyPlan_reorders_axes_by_destination_then_source_stride():
    plan = make_plan((2, 3), (1, 5), (1, 7))

    assert plan.rank == 2
    assert plan.axis_order == (1, 0)
    assert plan.shape == (3, 2)
    assert plan.source_strides == (5, 1)
    assert plan.destination_strides == (7, 1)


def test_DeviceCopyPlan_flips_destination_negative_axes():
    plan = make_plan((4,), (1,), (-1,), 10, 13)

    assert plan.rank == 1
    assert plan.shape == (4,)
    assert plan.source_strides == (-1,)
    assert plan.destination_strides == (1,)
    assert plan.source_element_offset == 13
    assert plan.destination_element_offset == 10


def test_DeviceCopyPlan_empty_copy_has_zero_rank():
    plan = make_plan((2, 0, 3), (3, 1, 1), (3, 1, 1))

    assert plan.empty
    assert plan.contiguous_1d
    assert plan.elements == 0
    assert plan.rank == 0
    assert plan.shape == ()
    assert plan.source_strides == ()
    assert plan.destination_strides == ()


def test_DeviceCopyPlan_scalar_copy_has_zero_rank():
    plan = make_plan((), (), ())

    assert not plan.empty
    assert plan.contiguous_1d
    assert plan.elements == 1
    assert plan.rank == 0
    assert plan.shape == ()


def test_DeviceCopyPlan_rejects_mismatched_ranks():
    with pytest.raises(ValueError, match="source strides rank"):
        make_plan((2, 3), (3,), (3, 1))

    with pytest.raises(ValueError, match="destination strides rank"):
        make_plan((2, 3), (3, 1), (3,))


def test_DeviceCopyPlan_rejects_negative_extents():
    with pytest.raises(ValueError, match="shape extents"):
        make_plan((2, -3), (3, 1), (3, 1))


def test_DeviceCopyPlan_rejects_non_unique_destination():
    with pytest.raises(ValueError, match="not unique"):
        make_plan((2, 3), (3, 1), (0, 1))


def test_DeviceCopyPlan_rejects_views_before_allocation_base():
    with pytest.raises(ValueError, match="source array view"):
        make_plan((4,), (-1,), (1,))


def test_DeviceCopyPlan_can_check_source_uniqueness_without_mutating_axis_order():
    plan = make_plan((2, 3), (0, 1), (3, 1))
    axis_order = plan.axis_order

    assert not plan.original_source_unique
    assert not plan.source_unique
    assert plan.destination_unique
    assert plan.axis_order == axis_order


def test_DeviceCopyPlan_collapsed_axis_order_uses_smallest_original_axis():
    plan = make_plan((2, 3, 4), (1, 2, 6), (1, 2, 6))

    assert plan.rank == 1
    assert plan.shape == (24,)
    assert plan.axis_order == (0,)
