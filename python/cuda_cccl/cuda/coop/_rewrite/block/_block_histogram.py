# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import cached_property

import cuda.coop._rewrite as _core

ArrayCallDefinition = _core.ArrayCallDefinition
CoopNode = _core.CoopNode
CoopNodeMixin = _core.CoopNodeMixin
Disposition = _core.Disposition
ir = _core.ir


# =============================================================================
# Histogram
# =============================================================================
@dataclass
class CoopBlockHistogramNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram"
    disposition = Disposition.PARENT

    def refine_match(self, rewriter):
        if self.is_two_phase:
            instance = self.two_phase_instance
            self.instance = instance
            instance.specialization.unique_id = self.unique_id
            self.item_dtype = instance.item_dtype
            self.counter_dtype = instance.counter_dtype
            self.items_per_thread = instance.items_per_thread
            self.bins = instance.bins
            self.algorithm = getattr(instance, "algorithm", None)

            launch_config = rewriter.launch_config
            if launch_config is None:
                return False

            self.threads_per_block = launch_config.blockdim
            self.children = []
            self.runtime_args = tuple()
            self.runtime_arg_types = tuple()
            self.runtime_arg_names = tuple()
            return

        bound = self.bound.arguments

        # Infer `items_per_thread` and `bins` from the shapes of the items and
        # histogram arrays, respectively.
        items_var = bound["items"]
        items_root = rewriter.get_root_def(items_var)
        items_leaf = items_root.leaf_constructor_call
        if not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected items to be an array call, got {items_leaf!r}"
            )
        items_per_thread = items_leaf.shape
        if not isinstance(items_per_thread, int):
            raise RuntimeError("Could not determine shape of items array.")

        histogram_var = bound["histogram"]
        histogram_root = rewriter.get_root_def(histogram_var)
        histogram_leaf = histogram_root.leaf_constructor_call
        if not isinstance(histogram_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected histogram to be an array call, got {histogram_leaf!r}"
            )
        bins = histogram_leaf.shape
        if not isinstance(bins, int):
            raise RuntimeError("Could not determine shape of histogram array.")

        self.algorithm = bound.get("algorithm")
        self.temp_storage = bound.get("temp_storage")

        items_ty = self.typemap[items_var.name]
        histogram_ty = self.typemap[histogram_var.name]

        self.items = items_var
        self.items_ty = items_ty
        self.items_root = items_root
        self.item_dtype = items_ty.dtype
        self.items_per_thread = items_per_thread

        self.histogram = histogram_var
        self.histogram_ty = histogram_ty
        self.histogram_root = histogram_root
        self.histogram_dtype = histogram_ty.dtype

        self.counter_dtype = histogram_ty.dtype
        self.bins = bins

        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        # Instantiate an instance now so our children can access it.
        self.instance = self.instantiate_impl(
            item_dtype=self.item_dtype,
            counter_dtype=self.counter_dtype,
            items_per_thread=self.items_per_thread,
            bins=self.bins,
            dim=self.threads_per_block,
            algorithm=self.algorithm,
            unique_id=self.unique_id,
            node=self,
            temp_storage=self.temp_storage,
        )
        self.children = []

        algo = self.instance.specialization
        assert len(algo.parameters) == 1, algo.parameters
        self.runtime_args = tuple()
        self.runtime_arg_types = tuple()
        self.runtime_arg_names = tuple()
        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        initial_value_assign = getattr(self, "initial_value_assign", None)
        if initial_value_assign is not None:
            instrs.append(initial_value_assign)
        instrs.append(rd.new_assign)
        return instrs

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockHistogramInitNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.init"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance or parent_node.two_phase_instance

        bound = self.bound.arguments
        histogram = bound.get("histogram")
        if histogram is None:
            histogram = parent_node.histogram
            histogram_ty = parent_node.histogram_ty
        else:
            histogram_ty = self.typemap[histogram.name]
            if getattr(parent_node, "histogram", None) is None:
                parent_node.histogram = histogram
                parent_node.histogram_ty = histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.init(self)

        self.runtime_args = [histogram]
        self.runtime_arg_types = [histogram_ty]
        self.runtime_arg_names = ["histogram"]

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign]
        initial_value_assign = getattr(self, "initial_value_assign", None)
        if initial_value_assign is not None:
            instrs.append(initial_value_assign)
        valid_items_assign = getattr(self, "valid_items_assign", None)
        if valid_items_assign is not None:
            instrs.append(valid_items_assign)
        instrs.append(rd.new_assign)
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()


@dataclass
class CoopBlockHistogramCompositeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.composite"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance or parent_node.two_phase_instance
        parent_root_def = parent_node.root_def
        assert self.parent_root_def is parent_root_def, (
            self.parent_root_def,
            parent_root_def,
        )

        bound = self.bound.arguments
        items = bound["items"]
        items_ty = self.typemap[items.name]
        parent_items_ty = getattr(parent_node, "items_ty", None)
        if parent_items_ty is None:
            parent_node.items_ty = items_ty
        elif items_ty != parent_items_ty:
            raise RuntimeError(
                f"Expected items type {parent_items_ty!r}, "
                f"got {items_ty!r} for {self!r}"
            )

        histogram = bound.get("histogram")
        if histogram is None:
            histogram = parent_node.histogram
            histogram_ty = parent_node.histogram_ty
        else:
            histogram_ty = self.typemap[histogram.name]
            if getattr(parent_node, "histogram", None) is None:
                parent_node.histogram = histogram
                parent_node.histogram_ty = histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.composite(self, items)

        self.runtime_args = [items, histogram]
        self.runtime_arg_types = [items_ty, histogram_ty]
        self.runtime_arg_names = ["items", "histogram"]

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        instrs = [rd.g_assign, rd.new_assign]
        return tuple(instrs)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
