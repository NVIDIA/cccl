# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import itertools
from collections import OrderedDict, defaultdict
from functools import cached_property
from types import ModuleType as PyModuleType
from typing import Any

from numba.core import types
from numba.cuda.core import ir, ir_utils
from numba.cuda.core.rewrites import Rewrite, register_rewrite

from ._core import (
    _GLOBAL_SYMBOL_ID_COUNTER,
    DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    MAX_SHARED_MEMORY_CARVEOUT_PERCENT,
    THREAD_DATA_DTYPE_INFERENCE_ARG_PAIRS,
    CoopLoadStoreNode,
    CoopNode,
    CoopTempStorageNode,
    CoopThreadDataNode,
    CudaLocalModuleTemplate,
    CudaSharedModuleTemplate,
    RootDefinition,
    Signature,
    SimpleNamespace,
    TempStorageCallDefinition,
    TempStorageInfo,
    TempStorageRewriteState,
    TempStorageType,
    TempStorageUseLayoutEntry,
    TempStorageUseRequirementEntry,
    ThreadDataCallDefinition,
    _get_env_bool,
    _warn_ltoir_bundle_failure,
    algo_coalesce_key,
    coop,
    functools,
    get_coop_class_and_instance_maps,
    get_element_count,
    get_kernel_param_value,
    get_root_definition,
    normalize_dtype_param,
    numba,
    operator,
    register_kernel_extension,
    struct,
)


def _ensure_current_launch_config():
    from . import ensure_current_launch_config

    return ensure_current_launch_config()


@register_rewrite("after-inference")
class CoopNodeRewriter(Rewrite):
    # This class is our cuda.coop rewriting workhorse.  It is responsible for
    # identifying calls to cooperative primitives, matching them to the
    # appropriate CoopNode subclass, and then invoking the rewrite logic on the
    # node to perform the actual IR transformation to lower the high-level
    # primitive call into the appropriate low-level IR that corresponds to the
    # underlying CUDA cooperative primitive implementation.

    # IR module globals we track so rewrite-generated calls can reuse
    # cached `numba` / `numba.cuda` module bindings.
    interesting_modules = {
        "numba",
        "numba.cuda",
    }

    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

        self._all_match_invocations_count = 0
        self._match_invocations_per_block_offset = defaultdict(int)
        self._unique_id_counter = itertools.count(0)

        self.first_match = True

        self.func_ir = None

        maps = get_coop_class_and_instance_maps()
        self._decl_classes = maps.decls
        self._node_classes = maps.nodes
        self._instances = maps.instances
        self._roots = {}

        # Map of fully-qualified module names to the ir.Assign instruction
        # for loading the module as a global variable.
        self._modules: dict[str, ir.Assign] = {}

        # Set of fully-qualified module names requested by nodes being
        # rewritten this pass.  Nodes that need a module will call our
        # `needs_module(self, fq_mod_name)` method, which will add the
        # name to this set.
        self._needs_module: set[str] = set()

        self._thread_data_info: dict[str, Any] = {}
        self._temp_storage_info: dict[str, Any] = {}
        self._temp_storage_state = TempStorageRewriteState()

        self._state = state
        self.typingctx = state.typingctx

        self.nodes = OrderedDict()

        self.typemap = None
        self.calltypes = None

        self._bundle_ltoir_enabled = _get_env_bool(
            "NUMBA_CCCL_COOP_BUNDLE_LTOIR", default=True
        )
        self._bundle_ltoir_done = False
        self._bundle_ltoir_failed = False
        self._bundle_ltoir = None
        self._symbol_id_counter = _GLOBAL_SYMBOL_ID_COUNTER
        self._symbol_id_map: dict[Any, int] = {}

        # Advanced C++ CUDA kernels will often leverage a templated struct
        # for specialized CUB primitives, e.g.
        #   template <T, int items_per_thread>
        #   struct KernelTraits {
        #       using BlockLoadT = cub::BlockLoad<T, items_per_thread>;
        #       ...
        #   };
        #   ...
        #   template <typename KernelTraitsT>
        #   __global__ void kernel(...) {
        #       constexpr auto items_per_thread =
        #           KernelTraitsT::items_per_thread;
        #       extern __shared__ char smem[];
        #       auto& smem_load = reinterpret_cast<
        #           typename KernelTraitsT::BlockLoadT::TempStorage&>(smem);
        #       ...
        #   }
        #
        # The equivalent Pythonic cuda.coop equivalent of the above (sans
        # kernel implementation) would be something along the lines of:
        #
        #   def make_kernel_traits(dtype, dim, items_per_thread):
        #       @dataclass
        #       class KernelTraits:
        #           items_per_thread: int
        #           block_load = coop.block.load(
        #               dtype, dim, items_per_thread,
        #           )
        #       return KernelTraits(dtype, dim, items_per_thread)
        #
        #   @cuda.jit
        #   def kernel(d_in, d_out, traits: KernelTraits):
        #       ...
        #       traits.block_load(d_in, d_out)
        #       ...
        #
        #   traits = make_kernel_traits(np.float32, 128, 4)
        #   kernel[blocks_per_grid, threads_per_block](d_in, d_out, traits)
        #
        # In order to support this pattern, we need to track instances of
        # kernel arguments for which we saw a function call of a coop primitive
        # where the primitive was an attribute of a struct-like object--i.e.
        # the `traits.block_load` in the example above.
        #
        # We generically refer to these container struct-like objects in this
        # routine as "structs", and we track the names of ones we've seen in
        # the set named `seen_structs`.
        #
        # When we encounter a new "struct", we synthesize a new class that
        # defines a `prepare_args(self, ty, val, *args, **kwds)` method to
        # pacify numba's _Kernel._parse_args() routine when it processes the
        # `traits` argument in the example above prior to kernel launch.  (See
        # `self.handle_new_kernel_traits_struct()` for more details.)
        #
        # This would normally be handled by supplying an appropriate "extension"
        # (i.e. an instance with a `prepare_args()` method) to the @cuda.jit
        # decorator's `extensions` kwarg, e.g.:
        #   @cuda.jit(extensions=[custom_prepare_args, ...])`
        #
        # That is not particularly user-friendly though, and becomes unwieldy
        # with non-trivial kernel trait "structs" that have multiple coop
        # primitive members (e.g. block load, scan, reduce, scan, etc.).
        #
        # So, to remove the requirement for the user to furnish all these
        # extension handlers up-front to the @cuda.jit decorator, we need to
        # add our synthesized class with a `prepare_args()` method to the
        # kernel's list of extensions prior to kernel launch.  We achieve
        # this by obtaining the launch configuration and appending to the
        # `pre_launch_callbacks` list, providing another synthesized routine
        # that, when invoked, will update the kernel's extensions list.
        self.seen_structs = set()

    def ensure_ltoir_bundle(self):
        if not self._bundle_ltoir_enabled:
            return
        if self._bundle_ltoir_done or self._bundle_ltoir_failed:
            return

        for node in self.nodes.values():
            if getattr(node, "parent_node", None) is not None:
                self._bundle_ltoir_done = True
                return

        algorithms = []
        seen = set()
        for node in self.nodes.values():
            instance = getattr(node, "instance", None)
            if instance is None:
                instance = getattr(node, "two_phase_instance", None)
            if instance is None:
                continue
            algo = getattr(instance, "specialization", None)
            if algo is None:
                continue
            if "lto_irs" in algo.__dict__:
                continue
            key = id(algo)
            if key in seen:
                continue
            seen.add(key)
            algorithms.append(algo)

        if len(algorithms) < 2:
            self._bundle_ltoir_done = True
            return

        try:
            from .. import _types
            from . import algo_coalesce_key

            coalesce_keys = {algo_coalesce_key(algo) for algo in algorithms}
            allow_single = len(coalesce_keys) == 1 and len(algorithms) > 1
            bundle = _types.prepare_ltoir_bundle(
                algorithms,
                bundle_name=f"cuda_coop_bundle_{id(self)}",
                allow_single=allow_single,
            )
            if bundle is not None:
                self._bundle_ltoir = bundle
            self._bundle_ltoir_done = True
        except Exception as exc:
            self._bundle_ltoir_failed = True
            _warn_ltoir_bundle_failure(exc)

    def _get_or_create_global_module(
        self,
        mod_key: str,
        module_obj,
        scope,
        loc,
        new_nodes: list[ir.Assign],
    ) -> ir.Assign:
        instr = self._modules.get(mod_key)
        if instr:
            new_nodes.append(instr)
            return instr

        # Unique SSA variable for the module object
        unique_name = f"$g_{mod_key.replace('.', '_')}_var"
        var_name = ir_utils.mk_unique_var(unique_name)
        var = ir.Var(scope, var_name, loc)

        # ir.Global wraps the Python module so it becomes a constant
        g_mod = ir.Global(mod_key.split(".")[-1], module_obj, loc)
        instr = ir.Assign(value=g_mod, target=var, loc=loc)

        # Type-map entry so later passes know this is a module object.
        self.typemap[var.name] = types.Module(module_obj)

        # Cache for reuse.
        self._modules[mod_key] = instr

        new_nodes.append(instr)
        return instr

    def get_or_create_global_numba_module_instr(self, scope, loc, new_nodes):
        return self._get_or_create_global_module("numba", numba, scope, loc, new_nodes)

    def get_or_create_global_numba_cuda_module_instr(self, scope, loc, new_nodes):
        return self._get_or_create_global_module(
            "numba.cuda", numba.cuda, scope, loc, new_nodes
        )

    @cached_property
    def _get_root_def(self):
        """
        Returns a function that retrieves the root definition for a given
        variable in the function IR, using the current launch configuration.
        """
        return functools.partial(
            get_root_definition,
            func_ir=self.func_ir,
            typemap=self.typemap,
            calltypes=self.calltypes,
            launch_config=self.launch_config,
            assignments_map=self.assignments_map,
            rewriter=self,
        )

    def get_root_def(self, instr: ir.Inst) -> RootDefinition:
        """
        Returns the root definition for the given instruction, creating it
        if it does not already exist.
        """
        root_def = self._roots.get(instr, None)
        if root_def is not None:
            return root_def

        # Create a new root definition for the instruction.
        root_def = self._get_root_def(instr)
        if root_def is None:
            raise RuntimeError(f"Could not find root definition for {instr!r}")
        self._roots[instr] = root_def
        return root_def

    def _iter_call_exprs(self):
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    yield expr

    def _iter_call_assigns(self):
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    yield block, instr, expr

    def _getitem_expr_temp_storage_arg(self, expr):
        if not isinstance(expr, ir.Expr):
            return None
        if expr.op not in ("getitem", "static_getitem"):
            return None

        base_var = expr.value
        if not isinstance(base_var, ir.Var):
            return None
        base_ty = self.typemap.get(base_var.name)
        if not (
            isinstance(base_ty, types.Function)
            and hasattr(base_ty, "templates")
            and base_ty.templates
            and getattr(base_ty.templates[0], "primitive_name", "").startswith("coop.")
        ):
            return None

        index_var = getattr(expr, "index", None)
        if not isinstance(index_var, ir.Var):
            index_var = getattr(expr, "index_var", None)
        if not isinstance(index_var, ir.Var):
            return None

        index_ty = self.typemap.get(index_var.name)
        if isinstance(index_ty, TempStorageType):
            return index_var

        try:
            root = self.get_root_def(index_var)
            leaf = root.leaf_constructor_call
        except Exception:
            leaf = None
        if not isinstance(leaf, TempStorageCallDefinition):
            return None

        return index_var

    def _call_expr_getitem_index_var(self, expr):
        if not isinstance(expr, ir.Expr) or expr.op != "call":
            return None

        func_var = expr.func
        if not isinstance(func_var, ir.Var):
            return None

        try:
            func_def = self.func_ir.get_definition(func_var.name)
        except Exception:
            return None

        if not isinstance(func_def, ir.Expr):
            return None
        return self._getitem_expr_temp_storage_arg(func_def)

    def get_getitem_temp_storage_arg(self, expr):
        index_var = self._call_expr_getitem_index_var(expr)
        if index_var is None:
            return None

        return index_var

    def _expr_uses_var(self, expr, var_name):
        args = expr.args
        kws = expr.kws
        if isinstance(kws, (list, tuple)):
            kws = dict(kws)
        elif kws is None:
            kws = {}
        if any(isinstance(arg, ir.Var) and arg.name == var_name for arg in args):
            return True
        if any(
            isinstance(arg, ir.Var) and arg.name == var_name for arg in kws.values()
        ):
            return True

        getitem_temp_storage = self.get_getitem_temp_storage_arg(expr)
        if getitem_temp_storage is not None and getitem_temp_storage.name == var_name:
            return True

        return False

    def _build_coop_node_for_call_expr(self, block, instr):
        expr = instr.value
        if not isinstance(expr, ir.Expr) or expr.op != "call":
            return None

        func_name = expr.func.name
        func = self.typemap.get(func_name)
        if func is None:
            return None

        template = None
        impl_class = None
        primitive_name = None
        type_instance = None
        two_phase_instance = None

        if hasattr(func, "templates"):
            templates = func.templates
            if len(templates) != 1:
                return None

            template = templates[0]
            impl_class = self._decl_classes.get(template)
            if not impl_class:
                return None

            primitive_name = getattr(template, "primitive_name", None)
            if primitive_name is None:
                return None
        elif func in self._instances:
            type_instance = func
            value_type = self.typemap.get(func_name)
            decl = getattr(value_type, "decl", None)
            if decl is None:
                return None

            template = decl.__class__
            impl_class = self._decl_classes.get(template)
            if not impl_class:
                return None

            primitive_name = decl.primitive_name

            func_root = self.get_root_def(expr.func)
            two_phase_instance = func_root.attr_instance
            if two_phase_instance is None:
                two_phase_instance = func_root.instance
            if isinstance(two_phase_instance, PyModuleType):
                return None
            if two_phase_instance is None:
                return None
        else:
            return None

        node_class = self._node_classes.get(primitive_name)
        if node_class is None:
            return None

        root_def = self.get_root_def(instr)
        node = node_class(
            index=0,
            block_line=block.loc.line,
            expr=expr,
            instr=instr,
            template=template,
            func=func,
            func_name=func_name,
            impl_class=impl_class,
            target=instr.target,
            calltypes=self.calltypes,
            typemap=self.typemap,
            func_ir=self.func_ir,
            typingctx=self._state.typingctx,
            launch_config=self.launch_config,
            needs_pre_launch_callback=False,
            unique_id=self.next_unique_id(),
            type_instance=type_instance,
            two_phase_instance=two_phase_instance,
            root_def=root_def,
            parent_struct_instance_type=None,
            parent_node=None,
            parent_root_def=None,
            rewriter=self,
        )
        return node

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def _infer_temp_storage_requirements_from_uses(self, var):
        var_name = var.name
        requirements = []
        failures = []
        ordinal = 0

        for block, instr, expr in self._iter_call_assigns():
            if not self._expr_uses_var(expr, var_name):
                continue

            node = self._build_coop_node_for_call_expr(block, instr)
            if node is None:
                continue

            try:
                bound = node.bound.arguments
            except Exception:
                continue

            temp_storage_arg = bound.get("temp_storage")
            if not (
                isinstance(temp_storage_arg, ir.Var)
                and temp_storage_arg.name == var_name
            ):
                continue

            original_ty = self.typemap.get(var_name)
            patched_typemap = False
            try:
                if isinstance(original_ty, TempStorageType):
                    del self.typemap[var_name]
                    self.typemap[var_name] = types.Array(types.uint8, 1, "C")
                    patched_typemap = True

                node.refine_match(self)

                instance = node.instance or node.two_phase_instance
                if instance is None and node.impl_kwds is not None:
                    instance = node.instantiate_impl(**node.impl_kwds)
                if instance is None and isinstance(node, CoopLoadStoreNode):
                    instance = node.instantiate_impl(
                        dtype=node.dtype,
                        dim=node.threads_per_block,
                        items_per_thread=node.items_per_thread,
                        algorithm=node.algorithm_id,
                        num_valid_items=node.num_valid_items,
                        oob_default=node.oob_default,
                        unique_id=node.unique_id,
                        node=node,
                        temp_storage=node.temp_storage,
                    )
                if instance is None:
                    raise RuntimeError(
                        f"Could not build instance for {node.primitive_name}"
                    )

                size_in_bytes = int(instance.temp_storage_bytes)
                alignment = int(instance.temp_storage_alignment or 1)
                if size_in_bytes <= 0:
                    raise RuntimeError(
                        f"Invalid temp_storage_bytes={size_in_bytes} for "
                        f"{node.primitive_name}"
                    )
                if alignment <= 0:
                    raise RuntimeError(
                        f"Invalid temp_storage_alignment={alignment} for "
                        f"{node.primitive_name}"
                    )

                requirements.append(
                    TempStorageUseRequirementEntry(
                        instr=instr,
                        expr=expr,
                        call_key=id(instr),
                        ordinal=ordinal,
                        primitive_name=node.primitive_name,
                        size_in_bytes=size_in_bytes,
                        alignment=alignment,
                    )
                )
                ordinal += 1
            except Exception as exc:
                failures.append((instr.loc, node.primitive_name, str(exc)))
            finally:
                if patched_typemap:
                    del self.typemap[var_name]
                    self.typemap[var_name] = original_ty

        if failures:
            details = "; ".join(
                f"{primitive}@{loc}: {reason}" for (loc, primitive, reason) in failures
            )
            raise RuntimeError(
                f"Failed to infer TempStorage size/alignment for {var_name}: {details}"
            )

        if not requirements:
            raise RuntimeError(
                "Could not infer TempStorage size/alignment for "
                f"{var_name}; pass size_in_bytes/alignment explicitly."
            )

        return requirements

    def _iter_temp_storage_root_vars(self):
        seen = set()
        vars_and_order = []
        for block in self.func_ir.blocks.values():
            for instr in block.body:
                if not isinstance(instr, ir.Assign):
                    continue
                target = instr.target
                if not isinstance(target, ir.Var):
                    continue
                try:
                    root = self.get_root_def(target)
                except Exception:
                    continue
                try:
                    leaf = root.leaf_constructor_call
                except Exception:
                    continue
                if not isinstance(leaf, TempStorageCallDefinition):
                    continue
                root_assign = root.root_assign
                if not isinstance(root_assign, ir.Assign):
                    continue
                root_var = root_assign.target
                if not isinstance(root_var, ir.Var):
                    continue
                if root_var.name in seen:
                    continue
                seen.add(root_var.name)
                order = leaf.order if leaf.order is not None else 0
                vars_and_order.append((order, root_var))

        vars_and_order.sort(key=lambda pair: pair[0])
        return [v for _, v in vars_and_order]

    def _temp_storage_var_is_used(self, var):
        var_name = var.name
        for expr in self._iter_call_exprs():
            if self._expr_uses_var(expr, var_name):
                return True
        return False

    def _get_device_shared_memory_limits(self):
        max_default = DEFAULT_STATIC_SHARED_MEMORY_BYTES
        max_optin = max_default
        try:
            device = numba.cuda.current_context().device
            max_default = int(
                getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK", max_default)
                or max_default
            )
            max_optin = int(
                getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 0) or 0
            )
            if max_optin <= 0:
                max_optin = max_default
        except Exception:
            pass
        return max_default, max_optin

    @staticmethod
    def _required_shared_memory_carveout_percent(
        required_dynamic_shared_bytes: int,
        max_optin_shared_bytes: int,
    ) -> int:
        if required_dynamic_shared_bytes <= 0 or max_optin_shared_bytes <= 0:
            return 0
        # Round up to the minimum percentage that can satisfy the requested
        # shared-memory bytes.
        percent = (
            (required_dynamic_shared_bytes * MAX_SHARED_MEMORY_CARVEOUT_PERCENT)
            + (max_optin_shared_bytes - 1)
        ) // max_optin_shared_bytes
        return max(0, min(MAX_SHARED_MEMORY_CARVEOUT_PERCENT, int(percent)))

    def _maybe_register_temp_storage_launch_callback(
        self,
        dynamic_shared_bytes,
        shared_memory_carveout_percent,
    ):
        if self._temp_storage_state.launch_callback_registered:
            return

        launch_config = self.launch_config

        if dynamic_shared_bytes <= 0:
            return

        try:
            if launch_config.sharedmem < dynamic_shared_bytes:
                launch_config.sharedmem = dynamic_shared_bytes
        except Exception:
            pass

        def _temp_storage_pre_launch_callback(kernel_obj, _launch_config):
            try:
                from numba.cuda.cudadrv import driver
                from numba.cuda.cudadrv.driver import binding

                attrs = binding.CUfunction_attribute
                max_dynamic_smem_attr = (
                    attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                )

                cufunc = kernel_obj._codelibrary.get_cufunc()
                driver.driver.cuKernelSetAttribute(
                    max_dynamic_smem_attr,
                    dynamic_shared_bytes,
                    cufunc.handle,
                    cufunc.device.id,
                )
                cufunc.set_shared_memory_carveout(shared_memory_carveout_percent)

                launch_patched_attr = "_coop_temp_storage_dynamic_launch_patched"
                if dynamic_shared_bytes > 0 and not getattr(
                    kernel_obj, launch_patched_attr, False
                ):
                    original_launch = kernel_obj.launch

                    def _launch_with_dynamic_smem(
                        args, griddim, blockdim, stream, sharedmem
                    ):
                        if sharedmem < dynamic_shared_bytes:
                            sharedmem = dynamic_shared_bytes
                        return original_launch(
                            args,
                            griddim,
                            blockdim,
                            stream,
                            sharedmem,
                        )

                    kernel_obj.launch = _launch_with_dynamic_smem
                    setattr(kernel_obj, launch_patched_attr, True)
            except Exception:
                return

        launch_config.pre_launch_callbacks.append(_temp_storage_pre_launch_callback)
        self._temp_storage_state.launch_callback_registered = True

    def _ensure_temp_storage_global_plan(self):
        if self._temp_storage_state.global_plan is not None:
            return self._temp_storage_state.global_plan
        if self._temp_storage_state.global_plan_in_progress:
            raise RuntimeError("Recursive TempStorage global planning detected.")

        self._temp_storage_state.global_plan_in_progress = True
        try:
            infos = []
            for root_var in self._iter_temp_storage_root_vars():
                info = self.get_temp_storage_info(root_var)
                infos.append(info)

            infos.sort(key=lambda info: info.order)
            offset = 0
            max_alignment = 1
            for info in infos:
                alignment = int(info.alignment or 1)
                offset = self._align_up(offset, alignment)
                info.base_offset = offset
                offset += int(info.size_in_bytes)
                max_alignment = max(max_alignment, alignment)

            total_size = self._align_up(offset, max_alignment)
            max_default, max_optin = self._get_device_shared_memory_limits()
            uses_dynamic_smem = total_size > max_default
            dynamic_shared_bytes = total_size if uses_dynamic_smem else 0
            shared_memory_carveout_percent = (
                self._required_shared_memory_carveout_percent(
                    dynamic_shared_bytes,
                    max_optin,
                )
            )
            if dynamic_shared_bytes > max_optin:
                raise RuntimeError(
                    "TempStorage requires "
                    f"{dynamic_shared_bytes} bytes dynamic shared memory, but "
                    f"device max opt-in is {max_optin} bytes."
                )

            plan = SimpleNamespace(
                infos=infos,
                total_size=total_size,
                max_alignment=max_alignment,
                uses_dynamic_smem=uses_dynamic_smem,
                dynamic_shared_bytes=dynamic_shared_bytes,
                shared_memory_carveout_percent=shared_memory_carveout_percent,
                max_default_smem=max_default,
                max_optin_smem=max_optin,
            )
            self._temp_storage_state.global_plan = plan
            if dynamic_shared_bytes > 0:
                self._maybe_register_temp_storage_launch_callback(
                    dynamic_shared_bytes,
                    shared_memory_carveout_percent,
                )
            return plan
        finally:
            self._temp_storage_state.global_plan_in_progress = False

    def _ensure_temp_storage_global_backing_var(self):
        if self._temp_storage_state.global_backing_var is not None:
            return self._temp_storage_state.global_backing_var

        plan = self._ensure_temp_storage_global_plan()
        if plan.total_size <= 0:
            return None

        entry_label = min(self.func_ir.blocks.keys())
        entry_block = self.func_ir.blocks[entry_label]
        scope = entry_block.scope
        loc = entry_block.loc

        var_name = ir_utils.mk_unique_var("$coop_temp_storage_backing")
        backing_var = ir.Var(scope, var_name, loc)

        instrs = self.emit_cuda_array_call(
            scope,
            loc,
            0 if plan.uses_dynamic_smem else plan.total_size,
            numba.uint8,
            alignment=plan.max_alignment,
            shared=True,
            target=backing_var,
        )
        self._temp_storage_state.global_backing_var = backing_var
        self._temp_storage_state.global_backing_prelude_instrs.extend(instrs)
        return backing_var

    def emit_array_slice_call(
        self,
        scope,
        loc,
        src_var,
        start,
        stop=None,
        *,
        target=None,
        symbol_prefix="$coop_slice",
    ):
        new_nodes = []

        start_name = ir_utils.mk_unique_var(f"{symbol_prefix}_start")
        start_var = ir.Var(scope, start_name, loc)
        self.typemap[start_var.name] = types.IntegerLiteral(start)
        new_nodes.append(ir.Assign(ir.Const(start, loc), start_var, loc))

        stop_name = ir_utils.mk_unique_var(f"{symbol_prefix}_stop")
        stop_var = ir.Var(scope, stop_name, loc)
        if stop is None:
            self.typemap[stop_var.name] = types.none
            new_nodes.append(ir.Assign(ir.Const(None, loc), stop_var, loc))
        else:
            self.typemap[stop_var.name] = types.IntegerLiteral(stop)
            new_nodes.append(ir.Assign(ir.Const(stop, loc), stop_var, loc))

        slice_fn_name = ir_utils.mk_unique_var(f"{symbol_prefix}_slice_fn")
        slice_fn_var = ir.Var(scope, slice_fn_name, loc)
        slice_fn_ty = self.typingctx.resolve_value_type(slice)
        self.typemap[slice_fn_var.name] = slice_fn_ty
        new_nodes.append(ir.Assign(ir.Global("slice", slice, loc), slice_fn_var, loc))

        slice_obj_name = ir_utils.mk_unique_var(f"{symbol_prefix}_slice_obj")
        slice_obj_var = ir.Var(scope, slice_obj_name, loc)
        slice_call = ir.Expr.call(
            func=slice_fn_var,
            args=(start_var, stop_var),
            kws=(),
            loc=loc,
        )
        slice_sig = slice_fn_ty.get_call_type(
            self.typingctx,
            args=(self.typemap[start_var.name], self.typemap[stop_var.name]),
            kws={},
        )
        self.calltypes[slice_call] = slice_sig
        self.typemap[slice_obj_var.name] = slice_sig.return_type
        new_nodes.append(ir.Assign(slice_call, slice_obj_var, loc))

        getitem_fn_name = ir_utils.mk_unique_var(f"{symbol_prefix}_getitem_fn")
        getitem_fn_var = ir.Var(scope, getitem_fn_name, loc)
        getitem_fn_ty = self.typingctx.resolve_value_type(operator.getitem)
        self.typemap[getitem_fn_var.name] = getitem_fn_ty
        new_nodes.append(
            ir.Assign(
                ir.Global("getitem", operator.getitem, loc),
                getitem_fn_var,
                loc,
            )
        )

        if target is None:
            target_name = ir_utils.mk_unique_var(f"{symbol_prefix}_target")
            target = ir.Var(scope, target_name, loc)

        getitem_call = ir.Expr.call(
            func=getitem_fn_var,
            args=(src_var, slice_obj_var),
            kws=(),
            loc=loc,
        )
        getitem_sig = getitem_fn_ty.get_call_type(
            self.typingctx,
            args=(self.typemap[src_var.name], self.typemap[slice_obj_var.name]),
            kws={},
        )
        self.calltypes[getitem_call] = getitem_sig
        existing = self.typemap.get(target.name)
        if existing is not None and existing != getitem_sig.return_type:
            del self.typemap[target.name]
        self.typemap[target.name] = getitem_sig.return_type
        new_nodes.append(ir.Assign(getitem_call, target, loc))

        return new_nodes, target

    def bind_temp_storage_runtime_arg(
        self,
        *,
        node,
        temp_storage,
        runtime_args,
        runtime_arg_types,
        runtime_arg_names,
        insert_pos=0,
    ):
        if temp_storage is None:
            return None, None, None

        assert isinstance(temp_storage, ir.Var)
        temp_storage_ty = self.typemap[temp_storage.name]
        temp_storage_info = None

        if isinstance(temp_storage_ty, TempStorageType):
            temp_storage_info = self.get_temp_storage_info(temp_storage)
            temp_storage_ty = types.Array(types.uint8, 1, "C")

            if temp_storage_info.sharing == "exclusive":
                use_info = temp_storage_info.use_layout.get(id(node.instr))
                if use_info is None:
                    raise RuntimeError(
                        "Could not resolve exclusive TempStorage slice for "
                        f"{node.primitive_name} at {node.instr.loc}"
                    )
                backing_var = self._ensure_temp_storage_global_backing_var()
                if backing_var is None:
                    raise RuntimeError(
                        "TempStorage global backing allocation is missing."
                    )
                offset = int(temp_storage_info.base_offset) + int(use_info.offset)
                size = int(use_info.size_in_bytes)
                prelude, sliced = self.emit_array_slice_call(
                    node.instr.target.scope,
                    node.expr.loc,
                    backing_var,
                    offset,
                    offset + size,
                    symbol_prefix=f"$coop_temp_storage_exclusive_{node.unique_id}",
                )
                if getattr(node, "temp_storage_prelude_instrs", None) is None:
                    node.temp_storage_prelude_instrs = []
                node.temp_storage_prelude_instrs.extend(prelude)
                temp_storage = sliced
                temp_storage_ty = self.typemap[temp_storage.name]

        runtime_args.insert(insert_pos, temp_storage)
        runtime_arg_types.insert(insert_pos, temp_storage_ty)
        runtime_arg_names.insert(insert_pos, "temp_storage")
        return temp_storage, temp_storage_ty, temp_storage_info

    @staticmethod
    def _kws_to_dict(kws):
        if isinstance(kws, (list, tuple)):
            return dict(kws)
        if isinstance(kws, dict):
            return kws
        return {}

    @staticmethod
    def _var_matches(obj, var_name):
        return isinstance(obj, ir.Var) and obj.name == var_name

    def _dtype_from_var(self, obj):
        if not isinstance(obj, ir.Var):
            return None
        ty = self.typemap.get(obj.name)
        if isinstance(ty, types.Array):
            dtype = ty.dtype
            if isinstance(dtype, types.Undefined):
                return None
            return dtype
        return None

    def _infer_thread_data_dtype_candidate(self, primitive_name, bound, var_name):
        arg_pairs = THREAD_DATA_DTYPE_INFERENCE_ARG_PAIRS.get(primitive_name)
        if not arg_pairs:
            return None

        for thread_arg_name, peer_arg_name in arg_pairs:
            thread_arg = bound.arguments.get(thread_arg_name)
            if not self._var_matches(thread_arg, var_name):
                continue
            peer_arg = bound.arguments.get(peer_arg_name)
            candidate = self._dtype_from_var(peer_arg)
            if candidate is not None:
                return candidate

        return None

    def _infer_thread_data_dtype_from_uses(self, var):
        var_name = var.name
        candidates = []

        for expr in self._iter_call_exprs():
            args = expr.args
            kws = self._kws_to_dict(expr.kws)
            uses_var = any(
                isinstance(arg, ir.Var) and arg.name == var_name for arg in args
            ) or any(
                isinstance(arg, ir.Var) and arg.name == var_name for arg in kws.values()
            )
            if not uses_var:
                continue

            func_ty = self.typemap.get(expr.func.name)
            templates = getattr(func_ty, "templates", None)
            if not templates:
                continue

            template = templates[0]
            primitive_name = getattr(template, "primitive_name", None)
            if primitive_name is None:
                continue

            try:
                bound = template.signature(*args, **kws)
            except Exception:
                continue

            candidate = self._infer_thread_data_dtype_candidate(
                primitive_name,
                bound,
                var_name,
            )

            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None

        dtype = candidates[0]
        for candidate in candidates[1:]:
            if candidate != dtype:
                msg = (
                    "Could not infer a consistent dtype for ThreadData; "
                    f"found {dtype} and {candidate}"
                )
                raise RuntimeError(msg)

        return dtype

    def get_thread_data_info(self, var):
        cached = self._thread_data_info.get(var.name)
        if cached is not None:
            return cached

        root = self.get_root_def(var)
        leaf = root.leaf_constructor_call
        if not isinstance(leaf, ThreadDataCallDefinition):
            raise RuntimeError(f"Expected ThreadData call for {var!r}, got {leaf!r}")

        items_per_thread = leaf.items_per_thread
        if not isinstance(items_per_thread, int):
            raise RuntimeError(
                "items_per_thread must be a compile-time integer for ThreadData"
            )
        if items_per_thread <= 0:
            raise RuntimeError("items_per_thread must be >= 1 for ThreadData")

        dtype = None
        if leaf.dtype is not None:
            try:
                dtype = normalize_dtype_param(leaf.dtype)
            except Exception as exc:
                msg = f"Invalid dtype for ThreadData: {leaf.dtype}"
                raise RuntimeError(msg) from exc

        if dtype is None:
            dtype = self._infer_thread_data_dtype_from_uses(var)

        if dtype is None:
            msg = (
                "Could not infer dtype for ThreadData; provide dtype explicitly "
                "or use coop.local.array instead."
            )
            raise RuntimeError(msg)

        info = SimpleNamespace(
            items_per_thread=items_per_thread,
            dtype=dtype,
        )
        self._thread_data_info[var.name] = info
        return info

    def get_temp_storage_info(self, var):
        root = self.get_root_def(var)
        root_assign = root.root_assign
        if not isinstance(root_assign, ir.Assign):
            raise RuntimeError(f"Expected root assign for TempStorage var {var!r}")
        root_var = root_assign.target
        if not isinstance(root_var, ir.Var):
            raise RuntimeError(f"Expected root var for TempStorage var {var!r}")
        key = root_var.name

        cached = self._temp_storage_info.get(key)
        if cached is not None:
            return cached

        leaf = root.leaf_constructor_call
        if not isinstance(leaf, TempStorageCallDefinition):
            raise RuntimeError(f"Expected TempStorage call for {var!r}, got {leaf!r}")

        size_in_bytes = leaf.size_in_bytes
        alignment = leaf.alignment
        auto_sync = leaf.auto_sync
        sharing = leaf.sharing if leaf.sharing is not None else "shared"
        if not isinstance(sharing, str):
            raise RuntimeError(
                "TempStorage sharing must be a compile-time string literal "
                f"(got {sharing!r})"
            )
        sharing = sharing.strip().lower()
        if sharing not in ("shared", "exclusive"):
            raise RuntimeError(
                f"TempStorage sharing must be 'shared' or 'exclusive', got {sharing!r}"
            )

        requirements = []
        inference_failed = None
        needs_inference = True

        if needs_inference:
            if key in self._temp_storage_state.info_inference_stack:
                raise RuntimeError(
                    f"Recursive TempStorage inference detected for {root_var.name!r}."
                )
            self._temp_storage_state.info_inference_stack.add(key)
            try:
                requirements = self._infer_temp_storage_requirements_from_uses(root_var)
            except RuntimeError as exc:
                inference_failed = exc
            finally:
                self._temp_storage_state.info_inference_stack.remove(key)

            if inference_failed is not None:
                # If the user gave explicit size+alignment in shared mode, allow
                # unused placeholders and skip inference-derived validation.
                explicit_shared = (
                    sharing == "shared"
                    and size_in_bytes is not None
                    and alignment is not None
                )
                if not explicit_shared or self._temp_storage_var_is_used(root_var):
                    raise inference_failed

        required_alignment = (
            max(req.alignment for req in requirements) if requirements else 1
        )
        required_shared_size = (
            max(req.size_in_bytes for req in requirements) if requirements else 0
        )
        use_layout: dict[int, TempStorageUseLayoutEntry] = {}

        if sharing == "shared":
            required_size = required_shared_size
            for req in requirements:
                use_layout[req.call_key] = TempStorageUseLayoutEntry(
                    offset=0,
                    size_in_bytes=req.size_in_bytes,
                    alignment=req.alignment,
                    primitive_name=req.primitive_name,
                )
        else:
            required_size = 0
            for req in requirements:
                required_size = self._align_up(required_size, req.alignment)
                offset = required_size
                use_layout[req.call_key] = TempStorageUseLayoutEntry(
                    offset=offset,
                    size_in_bytes=req.size_in_bytes,
                    alignment=req.alignment,
                    primitive_name=req.primitive_name,
                )
                required_size = offset + req.size_in_bytes

        if size_in_bytes is None:
            size_in_bytes = required_size
        if not isinstance(size_in_bytes, int) or size_in_bytes <= 0:
            raise RuntimeError("TempStorage size_in_bytes must be a positive integer")
        if required_size > 0 and size_in_bytes < required_size:
            raise RuntimeError(
                "TempStorage size_in_bytes is smaller than required by primitive "
                f"uses ({size_in_bytes} < {required_size})."
            )

        if alignment is None:
            alignment = required_alignment
        if not isinstance(alignment, int) or alignment <= 0:
            raise RuntimeError("TempStorage alignment must be a positive integer")

        pointer_size = struct.calcsize("P")
        if alignment % pointer_size != 0:
            alignment = self._align_up(alignment, pointer_size)
        if required_alignment > 0 and alignment < required_alignment:
            raise RuntimeError(
                "TempStorage alignment is smaller than required by primitive uses "
                f"({alignment} < {required_alignment})."
            )

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise RuntimeError("TempStorage auto_sync must be None/True/False")
        if sharing == "exclusive":
            if auto_sync is True:
                raise RuntimeError(
                    "TempStorage with sharing='exclusive' does not support "
                    "auto_sync=True."
                )
            effective_auto_sync = False
        else:
            effective_auto_sync = True if auto_sync is None else auto_sync

        info = TempStorageInfo(
            key=key,
            root_var=root_var,
            order=leaf.order if leaf.order is not None else 0,
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            required_size=required_size,
            required_alignment=required_alignment,
            use_layout=use_layout,
            requirements=requirements,
            sharing=sharing,
            auto_sync=effective_auto_sync,
        )
        self._temp_storage_info[key] = info
        return info

    def emit_cuda_array_call(self, scope, loc, shape, dtype, alignment, shared, target):
        new_nodes = []

        shape = get_element_count(shape)

        new_shape_name = ir_utils.mk_unique_var("$coop_array_shape")
        new_shape_var = ir.Var(scope, new_shape_name, loc)
        self.typemap[new_shape_var.name] = types.IntegerLiteral(shape)
        new_shape_assign = ir.Assign(
            value=ir.Const(shape, loc),
            target=new_shape_var,
            loc=loc,
        )
        new_nodes.append(new_shape_assign)

        dtype_attr = str(dtype)
        new_dtype_name = ir_utils.mk_unique_var("$coop_array_dtype")
        new_dtype_var = ir.Var(scope, new_dtype_name, loc)
        dtype_ty = types.DType(dtype)
        self.typemap[new_dtype_var.name] = dtype_ty
        if hasattr(numba, dtype_attr):
            g_numba_module_assign = self.get_or_create_global_numba_module_instr(
                scope,
                loc,
                new_nodes,
            )
            dtype_getattr = ir.Expr.getattr(
                value=g_numba_module_assign.target,
                attr=dtype_attr,
                loc=loc,
            )
            dtype_assign = ir.Assign(
                value=dtype_getattr,
                target=new_dtype_var,
                loc=loc,
            )
        else:
            dtype_assign = ir.Assign(
                value=ir.Global(dtype_attr, dtype, loc),
                target=new_dtype_var,
                loc=loc,
            )
        new_nodes.append(dtype_assign)

        if alignment is not None:
            alignment_name = ir_utils.mk_unique_var("$coop_array_alignment")
            alignment_var = ir.Var(scope, alignment_name, loc)
            self.typemap[alignment_var.name] = types.IntegerLiteral(alignment)
            alignment_assign = ir.Assign(
                value=ir.Const(alignment, loc),
                target=alignment_var,
                loc=loc,
            )
            new_nodes.append(alignment_assign)
        else:
            alignment_var = None

        g_numba_cuda_module_assign = self.get_or_create_global_numba_cuda_module_instr(
            scope,
            loc,
            new_nodes,
        )

        attr_name = "shared" if shared else "local"
        array_module = getattr(numba.cuda, attr_name)
        array_decl_name = f"Cuda_{attr_name}_array"
        array_decl = getattr(numba.cuda.cudadecl, array_decl_name)
        array_decl_ty = types.Function(array_decl)

        if shared:
            mod_ty = CudaSharedModuleTemplate
        else:
            mod_ty = CudaLocalModuleTemplate

        mod = mod_ty(context=None)
        array_func_ty = mod.resolve_array(None)
        assert array_decl_ty is array_func_ty, (array_decl_ty, array_func_ty)

        array_func_template_class = array_func_ty.templates[0]
        instance = array_func_template_class(context=None)
        typer = instance.generic()

        args = [types.IntegerLiteral(shape), dtype_ty]
        if alignment is not None:
            args.append(types.IntegerLiteral(alignment))

        return_type = typer(*args)
        if not isinstance(return_type, types.Array):
            msg = f"Expected an array type, got {return_type!r}"
            raise RuntimeError(msg)

        array_func_sig = Signature(return_type, tuple(args), recvr=None, pysig=None)
        array_decl_ty.get_call_type(self.typingctx, args=array_func_sig.args, kws={})
        check = array_decl_ty._impl_keys[array_func_sig.args]
        assert check is not None, check

        module_attr_getattr = ir.Expr.getattr(
            value=g_numba_cuda_module_assign.target,
            attr=attr_name,
            loc=loc,
        )
        module_attr_var_name = ir_utils.mk_unique_var(f"$coop_{attr_name}_module")
        module_attr_var = ir.Var(scope, module_attr_var_name, loc)
        self.typemap[module_attr_var.name] = types.Module(array_module)
        module_attr_assign = ir.Assign(
            value=module_attr_getattr,
            target=module_attr_var,
            loc=loc,
        )
        new_nodes.append(module_attr_assign)

        array_attr_getattr = ir.Expr.getattr(
            value=module_attr_var,
            attr="array",
            loc=loc,
        )
        array_attr_var_name = f"{module_attr_var_name}_array_getattr"
        array_attr_var = ir.Var(scope, array_attr_var_name, loc)
        self.typemap[array_attr_var.name] = array_decl_ty
        array_attr_assign = ir.Assign(
            value=array_attr_getattr,
            target=array_attr_var,
            loc=loc,
        )
        new_nodes.append(array_attr_assign)

        new_args = [new_shape_var, new_dtype_var]
        if alignment_var is not None:
            new_args.append(alignment_var)

        array_func_var_name = f"{module_attr_var_name}_array_call"
        array_func_var = ir.Var(scope, array_func_var_name, loc)
        self.typemap[array_func_var.name] = array_func_ty
        array_func_call = ir.Expr.call(
            func=array_func_var,
            args=tuple(new_args),
            kws=(),
            loc=loc,
        )
        self.calltypes[array_func_call] = array_func_sig
        if target.name in self.typemap:
            del self.typemap[target.name]
        self.typemap[target.name] = array_func_sig.return_type
        array_func_assign = ir.Assign(
            value=array_func_call,
            target=target,
            loc=loc,
        )
        new_nodes.append(array_func_assign)

        return new_nodes

    def emit_syncthreads_call(self, scope, loc):
        new_nodes = []
        g_numba_cuda_module_assign = self.get_or_create_global_numba_cuda_module_instr(
            scope,
            loc,
            new_nodes,
        )

        syncthreads_getattr = ir.Expr.getattr(
            value=g_numba_cuda_module_assign.target,
            attr="syncthreads",
            loc=loc,
        )
        syncthreads_var_name = ir_utils.mk_unique_var("$cuda_syncthreads")
        syncthreads_var = ir.Var(scope, syncthreads_var_name, loc)
        syncthreads_ty = self.typingctx.resolve_value_type(numba.cuda.syncthreads)
        self.typemap[syncthreads_var.name] = syncthreads_ty
        syncthreads_assign = ir.Assign(
            value=syncthreads_getattr,
            target=syncthreads_var,
            loc=loc,
        )
        new_nodes.append(syncthreads_assign)

        call_expr = ir.Expr.call(
            func=syncthreads_var,
            args=(),
            kws=(),
            loc=loc,
        )
        call_sig = syncthreads_ty.get_call_type(self.typingctx, args=(), kws={})
        self.calltypes[call_expr] = call_sig

        call_var_name = ir_utils.mk_unique_var("$cuda_syncthreads_call")
        call_var = ir.Var(scope, call_var_name, loc)
        self.typemap[call_var.name] = call_sig.return_type
        call_assign = ir.Assign(
            value=call_expr,
            target=call_var,
            loc=loc,
        )
        new_nodes.append(call_assign)

        return new_nodes

    def next_unique_id(self):
        return next(self._unique_id_counter)

    def next_symbol_id(self):
        return next(self._symbol_id_counter)

    def get_symbol_id(self, key):
        symbol_id = self._symbol_id_map.get(key)
        if symbol_id is None:
            symbol_id = self.next_symbol_id()
            self._symbol_id_map[key] = symbol_id
        return symbol_id

    def maybe_coalesce_algo(self, node, algo):
        if not self._bundle_ltoir_enabled:
            node.symbol_id = node.unique_id
            algo.unique_id = node.unique_id
            return
        key = algo_coalesce_key(algo, include_target_name=False)
        node.coalesce_key = key
        symbol_id = self.get_symbol_id(key)
        node.symbol_id = symbol_id
        if node.symbol_name is None:
            node.symbol_name = algo.c_name
        algo.unique_id = symbol_id
        for child in getattr(node, "children", ()) or ():
            child_instance = getattr(child, "instance", None)
            if child_instance is None:
                continue
            child_algo = getattr(child_instance, "specialization", None)
            if child_algo is None:
                continue
            child_algo.unique_id = symbol_id

    def get_or_create_parent_node(
        self,
        func_ir: ir.FunctionIR,
        current_block: ir.Block,
        parent_target_name: str,
        parent_root_def: RootDefinition,
        calltypes: dict[ir.Expr, types.Type],
        typemap: dict[str, types.Type],
        launch_config,
        child_expr: ir.Expr,
        child_template: Any,
    ) -> CoopNode:
        if parent_target_name in self.nodes:
            return self.nodes[parent_target_name]

        # Create a new CoopNode instance for the parent.
        needs_pre_launch_callback = parent_root_def.needs_pre_launch_callback

        primitive_name = parent_root_def.primitive_name
        node_class = self._node_classes[primitive_name]
        decl = self.decl_class_by_primitive_name[primitive_name]
        template = decl
        impl_class = self._decl_classes[decl]

        root_assign = parent_root_def.root_assign
        root_block = self.all_assignments[root_assign]
        expr = root_assign.value
        if isinstance(expr, ir.Expr) and expr.op == "call":
            func_name = expr.func.name
            func = typemap[func_name]
        else:
            func_name = parent_target_name
            func = typemap[parent_target_name]
        target = root_assign.target

        two_phase_instance = None
        if not isinstance(parent_root_def.instance, PyModuleType):
            two_phase_instance = parent_root_def.instance
        type_instance = None
        if two_phase_instance is not None:
            parent_type = typemap.get(parent_target_name)
            if parent_type in self._instances:
                type_instance = parent_type

        node = node_class(
            index=0,
            block_line=root_block.loc.line,
            expr=expr,
            # instr=parent_root_def.root_instr,
            instr=root_assign,
            template=template,
            func=func,
            func_name=func_name,
            impl_class=impl_class,
            target=target,
            calltypes=calltypes,
            typemap=typemap,
            func_ir=func_ir,
            typingctx=self._state.typingctx,
            launch_config=launch_config,
            needs_pre_launch_callback=needs_pre_launch_callback,
            unique_id=self.next_unique_id(),
            type_instance=type_instance,
            two_phase_instance=two_phase_instance,
            parent_struct_instance_type=None,
            parent_node=None,
            children=[],
            rewriter=self,
            child_expr=child_expr,
            child_template=child_template,
            root_def=parent_root_def,
        )

        self.nodes[parent_target_name] = node
        node.refine_match(self)
        if (
            two_phase_instance is not None
            and getattr(two_phase_instance, "node", None) is None
        ):
            two_phase_instance.node = node

        return node

    def handle_new_kernel_traits_struct(self, struct, name: str, launch_config):
        # N.B. See the comment in the `match()` method for more details about
        #      the purpose of this method and the `CustomPrepareArgs` class.
        needs_custom = not hasattr(struct, "prepare_args") or not hasattr(
            struct, "pre_launch_callback"
        )
        if not needs_custom:
            # Use the struct directly.
            custom = struct
        else:
            # Synthesize a custom prepare_args handler for the struct.
            class CustomPrepareArgs:
                """
                Custom prepare_args handler for the struct-like object.
                This will be added to the kernel's extensions list.
                """

                def __init__(self, struct: Any, name: str):
                    self.struct = struct
                    self.name = name

                def pre_launch_callback(self, kernel, launch_config):
                    register_kernel_extension(kernel, self)

                def prepare_args(self, ty, val, *args, **kwds):
                    if val is not self.struct:
                        return (ty, val)

                    # The values we return here just need to pacify _Kernel's
                    # _parse_args() routines--we never actually use the kernel
                    # parameters by way of the arguments provided at kernel
                    # launch.
                    addr = id(val)
                    return (types.uint64, addr)

            custom = CustomPrepareArgs(struct, name)

        launch_config.pre_launch_callbacks.append(custom.pre_launch_callback)

    @property
    def launch_config(self):
        config = _ensure_current_launch_config()
        config.dispatcher.mark_launch_config_sensitive()
        return config

    @cached_property
    def all_assignments(self) -> dict[ir.Assign, ir.Block]:
        """
        Returns a dict that maps all `ir.Assign` instructions in the function
        IR to their corresponding `ir.Block`.
        """
        assignments = {}
        for block in self.func_ir.blocks.values():
            assignments.update({a: block for a in block.find_insts(ir.Assign)})
        return assignments

    @cached_property
    def impl_to_decl_classes(self):
        decl_classes = self._decl_classes
        impl_to_decl_classes = {}
        for decl, impl in decl_classes.items():
            # Keep the first mapping for a given impl; duplicates are expected
            # for method-style helpers (e.g. init/composite).
            impl_to_decl_classes.setdefault(impl, decl)
        return impl_to_decl_classes

    @cached_property
    def decl_class_by_primitive_name(self):
        decl_classes = self._decl_classes
        decl_by_name = {}
        for decl in decl_classes.keys():
            name = decl.primitive_name
            if name not in decl_by_name:
                decl_by_name[name] = decl
                continue
            existing = decl_by_name[name]
            if (
                existing.__module__ != "cuda.coop._decls"
                and decl.__module__ == "cuda.coop._decls"
            ):
                decl_by_name[name] = decl
        return decl_by_name

    @cached_property
    def assignments_map(self):
        """
        Returns a map of `ir.Assign` values to their corresponding `ir.Assign`
        node.  This allows us to find any assignment to a variable in any block
        at any point during match processing (i.e. handy for finding assignments
        in blocks other than the current one being processed by `match()`).
        """
        assignments = self.all_assignments
        return {assign.value: assign for assign in assignments.keys()}

    def match(self, func_ir, block, typemap, calltypes, **kw):
        # If there are no calls in this block, we can immediately skip it.
        num_calltypes = len(calltypes)
        if num_calltypes == 0:
            return False

        try:
            launch_config = self.launch_config
        except RuntimeError:
            return False

        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes

        self.current_block = block

        # We return this value at the end of the routine.  A true value
        # indicates that we want `apply()` to be called after we return
        # from this routine in order to rewrite the block we just processed.
        we_want_apply = False

        decl_classes = self._decl_classes
        node_classes = self._node_classes
        type_instances = self._instances
        interesting_modules = self.interesting_modules

        # N.B. I keep thinking I need these, but then never end up using them.
        #      Keeping for now so I can easily uncomment if I do actually end
        #      up needing them.
        #
        # py_func = func_ir.func_id.func
        # code_obj = py_func.__code__
        # free_vars = code_obj.co_freevars
        # cells = py_func.__closure__

        found = False
        block_offset = None
        for block_no, (offset, ir_block) in enumerate(func_ir.blocks.items()):
            if ir_block is block:
                found = True
                block_offset = offset
                break

        if not found:
            raise RuntimeError(
                f"Block {block!r} not found in function IR blocks: "
                f"{list(func_ir.blocks.keys())}"
            )

        self._all_match_invocations_count += 1
        self._match_invocations_per_block_offset[block_offset] += 1

        invocation_count = self._match_invocations_per_block_offset[block_offset]

        for i, instr in enumerate(block.body):
            # We're only interested in ir.Assign nodes.  Skip the rest.
            if not isinstance(instr, ir.Assign):
                continue

            # `instr.value` will hold the right-hand side of the assignment.
            # The left-hand side (i.e. the `foo` in `foo = bar`) is accessed
            # via `instr.target`.  Initialize aliases.
            rhs = instr.value
            target = instr.target
            target_name = target.name

            node = self.nodes.get(target_name, None)
            if node:
                # If the node is a parent node, it must have been created
                # in response to a child primitive invocation being processed.
                if node.is_parent:
                    # Check to see if we should still register for a rewrite.
                    if node.wants_rewrite and not node.has_been_rewritten:
                        we_want_apply = True

                    # If we're processing the genesis instruction for the
                    # parent (i.e. the first invocation of the primitive),
                    # the node's `instr.loc` will match the current
                    # instruction's loc.
                    if node.instr.loc == instr.loc:
                        # Check to see if we need to kick-off codegen for
                        # the parent and all children nodes.
                        if node.wants_codegen and not node.has_been_codegened:
                            node.codegen_callback()
                    continue

                # If we're not a parent, the only way we could have a node
                # for this target is if we're getting reinvoked against a
                # block we've already processed (which can happen if a
                # rewrite occurs).  Thus, our invocation count should be
                # greater than 1, and block line and root instruction should
                # match the current block and instruction.  Verify these
                # critical invariants now.
                assert invocation_count > 0
                if invocation_count == 1:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        "already exists, but this is the first invocation "
                        "of the rewriter.match() method for this block."
                    )

                if node.block_line != block.loc.line:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        f"has a different block line ({node.block_line}) "
                        f"than the current block ({block.loc.line})"
                    )

                if node.instr.loc != instr.loc:
                    raise RuntimeError(
                        f"Node {node.shortname} for target {target_name!r} "
                        f"has a different instruction location "
                        f"({node.instr.loc}) than the current instruction "
                        f"({instr.loc})"
                    )

                continue

            if isinstance(rhs, ir.Global) and isinstance(rhs.value, PyModuleType):
                module_name = rhs.value.__name__
                if (
                    module_name in interesting_modules
                    and module_name not in self._modules
                ):
                    self._modules[module_name] = instr
                continue

            if not isinstance(rhs, ir.Expr):
                continue

            expr = rhs

            if expr.op == "getattr":
                base_var = expr.value
                if isinstance(base_var, ir.Var):
                    try:
                        root = self.get_root_def(base_var)
                    except Exception:
                        root = None
                    if root is not None and isinstance(root.root_instr, ir.Arg):
                        arg_name = root.root_instr.name
                        if arg_name not in self.seen_structs:
                            struct = root.instance
                            if struct is not None and (
                                getattr(struct, "__cuda_coop_gpu_dataclass__", False)
                                or (
                                    hasattr(struct, "prepare_args")
                                    and hasattr(struct, "pre_launch_callback")
                                )
                            ):
                                self.handle_new_kernel_traits_struct(
                                    struct, arg_name, launch_config
                                )
                                self.seen_structs.add(arg_name)
                continue

            # We can ignore nodes that aren't function calls herein.
            if expr.op != "call":
                continue

            func_name = expr.func.name
            func = typemap.get(func_name)
            if func is None:
                continue

            func_obj = None
            try:
                func_def = func_ir.get_definition(func_name)
            except KeyError:
                continue
            if isinstance(func_def, (ir.Global, ir.FreeVar)):
                func_obj = func_def.value

            py_func = getattr(func, "typing_key", None)
            if py_func is coop.ThreadData or func_obj is coop.ThreadData:
                node = CoopThreadDataNode(
                    expr=expr,
                    instr=instr,
                    target=target,
                    func_ir=func_ir,
                    typemap=typemap,
                    calltypes=calltypes,
                    block_line=block.loc.line,
                )
                self.nodes[target_name] = node
                node.refine_match(self)
                we_want_apply = True
                continue

            if py_func is coop.TempStorage or func_obj is coop.TempStorage:
                node = CoopTempStorageNode(
                    expr=expr,
                    instr=instr,
                    target=target,
                    func_ir=func_ir,
                    typemap=typemap,
                    calltypes=calltypes,
                    block_line=block.loc.line,
                )
                self.nodes[target_name] = node
                node.refine_match(self)
                we_want_apply = True
                continue

            # Reset our per loop iteration local variables.
            decl = None
            template = None
            impl_class = None
            node_class = None
            value_type = None
            parent_node = None
            type_instance = None
            parent_node = None
            primitive_name = None
            two_phase_instance = None
            parent_struct_instance_type = None
            needs_pre_launch_callback = False

            # Attempt to obtain the implementation class based on whether we're
            # dealing with an instance or a declaration/template.  Specifically,
            # we want to obtain the implementation class (`impl_class`) and node
            # rewriting class for this function call.
            #
            # For example, if we're dealing with a block load operation, the
            # `impl_class` will be `coop.block.load` and the `node_class` will
            # be `CoopBlockLoadNode`.
            #
            # If we're dealing with a single-phase primitive instance, we go
            # via the function's templates, looking for the first template that
            # has an implementation class registered.
            #
            # If we're dealing with a two-phase primitive instance, `func` will
            # be in our `type_instances` map, and we can obtain the definition
            # by way of the `func_ir.get_definition(func_name)` call.
            #
            # If neither of these conditions are met, we've hit an unexpected
            # code path, and we raise an error.
            if hasattr(func, "templates"):
                templates = func.templates
                if len(templates) != 1:
                    # Our primitives will never have more than one template,
                    # so this isn't one of ours.
                    continue

                template = templates[0]
                impl_class = decl_classes.get(template, None)
                if not impl_class:
                    # This isn't one of our primitives, so we can skip it.
                    continue

                primitive_name = template.primitive_name

            elif func in type_instances:
                # We're dealing with a two-phase primitive instance.
                type_instance = func
                def_instr = func_ir.get_definition(func_name)
                was_getattr = False
                if isinstance(def_instr, (ir.Global, ir.FreeVar)):
                    # Globals and free variables are easy; we can get the
                    # instance directly from the instruction's value.
                    two_phase_instance = def_instr.value
                elif isinstance(def_instr, ir.Arg):
                    # For arguments, we can get the instance from the launch
                    # configuration arguments.
                    two_phase_instance = get_kernel_param_value(
                        def_instr.name,
                        launch_config,
                    )
                    needs_pre_launch_callback = True
                elif isinstance(def_instr, ir.Expr):
                    if def_instr.op == "getattr":
                        was_getattr = True
                        obj = def_instr.value
                        obj_instr = func_ir.get_definition(obj.name)
                        if isinstance(obj_instr, (ir.Global, ir.FreeVar)):
                            two_phase_instance = obj_instr.value
                        elif isinstance(obj_instr, ir.Arg):
                            # Assume we're dealing with a kernel traits "struct"
                            # (see earlier long comment for details).
                            arg_value = get_kernel_param_value(
                                obj_instr.name,
                                launch_config,
                            )
                            if obj.name not in self.seen_structs:
                                # If we haven't seen this struct before, we
                                # need to create a new class for it.
                                self.handle_new_kernel_traits_struct(
                                    arg_value, obj.name, launch_config
                                )
                                self.seen_structs.add(obj.name)
                            two_phase_instance = getattr(arg_value, def_instr.attr)

                            # N.B. We don't need to set `needs_pre_launch_callback`
                            #      here; that's only required when the primitive
                            #      instance has been provided directly as a
                            #      kernel argument.
                    else:
                        msg = f"Unexpected expression op: {def_instr!r}"
                        raise RuntimeError(msg)
                else:
                    msg = f"Unexpected instruction type: {def_instr!r}"
                    raise RuntimeError(msg)

                root_def = self.get_root_def(def_instr)
                if was_getattr:
                    if root_def.attr_instance is not two_phase_instance:
                        msg = (
                            "Invariant violation: getattr root attribute "
                            "instance mismatch; "
                            f"got {root_def.attr_instance!r}, "
                            f"expected {two_phase_instance!r}"
                        )
                        raise RuntimeError(msg)
                else:
                    if root_def.instance is not two_phase_instance:
                        msg = (
                            "Invariant violation: root instance mismatch; "
                            f"got {root_def.instance!r}, "
                            f"expected {two_phase_instance!r}"
                        )
                        raise RuntimeError(msg)

                value_type = typemap[func_name]
                primitive_name = repr(value_type)
                # Example values at this point for e.g. block load:
                #
                #   >>> two_phase_instance
                #   <cuda.coop.block.\
                #       _block_load_store.load object at 0x757ed7f44f70>
                #
                #   >>> value_type
                #   coop.block.load
                #
                #   >>> type(value_type)
                #   <class 'cuda.coop.\
                #       _decls.CoopBlockLoadInstanceType'>
                #
                #   >>> primitive_name
                #   'coop.block.load'

                decl = value_type.decl
                template = decl.__class__
                impl_class = decl_classes.get(template, None)
                if not impl_class:
                    msg = (
                        f"Could not find declaration class for decl: {decl!r}"
                        f" (template: {template!r}, primitive_name: "
                        f"{primitive_name!r})"
                    )
                    raise RuntimeError(msg)

                # We've derived everything we need for the two-phase invocation;
                # the remaining code simply verifies various invariants.

                # Sanity check the primitive name via the decl.
                assert primitive_name == decl.primitive_name, (
                    primitive_name,
                    decl.primitive_name,
                    decl,
                )

                # Sanity-check names are correct.
                assert type_instance.name == decl.primitive_name, (
                    type_instance.name,
                    decl.primitive_name,
                )
                assert type_instance.name.endswith(decl.key.__name__), (
                    type_instance.name,
                    decl.key.__name__,
                )

            elif isinstance(func, types.BoundFunction):
                # BoundFunction is used for primitives that expose one or
                # more methods against the containing parent struct, e.g.
                # `histo.composite()`.  It'll look something like this:
                #
                #   >>> func
                #   BoundFunction(coop.block.histogram.composite for \
                #                 coop.block.histogram)
                #
                # It will be called against an instance of a prior primitive
                # that should have already had a node created and registered
                # in `self.nodes` by way of the target name.  We want to get
                # this instance so it can be included in the new node
                # construction below, so we first have to find the target name.
                #
                # How do we do that?  Let's start with what the simplified IR
                # would have looked like up to this point:
                #
                #   1: block_histogram = freevar(block_histogram: <coop.block.hist...>
                #   2: histo = call block_histogram()
                #   3: histo_init = getattr(value=histo, attr=init)
                #   4: call histo_init(smem_histogram, ...)
                #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # The last call line is the instruction we're currently
                # handling.  We get the definition of the `histo_init` variable
                # being called, which will return us the instruction for the
                # corresponding getattr() against the parent object; item 3
                # above.
                #
                # We can then see that the getattr() is being called against
                # the `histo` variable, so we look up the definition of that
                # next, which returns us the instruction for item 2 above, when
                # the `histo` instance was created/constructed.
                #
                # Thus, our parent's target name is 'histo', and we can get
                # that directly from our `self.nodes` map.

                template = func.template
                if not hasattr(template, "primitive_name"):
                    # This isn't one of our bound functions.
                    continue
                primitive_name = template.primitive_name
                parent_struct_instance_type = func.key[1]

                # Find the `histo_init` variable's definition, which will return
                # the instruction for the getattr() call (item 3 above), e.g.:
                # `getattr(value=histo, attr=init)`.
                getattr_expr = func_ir.get_definition(func_name)

                # Verify that we're dealing with a `getattr()` expression, and
                # that the attribute being accessed matches the end of our
                # primitive name (i.e. `init` for `coop.block.histogram.init`).
                ignore_instruction = (
                    not isinstance(getattr_expr, ir.Expr)
                    or getattr_expr.op != "getattr"
                    or not primitive_name.endswith(getattr_expr.attr)
                )
                if ignore_instruction:
                    # Not one of our primitives, ignore and continue.
                    continue

                # The `getattr_expr.value` will be the parent object for which
                # the `getattr()` was called, e.g. `histo` in the example.
                parent_obj = getattr_expr.value
                parent_target_name = parent_obj.name

                # Get the root definition of the parent object.
                parent_root_def = self.get_root_def(parent_obj)
                if not parent_root_def:
                    raise RuntimeError(
                        f"Could not find root definition for {parent_target_name!r}"
                    )

                parent_node = self.get_or_create_parent_node(
                    func_ir,
                    block,
                    parent_target_name,
                    parent_root_def,
                    calltypes,
                    typemap,
                    launch_config,
                    expr,
                    template,
                )

                impl_class = func.key[0]

            else:
                # Not something we recognize; continue.
                continue

            # We can now obtain the node rewriting class from the primitive
            # name.
            node_class = node_classes[primitive_name]

            # If we reach here, impl_class and template should be non-None.
            assert impl_class is not None
            assert template is not None

            # Invariant checks: if we have a type instance we must have a
            # two-phase instance, and if we don't, we shouldn't.
            if type_instance is not None:
                if two_phase_instance is None:
                    # If we have a type instance, we must have a two-phase
                    # instance.
                    msg = (
                        "Invariant check failed: type_instance is set, "
                        "but two_phase_instance is None; "
                        f"{type_instance!r} ({primitive_name!r})"
                    )
                    raise RuntimeError(msg)
            elif two_phase_instance is not None:
                msg = (
                    "Invariant check failed: two_phase_instance is set, "
                    "but type_instance is None; "
                    f"{two_phase_instance!r} ({primitive_name!r})"
                )
                raise RuntimeError(msg)

            node = self.nodes.get(target_name, None)
            if node is not None:
                # We shouldn't ever hit this due to our invariant checks
                # earlier in the routine.
                raise RuntimeError(
                    f"Node for target name {target_name!r} already exists: {node!r}"
                )

            # Get our root definition, plus the parent root definition if
            # applicable.
            root_def = self.get_root_def(instr)
            if parent_node is not None:
                parent_root_def = parent_node.root_def
            else:
                parent_root_def = None

            node = node_class(
                index=i,
                block_line=block.loc.line,
                expr=expr,
                instr=instr,
                template=template,
                func=func,
                func_name=func_name,
                impl_class=impl_class,
                target=instr.target,
                calltypes=calltypes,
                typemap=typemap,
                func_ir=func_ir,
                typingctx=self._state.typingctx,
                launch_config=launch_config,
                needs_pre_launch_callback=needs_pre_launch_callback,
                unique_id=self.next_unique_id(),
                type_instance=type_instance,
                two_phase_instance=two_phase_instance,
                root_def=root_def,
                parent_struct_instance_type=parent_struct_instance_type,
                parent_node=parent_node,
                parent_root_def=parent_root_def,
                rewriter=self,
            )

            if two_phase_instance is not None:
                if two_phase_instance.node is None:
                    # Register the node with the two-phase instance so we can
                    # access it during lowering.
                    two_phase_instance.node = node

            target_name = node.target.name
            assert target_name not in self.nodes, target_name
            self.nodes[target_name] = node
            node.refine_match(self)

            if not we_want_apply:
                if node.wants_rewrite and not node.has_been_rewritten:
                    # If the node wants a rewrite, request apply.
                    we_want_apply = True

        self.current_block_no = block_no
        return we_want_apply

    def apply(self):
        new_block = ir.Block(self.current_block.scope, self.current_block.loc)

        if (
            not self._temp_storage_state.global_backing_inserted
            and self._temp_storage_state.global_backing_prelude_instrs
        ):
            entry_label = min(self.func_ir.blocks.keys())
            if self.current_block_no == entry_label:
                for instr in self._temp_storage_state.global_backing_prelude_instrs:
                    new_block.append(instr)
                self._temp_storage_state.global_backing_inserted = True
            else:
                # Fallback for traversal orders where apply() is reached on a
                # non-entry block first.
                for instr in self._temp_storage_state.global_backing_prelude_instrs:
                    new_block.append(instr)
                self._temp_storage_state.global_backing_inserted = True

        skipped = 0
        ignored = 0
        rewrote = 0
        desugared_getitems = 0
        no_new_instructions = 0

        for instr in self.current_block.body:
            if not isinstance(instr, ir.Assign):
                # If the instruction is not an assignment, copy it verbatim.
                new_block.append(instr)
                ignored += 1
                continue

            expr = instr.value
            getitem_temp_storage = self._getitem_expr_temp_storage_arg(expr)
            if getitem_temp_storage is not None and isinstance(expr.value, ir.Var):
                # Desugar `primitive[temp_storage]` into a plain alias of the
                # primitive callable so no getitem IR reaches lowering.
                replacement = ir.Assign(
                    value=expr.value,
                    target=instr.target,
                    loc=instr.loc,
                )
                source_ty = self.typemap.get(expr.value.name)
                if source_ty is not None:
                    target_name = instr.target.name
                    existing = self.typemap.get(target_name)
                    if existing != source_ty:
                        if existing is not None:
                            del self.typemap[target_name]
                        self.typemap[target_name] = source_ty
                self.calltypes.pop(expr, None)
                new_block.append(replacement)
                desugared_getitems += 1
                continue

            # We're dealing with an assignment instruction.  See if we
            # created a node for the target name.
            target_name = instr.target.name
            node = self.nodes.get(target_name, None)
            if not node:
                new_block.append(instr)
                ignored += 1
                continue

            if not node.wants_rewrite or node.has_been_rewritten:
                # If the node doesn't want a rewrite, or it has already
                # been rewritten, we can just copy the instruction
                # verbatim.
                new_block.append(instr)
                skipped += 1
                continue

            results = node.rewrite(self)
            node.has_been_rewritten = True
            if results:
                rewrote += 1
                prelude_instrs = getattr(node, "temp_storage_prelude_instrs", None)
                if prelude_instrs:
                    for prelude_instr in prelude_instrs:
                        new_block.append(prelude_instr)
                for new_instr in results:
                    new_block.append(new_instr)
            else:
                no_new_instructions += 1
                new_block.append(instr)

        return new_block
