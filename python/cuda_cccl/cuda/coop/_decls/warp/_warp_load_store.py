# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core.imputils import lower_constant
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ...warp._warp_load_store import (
    _make_load_rewrite,
    _make_store_rewrite,
)
from .. import (
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopSimpleInstanceType,
    CoopWarpLoadStoreBaseTemplate,
    WarpLoadMixin,
    WarpStoreMixin,
    register,
    register_global,
)


# =============================================================================
# Load / Store
# =============================================================================
@register_global(coop.warp.load)
class CoopWarpLoadDecl(CoopWarpLoadStoreBaseTemplate, WarpLoadMixin, CoopDeclMixin):
    key = coop.warp.load
    impl_key = _make_load_rewrite
    primitive_name = "coop.warp.load"
    algorithm_enum = coop.WarpLoadAlgorithm
    default_algorithm = coop.WarpLoadAlgorithm.DIRECT


@register_global(coop.warp.store)
class CoopWarpStoreDecl(CoopWarpLoadStoreBaseTemplate, WarpStoreMixin, CoopDeclMixin):
    key = coop.warp.store
    impl_key = _make_store_rewrite
    primitive_name = "coop.warp.store"
    algorithm_enum = coop.WarpStoreAlgorithm
    default_algorithm = coop.WarpStoreAlgorithm.DIRECT


# =============================================================================
# Instance-related Load & Store Scaffolding
# =============================================================================
class CoopWarpLoadInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpLoadDecl


warp_load_instance_type = CoopWarpLoadInstanceType()


@typeof_impl.register(coop.warp.load)
def typeof_warp_load_instance(*args, **kwargs):
    return warp_load_instance_type


@register
class CoopWarpLoadInstanceDecl(CoopInstanceTemplate):
    key = warp_load_instance_type
    instance_type = warp_load_instance_type
    primitive_name = "coop.warp.load"


@register_model(CoopWarpLoadInstanceType)
class CoopWarpLoadInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpLoadInstanceType)
def lower_constant_warp_load_instance_type(context, builder, typ, value):
    return context.get_dummy_value()


class CoopWarpStoreInstanceType(CoopSimpleInstanceType):
    decl_class = CoopWarpStoreDecl


warp_store_instance_type = CoopWarpStoreInstanceType()


@typeof_impl.register(coop.warp.store)
def typeof_warp_store_instance(*args, **kwargs):
    return warp_store_instance_type


@register
class CoopWarpStoreInstanceDecl(CoopInstanceTemplate):
    key = warp_store_instance_type
    instance_type = warp_store_instance_type
    primitive_name = "coop.warp.store"


@register_model(CoopWarpStoreInstanceType)
class CoopWarpStoreInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopWarpStoreInstanceType)
def lower_constant_warp_store_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
