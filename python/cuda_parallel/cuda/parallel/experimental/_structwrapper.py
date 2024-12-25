import dataclasses
import operator

import numba
import numpy as np

from numba import types
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
)
from numba.core import cgutils
from numba.core.typing import signature
from numba.core.typing.templates import AttributeTemplate, CallableTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower as cuda_lower


def wrap_struct(dtype: np.dtype) -> numba.types.Type:
    """
    Wrap the given numpy structure dtype in a numba type.
    """
    Wrapper = dataclasses.make_dataclass(
        "wrapper",
        [(name, dt) for name, (dt, _) in dtype.fields.items()],  # type: ignore
    )

    class WrapperType(types.Type):
        def __init__(self):
            super().__init__(name="StructWrapper")

    this_type = WrapperType()

    @typeof_impl.register(Wrapper)
    def typeof_wrapper(val, c):
        return WrapperType()

    class WrapperAttrsTemplate(AttributeTemplate):
        pass

    fields = dataclasses.fields(Wrapper)
    for f in fields:
        name = f.name
        typ = f.type

        def resolver(self, this):
            return numba.from_dtype(typ)

        setattr(WrapperAttrsTemplate, f"resolve_{name}", resolver)

    # Register the typing for Pixel attributes with Numba.
    @cuda_registry.register_attr
    class WrapperAttrs(WrapperAttrsTemplate):
        key = this_type

    @register_model(WrapperType)
    class WrapperModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(f.name, numba.from_dtype(f.type)) for f in fields]
            super().__init__(dmm, fe_type, members)

    for f in fields:
        make_attribute_wrapper(WrapperType, f.name, f.name)

    @cuda_registry.register_global(operator.getitem)
    class WrapperGetitem(CallableTemplate):
        def generic(self):
            def typer(obj, index):
                if not isinstance(obj, WrapperType):
                    return None
                if not isinstance(index, types.StringLiteral):
                    return None
                retty = numba.from_dtype(dtype[index.literal_value])
                return signature(retty, obj, index)

            return typer

    @cuda_lower(operator.getitem, this_type, types.StringLiteral)
    def wrapper_getitem(context, builder, sig, args):
        obj_arg, index_arg = args
        obj = cgutils.create_struct_proxy(this_type)(context, builder, value=obj_arg)
        return getattr(obj, sig.args[1].literal_value)

    return this_type
