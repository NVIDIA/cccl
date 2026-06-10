# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from io import StringIO
from textwrap import dedent
from types import FunctionType as PyFunctionType
from typing import BinaryIO, Literal, Sequence

import numba
from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import intrinsic, overload
from numba.core.typing import signature

try:
    from cuda.core import Linker, LinkerOptions, ObjectCode
except ImportError:
    from cuda.core.experimental import Linker, LinkerOptions, ObjectCode

from . import _nvrtc as nvrtc
from ._common import find_unsigned

NUMBA_TYPES_TO_CPP = {
    types.boolean: "bool",
    types.int8: "::cuda::std::int8_t",
    types.int16: "::cuda::std::int16_t",
    types.int32: "::cuda::std::int32_t",
    types.int64: "::cuda::std::int64_t",
    types.uint8: "::cuda::std::uint8_t",
    types.uint16: "::cuda::std::uint16_t",
    types.uint32: "::cuda::std::uint32_t",
    types.uint64: "::cuda::std::uint64_t",
    types.float16: "__half",
    types.float32: "float",
    types.float64: "double",
}


def numba_type_to_cpp(numba_type):
    return NUMBA_TYPES_TO_CPP.get(numba_type, "storage_t")


def method_to_signature(numba_type, method):
    ptr_type = types.CPointer(numba_type)
    if method == "construct":
        return signature(types.void, ptr_type)
    elif method == "assign":
        return signature(types.void, ptr_type, ptr_type)
    else:
        raise ValueError("Unexpected method {}".format(method))


class TypeWrapper:
    def __init__(self, numba_type, methods):
        self.lto_irs = []

        if numba_type in NUMBA_TYPES_TO_CPP:
            self.code = ""
            return

        context = cuda.descriptor.cuda_target.target_context
        value_type = context.get_value_type(numba_type)
        size = value_type.get_abi_size(context.target_data)
        alignment = value_type.get_abi_alignment(context.target_data)

        buf = StringIO()
        w = buf.write
        if "construct" in methods:
            construct_name = methods["construct"].__name__
            w(f'extern "C" __device__ void {construct_name}(void *ptr);\n')
        if "assign" in methods:
            assign_name = methods["assign"].__name__
            w(
                f'extern "C" __device__ void {assign_name}'
                "(void *dst, const void *src);\n"
            )

        w(f"struct __align__({alignment}) storage_t\n")
        w("{\n")
        if "construct" in methods:
            w("    __device__ storage_t() {\n")
            w(f"        {construct_name}(data);\n")
            w("    }\n")
        if "assign" in methods:
            w("    __device__ storage_t& operator=(const storage_t &rhs) {\n")
            w(f"        {assign_name}(data, rhs.data);\n")
            w("        return *this;\n")
            w("    }\n")
        w(f"    char data[{size}];\n")
        w("};\n")

        self.code = buf.getvalue()

        for method in methods:
            lto_fn, _ = cuda.compile(
                methods[method],
                sig=method_to_signature(numba_type, method),
                output="ltoir",
            )
            self.lto_irs.append(lto_fn)


def numba_type_to_wrapper(
    numba_type: numba.types.Type, methods: Literal["construct", "assign"] = None
):
    if methods is None:
        methods = {}
    for method in methods:
        if method not in ["construct", "assign"]:
            raise ValueError("Unexpected method {}".format(method))
    return TypeWrapper(numba_type, methods)


class Parameter:
    def __init__(self, is_output=False):
        self.is_output = is_output

    def __repr__(self) -> str:
        return f"Parameter(out={self.is_output})"

    def specialize(self, _):
        return self

    def is_provided_by_user(self):
        return not self.is_output


class Value(Parameter):
    def __init__(self, value_type, is_output=False):
        self.value_type = value_type
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Value(dtype={self.value_type}, out={self.is_output})"

    def dtype(self):
        return self.value_type

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_type) + " " + name

    def mangled_name(self):
        return f"{self.value_type}"


class Pointer(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Pointer(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "* " + name

    def dtype(self):
        return numba.types.Array(self.value_dtype, 1, "A")

    def mangled_name(self):
        return f"P{self.value_dtype}"


class DependentPointer(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentPointer(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Pointer(self.value_dtype.resolve(template_arguments), self.is_output)


class Reference(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Reference(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "& " + name

    def dtype(self):
        return self.value_dtype

    def mangled_name(self):
        return f"R{self.value_dtype}"


class DependentReference(Parameter):
    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentReference(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Reference(self.value_dtype.resolve(template_arguments), self.is_output)


class Array(Pointer):
    def __init__(self, value_dtype, size, is_output=False):
        self.size = size
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"Array(dtype={self.value_dtype}, size={self.size}, out={self.is_output})"
        )

    def cpp_decl(self, name):
        return f"{numba_type_to_cpp(self.value_dtype)} (&{name})[{self.size}]"

    def dtype(self):
        return numba.types.Array(self.value_dtype, 1, "C")

    def mangled_name(self):
        return f"P{self.value_dtype}"


class SubstitutionFailure(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Dependency:
    def __init__(self, dep):
        self.dep = dep

    def resolve(self, template_arguments):
        if self.dep not in template_arguments:
            raise SubstitutionFailure(f"Template argument {self.dep} not provided")
        if template_arguments[self.dep] is None:
            raise SubstitutionFailure(f"Template argument {self.dep} is None")
        return template_arguments[self.dep]


class Constant:
    def __init__(self, val):
        self.val = val

    def resolve(self, _):
        return self.val


class StatefulFunction:
    def __init__(self, op, dtype):
        self.op = op
        self.dtype = dtype


class StatefulOperator:
    def __init__(self, mangled_name, op_type, ret_cpp_type, arg_cpp_types, ltoir):
        self.op_type = op_type
        self.name = mangled_name
        self.ltoir = ltoir
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = arg_cpp_types
        self.is_output = False

    def mangled_name(self):
        return f"{self.name}"

    def forward_decl(self):
        arg_decls = ["char *state"]
        if self.ret_cpp_type == "storage_t":
            ret_decl = "void"
            arg_decls.append("void*")
        else:
            ret_decl = self.ret_cpp_type
        for arg in self.arg_cpp_types:
            arg_decls.append("const void*" if arg == "storage_t" else arg)
        return f'extern "C" __device__ {ret_decl} {self.mangled_name()}({", ".join(arg_decls)});'

    def cpp_decl(self, name):
        return f"char* {name}_state"

    def dtype(self):
        return numba.types.Array(self.op_type, 1, "C")

    def wrap_decl(self, name):
        param_decls = []
        param_refs = []
        for aid, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{aid}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append("&" + arg_name if arg_type == "storage_t" else arg_name)

        param_decls_csv = ", ".join(param_decls)
        param_refs_csv = ", ".join(param_refs)

        buf = StringIO()
        w = buf.write
        state_name = f"{name}_state"
        w(f"auto {name} = [{state_name}]({param_decls_csv}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w(f"    {self.ret_cpp_type} result;\n")
            w(f"    {self.mangled_name()}(\n")
            w(f"        {state_name}, &result, {param_refs_csv});\n")
            w("    return result;\n")
        else:
            w(f"    return {self.mangled_name()}(\n")
            w(f"        {state_name}, {param_refs_csv});\n")
        w("};\n")
        src = buf.getvalue()
        return src

    def is_provided_by_user(self):
        return True


class StatelessOperator:
    def __init__(self, mangled_name, ret_cpp_type, arg_cpp_types, ltoir):
        self.name = mangled_name
        self.ltoir = ltoir
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = arg_cpp_types

    def mangled_name(self):
        return f"{self.name}"

    def forward_decl(self):
        arg_decls = []
        if self.ret_cpp_type == "storage_t":
            ret_decl = "void"
            arg_decls.append("void*")
        else:
            ret_decl = self.ret_cpp_type
        for arg in self.arg_cpp_types:
            arg_decls.append("const void*" if arg == "storage_t" else arg)
        mangled_name = self.mangled_name()
        arg_decls_csv = ", ".join(arg_decls)
        return f'extern "C" __device__ {ret_decl} {mangled_name}({arg_decls_csv});'

    def wrap_decl(self, name):
        param_decls = []
        param_refs = []
        for aid, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{aid}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append("&" + arg_name if arg_type == "storage_t" else arg_name)

        buf = StringIO()
        w = buf.write
        param_decls_csv = ", ".join(param_decls)
        param_refs_csv = ", ".join(param_refs)
        mangled_name = self.mangled_name()

        w(f"auto {name} = []({param_decls_csv}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w("    storage_t result;\n")
            w(f"    {mangled_name}(&result, {param_refs_csv});\n")
            w("    return result;\n")
        else:
            w(f"    return {mangled_name}({param_refs_csv});\n")
        w("};\n")
        src = buf.getvalue()
        return src

    def is_provided_by_user(self):
        return False


class DependentPythonOperator:
    def __init__(self, ret_dtype, arg_dtypes, op):
        self.ret_dtype = ret_dtype
        self.arg_dtypes = arg_dtypes
        self.op = op

    def specialize(self, template_arguments):
        op = self.op.resolve(template_arguments)
        ret_dtype = self.ret_dtype.resolve(template_arguments)
        ret_cpp_type = numba_type_to_cpp(ret_dtype)
        ret_numba_type = (
            types.CPointer(ret_dtype) if ret_cpp_type == "storage_t" else ret_dtype
        )
        arg_cpp_types = []
        arg_dtypes = []
        arg_numba_types = []
        for arg in self.arg_dtypes:
            arg_dtype = arg.resolve(template_arguments)
            arg_cpp_type = numba_type_to_cpp(arg_dtype)
            arg_cpp_types.append(arg_cpp_type)
            arg_dtypes.append(str(arg_dtype))
            arg_numba_types.append(
                types.CPointer(arg_dtype) if arg_cpp_type == "storage_t" else arg_dtype
            )

        if isinstance(op, StatefulFunction):
            binary_op = op.op.__call__
            mangled_name = f"F{binary_op.__name__}_{ret_dtype}__" + "_".join(arg_dtypes)
            if ret_cpp_type == "storage_t":
                binary_op_signature = signature(
                    types.void,
                    types.CPointer(op.dtype),
                    *arg_numba_types,
                    ret_numba_type,
                )
            else:
                binary_op_signature = signature(
                    ret_numba_type, types.CPointer(op.dtype), *arg_numba_types
                )
            abi_info = {"abi_name": mangled_name}
            ltoir, _ = cuda.compile(
                binary_op, sig=binary_op_signature, output="ltoir", abi_info=abi_info
            )
            return StatefulOperator(
                mangled_name, op.dtype, ret_cpp_type, arg_cpp_types, ltoir
            )
        else:
            binary_op = op
            mangled_name = f"F{binary_op.__name__}_{ret_dtype}__" + "_".join(arg_dtypes)
            if ret_cpp_type == "storage_t":
                binary_op_signature = signature(
                    types.void, *arg_numba_types, ret_numba_type
                )
            else:
                binary_op_signature = signature(ret_numba_type, *arg_numba_types)
            abi_info = {"abi_name": mangled_name}
            ltoir, _ = cuda.compile(
                binary_op, sig=binary_op_signature, output="ltoir", abi_info=abi_info
            )
            return StatelessOperator(mangled_name, ret_cpp_type, arg_cpp_types, ltoir)


class CxxFunction(Parameter):
    def __init__(self, cpp, func_dtype):
        super().__init__()
        self.cpp = cpp
        self.func_dtype = func_dtype

    def __repr__(self) -> str:
        return f"CxxFunction(cpp={self.cpp})"

    def mangled_name(self):
        return f"F{internal_mangle_cpp(self.cpp)}"

    def dtype(self):
        return self.func_dtype

    def is_provided_by_user(self):
        return False


class DependentCxxOperator:
    def __init__(self, dep: Dependency, cpp: str):
        self.dep = dep
        self.cpp = cpp

    def specialize(self, template_arguments):
        dtype = self.dep.resolve(template_arguments)
        dtype_cpp = numba_type_to_cpp(dtype)
        source = f"<{self.dep.dep}>"
        target = f"<{dtype_cpp}>"
        cpp = self.cpp.replace(source, target)
        return CxxFunction(cpp=f"{cpp}{{}}", func_dtype=dtype)


class DependentArray(Parameter):
    def __init__(self, value_dtype, size, is_output=False):
        self.value_dtype = value_dtype
        self.size = size
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentArray(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Array(
            self.value_dtype.resolve(template_arguments),
            self.size.resolve(template_arguments),
            self.is_output,
        )


class TemplateParameter:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}"


def internal_mangle_cpp(cpp_name: str):
    """
    Substitutes non-alphanumeric characters in a C++ name with underscores,
    such that they can be used as valid, unique identifiers in C code.  This
    is for internal use only, and does not comport with C++ ABI name mangling.

    :param cpp_name: Supplies a C++ name to be mangled.
    :type cpp_name: str

    :return: Returns the mangled C++ name with non-alphanumeric characters
    substituted with underscores.
    :rtype: str

    Example
    -------

    .. code-block:: python

        >>> mangle("std::vector<int>")
        'std_vector_int_'
        >>> mangle("::cuda::std::min<::cuda::std::uint32_t>{}")
        '__cuda__std__min__cuda__std__uint32_t__'
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", cpp_name)


def mangle_symbol(name, template_parameters):
    return "_".join(
        [name]
        + [
            template_parameter.mangled_name()
            for template_parameter in template_parameters
        ]
    )


def war_introspection(fn, n):
    arglist = ", ".join(f"param{i}" for i in range(n))
    mod_str = dedent(f"""
    def impl({arglist}):
        return fn({arglist})
    """)
    mod_code = compile(mod_str, "<string>", "exec")
    func_code = mod_code.co_consts[0]
    return PyFunctionType(func_code, locals())


class Algorithm:
    def __init__(
        self,
        struct_name,
        method_name,
        c_name,
        includes,
        template_parameters,
        parameters,
        type_definitions=None,
        fake_return=False,
    ):
        self.struct_name = struct_name
        self.method_name = method_name
        self.c_name = c_name
        self.includes = includes
        self.template_parameters = template_parameters
        self.parameters = parameters
        self.type_definitions = type_definitions
        self.fake_return = fake_return
        self._temp_storage_bytes = None
        self._temp_storage_alignment = None

    def __repr__(self) -> str:
        return f"{self.struct_name}::{self.method_name}{self.template_parameters}: {self.parameters}"

    def mangled_name(self, parameters):
        return mangle_symbol(self.c_name, parameters)

    def specialize(self, template_arguments):
        # No partial specializations for now
        template_list = []
        for template_parameter in self.template_parameters:
            if template_parameter.name not in template_arguments:
                raise ValueError(
                    f"Template argument {template_parameter.name} not provided"
                )
            template_argument = template_arguments[template_parameter.name]
            if isinstance(template_argument, int):
                template_list.append(str(template_argument))
            elif isinstance(template_argument, str):
                template_list.append(template_argument)
            else:
                template_list.append(numba_type_to_cpp(template_argument))
        template_list = ", ".join(template_list)

        # '::cuda::std::int32_t, 32' -> __cuda__std__int32_t__32
        mangle = internal_mangle_cpp(template_list)

        specialized_parameters = []
        for method in self.parameters:
            specialized_signature = []
            try:
                for parameter in method:
                    specialized_signature.append(
                        parameter.specialize(template_arguments)
                    )
                specialized_parameters.append(specialized_signature)
            except SubstitutionFailure:
                pass  # Substitution failure is not an error

        specialized_name = f"{self.struct_name}<{template_list}>"
        return Algorithm(
            specialized_name,
            self.method_name,
            self.c_name + mangle,
            self.includes,
            [],
            specialized_parameters,
            type_definitions=self.type_definitions,
            fake_return=self.fake_return,
        )

    @property
    def temp_storage_bytes(self):
        if self._temp_storage_bytes is None:
            raise RuntimeError(
                "Temporary storage bytes not computed yet.  Call get_lto_ir() first."
            )
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        if self._temp_storage_alignment is None:
            raise RuntimeError(
                "Temporary storage alignment not computed yet.  "
                "Call get_lto_ir() first."
            )
        return self._temp_storage_alignment

    def get_lto_ir(self, threads=None):
        lto_irs = []

        if self.type_definitions:
            for type_definition in self.type_definitions:
                lto_irs.extend(type_definition.lto_irs)

        udf_declarations = {}

        for method in self.parameters:
            for param in method:
                if isinstance(param, StatelessOperator) or isinstance(
                    param, StatefulOperator
                ):
                    if param.name not in udf_declarations:
                        udf_declarations[param.name] = param.forward_decl()
                        lto_irs.append(param.ltoir)

        algorithm_name = self.struct_name
        includes = self.includes or []
        type_definitions = self.type_definitions or []

        buf = StringIO()
        w = buf.write

        w("#include <cuda/std/cstdint>\n")
        for include in includes:
            w(f"#include <{include}>\n")
        for type_definition in type_definitions:
            w(f"{type_definition.code}\n")

        w("\n")
        for decl in udf_declarations.values():
            w(f"{decl}\n")
        w("\n")

        w(f"using algorithm_t = cub::{algorithm_name};\n")
        w("using temp_storage_t = typename algorithm_t::TempStorage;\n")
        prefix = "__device__ constexpr unsigned temp_storage_"
        w(f"{prefix}bytes = sizeof(temp_storage_t);\n")
        w(f"{prefix}alignment = alignof(temp_storage_t);\n")

        src = buf.getvalue()

        method_name = self.method_name

        for method in self.parameters:
            param_decls = []
            func_decls = []
            param_args = []
            out_param = None

            for pid, param in enumerate(method[1:]):
                if isinstance(param, StatelessOperator):
                    func_decls.append(param.wrap_decl(f"param_{pid}"))
                    param_args.append(f"param_{pid}")
                elif isinstance(param, StatefulOperator):
                    name = f"param_{pid}"
                    func_decls.append(param.wrap_decl(name))
                    param_args.append(name)
                    param_decls.append(param.cpp_decl(name))
                elif isinstance(param, CxxFunction):
                    param_args.append(param.cpp)
                else:
                    name = f"param_{pid}"
                    param_decls.append(param.cpp_decl(name))
                    if not self.fake_return and param.is_output:
                        if out_param is not None:
                            raise ValueError("Multiple output parameters not supported")
                        out_param = name
                    else:
                        param_args.append(name)

            if self.struct_name.startswith("Warp"):
                if threads is None:
                    raise ValueError("Warp algorithm must specify number of threads")
                # sub hw warps require computing masks for syncwarp, which is not supported
                # allocate temporary storage explicitly
                provide_alloc_version = threads == 32

                # pessimistic temporary storage allocation for 1024 threads
                storage = (
                    "__shared__ temp_storage_t temp_storages"
                    f"[1024 / {threads}];\n"
                    "    temp_storage_t &temp_storage = temp_storages"
                    f"[threadIdx.x / {threads}];\n"
                )
                sync = "__syncwarp();"
            elif self.struct_name.startswith("Block"):
                provide_alloc_version = True
                storage = "__shared__ temp_storage_t temp_storage;"
                sync = "__syncthreads();"

            buf = StringIO()
            w = buf.write

            mangled_name = self.mangled_name(method)
            param_decls_csv = ", ".join(param_decls)
            param_args_csv = ", ".join(param_args)

            if provide_alloc_version:
                w(f'extern "C" __device__ void {mangled_name}_alloc(')
                w(f"{param_decls_csv}) {{\n")
                w(f"    {storage}\n")
                for decl in func_decls:
                    w(f"    {decl}\n")
                if out_param:
                    w(f"    {out_param} = ")
                else:
                    w("    ")
                w("algorithm_t(temp_storage).")
                w(f"{method_name}({param_args_csv});\n")
                w(f"    {sync}\n")
                w("}\n")

            w(f'extern "C" __device__ void {mangled_name}(')
            w("temp_storage_t *temp_storage, ")
            w(f"{param_decls_csv}) {{\n")
            for decl in func_decls:
                w(f"    {decl}\n")
            if out_param:
                w(f"    {out_param} = ")
            else:
                w("    ")
            w("algorithm_t(*temp_storage).")
            w(f"{method_name}({param_args_csv});\n")
            w("}\n\n")

            chunk = buf.getvalue()
            src += chunk

        device = cuda.get_current_device()
        cc_major, cc_minor = device.compute_capability
        cc = cc_major * 10 + cc_minor
        # N.B. Uncomment this to immediately print generated source to stdout.
        # print(src)
        _, blob = nvrtc.compile(cpp=src, cc=cc, rdc=True, code="lto")
        lto_irs.append(blob)

        # Convert the LTO into PTX in order to extract the size and alignment
        # variables.
        ltoir_obj = ObjectCode.from_ltoir(blob, name=self.c_name)
        linker_options = LinkerOptions(
            arch=f"sm_{cc}",
            link_time_optimization=True,
            ptx=True,
        )
        linker = Linker(ltoir_obj, options=linker_options)
        linked_ptx = linker.link("ptx")
        ptx = linked_ptx.code.decode("utf-8")
        self._temp_storage_bytes = find_unsigned("temp_storage_bytes", ptx)
        self._temp_storage_alignment = find_unsigned("temp_storage_alignment", ptx)

        return lto_irs

    def codegen(self, func_to_overload):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        for method in self.parameters:
            self.codegen_method(func_to_overload, method, self.mangled_name(method))
            self.codegen_method(
                func_to_overload, method[1:], self.mangled_name(method) + "_alloc"
            )

    def codegen_method(self, func_to_overload, method, mangled_name):
        if len(self.template_parameters):
            raise ValueError("Cannot generate codegen for a template")

        def ignore_param(param):
            # Stateless operators and C++ functions do not require any
            # additional argument handling or code generation, so we can
            # safely ignore them during this code gen phase.
            ignore = isinstance(param, StatelessOperator) or isinstance(
                param, CxxFunction
            )
            return ignore

        def intrinsic_impl(*args):
            def codegen(context, builder, sig, args):
                types = []
                arguments = []
                ret = None
                arg_id = 0
                for param in method:
                    if ignore_param(param):
                        continue

                    dtype = param.dtype()
                    if isinstance(param, StatefulOperator):
                        arg = args[arg_id]
                        state_ptr = cgutils.create_struct_proxy(dtype)(
                            context, builder, arg
                        ).data
                        void_ptr = builder.bitcast(
                            state_ptr, ir.PointerType(ir.IntType(8))
                        )
                        types.append(ir.PointerType(ir.IntType(8)))
                        arguments.append(void_ptr)
                    elif isinstance(param, Reference):
                        if param.is_output:
                            ptr = cgutils.alloca_once(
                                builder, context.get_value_type(dtype)
                            )
                            void_ptr = builder.bitcast(
                                ptr, ir.PointerType(ir.IntType(8))
                            )
                            types.append(ir.PointerType(ir.IntType(8)))
                            arguments.append(void_ptr)
                            ret = ptr
                        else:
                            arg = args[arg_id]
                            ptr = cgutils.alloca_once_value(builder, arg)
                            data_type = context.get_value_type(dtype)
                            void_ptr = builder.bitcast(
                                ptr, ir.PointerType(ir.IntType(8))
                            )
                            types.append(ir.PointerType(ir.IntType(8)))
                            arguments.append(void_ptr)
                    elif isinstance(param, Array) or isinstance(param, Pointer):
                        if param.is_output:
                            raise ValueError("Output arrays not supported")
                        arg = args[arg_id]
                        data_type = context.get_value_type(dtype.dtype)
                        types.append(ir.PointerType(data_type))
                        arguments.append(
                            cgutils.create_struct_proxy(dtype)(
                                context, builder, arg
                            ).data
                        )
                    else:
                        if param.is_output:
                            raise ValueError("Output values not supported")
                        arg = args[arg_id]
                        data_type = context.get_value_type(dtype)
                        types.append(data_type)
                        arguments.append(arg)

                    if not param.is_output:
                        arg_id += 1

                function_type = ir.FunctionType(ir.VoidType(), types)
                function = cgutils.get_or_insert_function(
                    builder.module, function_type, mangled_name
                )
                builder.call(function, arguments)

                if ret is not None:
                    return builder.load(ret)

            params = []
            ret = numba.types.void
            for param in method:
                if ignore_param(param):
                    continue

                if param.is_output:
                    if ret is not numba.types.void:
                        raise ValueError("Multiple output parameters not supported")
                    ret = param.dtype()
                else:
                    params.append(param.dtype())

            return signature(ret, *params), codegen

        num_user_provided_params = sum(
            [param.is_provided_by_user() for param in method]
        )
        numba_intrinsic = intrinsic(
            war_introspection(intrinsic_impl, 1 + num_user_provided_params)
        )

        def algorithm_impl(*args):
            return war_introspection(numba_intrinsic, len(args))

        wrapped_algorithm_impl = war_introspection(
            algorithm_impl, num_user_provided_params
        )
        overload(func_to_overload, target="cuda")(wrapped_algorithm_impl)


class Invocable:
    def __init__(
        self,
        temp_files: Sequence[BinaryIO],
        temp_storage_bytes: int,
        temp_storage_alignment: int,
        algorithm: Algorithm,
    ):
        self._temp_files = temp_files
        self._temp_storage_bytes = temp_storage_bytes
        self._temp_storage_alignment = temp_storage_alignment
        algorithm.codegen(self)

    @property
    def temp_storage_bytes(self):
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self._temp_storage_alignment

    @property
    def files(self):
        return [v.name for v in self._temp_files]

    def __call__(self, *args):
        raise Exception(
            "__call__ should not be called directly outside of a numba.cuda.jit(...) kernel."
        )
