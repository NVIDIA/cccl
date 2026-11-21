# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Generate frontend declarations from the library-owned TileIR interface.

This deliberately accepts only the declaration grammar used by cuda.coop.tile.
New TileIR types or attributes need an explicit frontend binding decision.
Outputs are checked in; consumers do not parse TileIR or run this generator.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TILE_ROOT = REPO_ROOT / "tile" / "cpp"
INTERFACE = TILE_ROOT / "tilelib" / "interface" / "cuda_coop_tile.mlir"
HEADER = TILE_ROOT / "include" / "cuda" / "coop" / "__tile" / "foreign_declarations.cuh"
PYTHON_SCHEMA = (
    REPO_ROOT
    / "python"
    / "cuda_coop_tile"
    / "cuda"
    / "coop"
    / "tile"
    / "_declarations.py"
)
NATIVE_SCHEMA = TILE_ROOT / "tilelib" / "provider" / "declaration_schemas.h"

CONSTANT_TYPES = {
    "index": "size_t",
    "!cuda_tile.preview$Datatype": "typename",
    "!cuda_tile.preview$string": "cte::fixedString",
}
# These are the existing concrete C++ ABI bindings. Integer TileIR types are
# signless; symbolic integer signedness is preserved separately by DTypeToken.
CONCRETE_TYPES = {"i8": "unsigned char", "i32": "int"}


@dataclass(frozen=True)
class TileType:
    dtype: str
    extent: str = ""

    def cpp(self):
        dtype = CONCRETE_TYPES.get(self.dtype, self.dtype)
        return f"ct::tile<{dtype}, ct::shape<{self.extent}>>"


@dataclass(frozen=True)
class Operand:
    name: str
    type: TileType
    access: str


@dataclass(frozen=True)
class Declaration:
    symbol: str
    constants: tuple[tuple[str, str], ...]
    operands: tuple[Operand, ...]
    results: tuple[TileType, ...]


def split_types(source):
    """Split a type list at commas outside nested angle brackets.

    Tile element types may contain their own comma-separated arguments. Keep
    those arguments together and reject unbalanced brackets before parsing
    each resulting type.
    """

    parts = []
    start = depth = 0
    for index, character in enumerate(source):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(source[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError(f"unbalanced TileIR type: {source}")
    if depth:
        raise ValueError(f"unbalanced TileIR type: {source}")
    parts.append(source[start:].strip())
    return parts


def parse_type(source, constants):
    """Read a scalar or one-dimensional Tile type from a declaration.

    ``constants`` maps the declaration's template parameter names to TileIR
    kinds. Symbolic extents must name an index parameter; symbolic element
    types must name a datatype parameter. Reject other forms instead of
    guessing a Python or C++ representation.
    """

    compact = re.sub(r"\s+", "", source)
    match = re.fullmatch(
        r"!cuda_tile\.tile<"
        r"(?:(?:\[([A-Za-z][A-Za-z0-9]*)\]|([0-9]+))x)?"
        r"(?:!cuda_tile\.preview\$sym_t<([A-Za-z][A-Za-z0-9]*)>|(i8|i32))>",
        compact,
    )
    if not match:
        raise ValueError(f"unsupported TileIR type: {source}")
    extent_parameter, extent_literal, dtype_parameter, dtype_literal = match.groups()
    if extent_parameter and constants.get(extent_parameter) != "index":
        raise ValueError(f"unbound extent: {extent_parameter}")
    if (
        dtype_parameter
        and constants.get(dtype_parameter) != "!cuda_tile.preview$Datatype"
    ):
        raise ValueError(f"unbound dtype: {dtype_parameter}")
    return TileType(
        dtype_parameter or dtype_literal,
        extent_parameter or extent_literal or "",
    )


def parse_declarations(source):
    """Validate the interface and return declarations in source order.

    The returned records retain constant-parameter order and operand access
    modes, which determine the generated call ABI. Results must match the
    output and input/output operands in that same order. Duplicate symbols,
    unbound parameters, and unsupported syntax raise ``ValueError``.
    """

    source = re.sub(r"//[^\n]*", "", source).strip()
    module = re.fullmatch(
        r"cuda_tile\.module @cuda_coop_tile_interface\s*\{(.*)\}",
        source,
        re.DOTALL,
    )
    if not module:
        raise ValueError("expected the cuda_coop_tile_interface module")
    declarations = []
    symbols = set()
    for item in re.split(r"(?=cuda_tile\.preview\$func)", module[1]):
        if not item.strip():
            continue
        match = re.fullmatch(
            r"\s*cuda_tile\.preview\$func extern\s+@(cuda_coop_tile_[a-z0-9_]+)"
            r"<\s*\[([^]]+)\]>\s*\((.*?)\)\s*->\s*(.*?)\s*",
            item,
            re.DOTALL,
        )
        if not match:
            raise ValueError(f"unsupported TileIR declaration: {item.strip()}")
        symbol, constant_source, operand_source, result_source = match.groups()
        if symbol in symbols:
            raise ValueError(f"duplicate declaration: {symbol}")
        symbols.add(symbol)
        constants = {}
        for constant in constant_source.split(","):
            name, kind = (part.strip() for part in constant.split(":", maxsplit=1))
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name)
                or kind not in CONSTANT_TYPES
            ):
                raise ValueError(f"unsupported constant parameter: {constant}")
            if name in constants:
                raise ValueError(f"duplicate constant parameter: {name}")
            constants[name] = kind
        operands = []
        for operand in re.split(r",\s*(?=%)", operand_source):
            match = re.fullmatch(
                r"\s*%([a-z0-9_]+):\s*(.*?)\s*"
                r"\{cuda_tile\.preview\$access\s*=\s*"
                r"#cuda_tile\.preview\$access<(in|out|inout)>\}\s*",
                operand,
                re.DOTALL,
            )
            if not match:
                raise ValueError(f"unsupported operand: {operand}")
            name, type_source, access = match.groups()
            operands.append(Operand(name, parse_type(type_source, constants), access))
        if result_source.startswith("(") and result_source.endswith(")"):
            result_source = result_source[1:-1]
        results = tuple(
            parse_type(part, constants) for part in split_types(result_source)
        )
        if results != tuple(
            operand.type for operand in operands if operand.access != "in"
        ):
            raise ValueError(
                f"{symbol}: results must match out/inout operands in order"
            )
        declarations.append(
            Declaration(symbol, tuple(constants.items()), tuple(operands), results)
        )
    if not declarations:
        raise ValueError("interface has no foreign declarations")
    return declarations


def render_header(declarations):
    """Emit C++ foreign declarations with the interface's template order."""

    lines = [
        (
            "// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. "
            "ALL RIGHTS RESERVED."
        ),
        "//",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// Generated by tile/tools/generate_tile_declarations.py.",
        "// Edit tile/cpp/tilelib/interface/cuda_coop_tile.mlir instead.",
        "// clang-format off",
        "",
        "#ifndef CUDA_COOP_TILE_FOREIGN_DECLARATIONS_CUH",
        "#define CUDA_COOP_TILE_FOREIGN_DECLARATIONS_CUH",
        "",
        "#include <cstddef>",
        "#include <cuda_tile.h>",
        "",
        (
            "#if !defined(__CUDACC_TILE__) || "
            "!defined(__CUDACC_TILE_PREVIEW_SIMT__) || "
            "!defined(__experimental_foreign_decl__) || "
            "!defined(__experimental_foreign_name__)"
        ),
        (
            '#  error "cuda::coop::tile requires a CUDA C++ Tile frontend with '
            "experimental linkage-provider declarations; "
            'compile with --enable-tile --tile-preview=simt"'
        ),
        "#endif",
        "",
        "namespace cuda::coop::tile::detail",
        "{",
        "namespace ct = ::cuda::tiles;",
        "namespace cte = ::cuda::tiles::experimental;",
        "using ::std::size_t;",
    ]
    for declaration in declarations:
        parameters = [
            f"{CONSTANT_TYPES[kind]} {name}" for name, kind in declaration.constants
        ]
        lines.extend(["", "template <" + ",\n          ".join(parameters) + ">"])
        result_types = [result.cpp() for result in declaration.results]
        result = (
            result_types[0]
            if len(result_types) == 1
            else f"cte::multi_outputs<{', '.join(result_types)}>"
        )
        lines.extend(
            [
                "__experimental_foreign_decl__",
                f'__experimental_foreign_name__("{declaration.symbol}")',
                result,
            ]
        )
        arguments = []
        for operand in declaration.operands:
            if operand.access == "out":
                continue
            # The existing scalar i32 controls use the frontend's scalar ABI;
            # symbolic scalar controls retain a rank-zero Tile binding.
            type_name = "int" if operand.type == TileType("i32") else operand.type.cpp()
            attribute = (
                "[[cutile::experimental_inout]] " if operand.access == "inout" else ""
            )
            arguments.append(f"{attribute}{type_name} {operand.name}")
        name = declaration.symbol.removeprefix("cuda_coop_tile_") + "_foreign"
        lines.append(f"{name}(\n  " + ",\n  ".join(arguments) + ");")
    lines.extend(
        [
            "} // namespace cuda::coop::tile::detail",
            "",
            "#endif // CUDA_COOP_TILE_FOREIGN_DECLARATIONS_CUH",
            "",
        ]
    )
    return "\n".join(lines)


def render_python(declarations):
    """Emit immutable schemas used to validate Python foreign-call operands."""

    lines = [
        ("# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED."),
        "#",
        "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "# Generated by tile/tools/generate_tile_declarations.py.",
        "# Edit tile/cpp/tilelib/interface/cuda_coop_tile.mlir instead.",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class TileType:",
        "    dtype: str",
        "    shape: tuple[int | str, ...]",
        "",
        "",
        "@dataclass(frozen=True)",
        "class Declaration:",
        "    constants: tuple[tuple[str, str], ...]",
        "    operands: tuple[TileType, ...]",
        "    results: tuple[TileType, ...]",
        "    access: tuple[str, ...]",
        "",
        "",
        "DECLARATIONS = {",
    ]
    kinds = {
        "index": "Dim",
        "!cuda_tile.preview$Datatype": "DType",
        "!cuda_tile.preview$string": "String",
    }
    for declaration in declarations:
        lines.extend(
            [
                f"    {json.dumps(declaration.symbol)}: Declaration(",
                "        constants=(",
            ]
        )
        for name, kind in declaration.constants:
            lines.append(
                f"            ({json.dumps(name)}, {json.dumps(kinds[kind])}),"
            )
        lines.append("        ),")
        for field, types in (
            (
                "operands",
                tuple(operand.type for operand in declaration.operands),
            ),
            ("results", declaration.results),
        ):
            rendered_types = []
            for tile_type in types:
                extent = tile_type.extent
                shape = (
                    "()"
                    if not extent
                    else (f"({extent if extent.isdigit() else json.dumps(extent)},)")
                )
                rendered_types.append(
                    f"TileType({json.dumps(tile_type.dtype)}, {shape})"
                )
            if len(rendered_types) == 1:
                lines.append(f"        {field}=({rendered_types[0]},),")
            else:
                lines.append(f"        {field}=(")
                lines.extend(
                    f"            {tile_type}," for tile_type in rendered_types
                )
                lines.append("        ),")
        access = ", ".join(
            json.dumps(operand.access) for operand in declaration.operands
        )
        if len(declaration.operands) == 1:
            access += ","
        lines.extend([f"        access=({access}),", "    ),"])
    lines.extend(["}", ""])
    return "\n".join(lines)


def render_native_schema(declarations):
    """Emit provider schemas with positional indices into constant arguments.

    The native provider uses these indices to recover signed dtype metadata
    from a compiler request. Keep operand access modes and result counts tied
    to the same validated interface used by both frontends.
    """

    lines = [
        (
            "// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. "
            "ALL RIGHTS RESERVED."
        ),
        "//",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// Generated by tile/tools/generate_tile_declarations.py.",
        "// Edit tile/cpp/tilelib/interface/cuda_coop_tile.mlir instead.",
        "// clang-format off",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "#include <tile_ir/linkage_provider/LinkageProviderAPI.h>",
        "",
        "namespace cuda::coop::tile::detail::provider::generated",
        "{",
        "struct ConstantParameterSchema",
        "{",
        "  const char* name;",
        "  TileIRConstantParameterKind kind;",
        "};",
        "",
        "struct DeclarationSchema",
        "{",
        "  const char* symbol;",
        "  const ConstantParameterSchema* constants;",
        "  std::size_t num_constants;",
        "  const TileIRLPAccessKind* access;",
        "  std::size_t num_operands;",
        "  std::size_t num_results;",
        "  // -1 denotes a concrete dtype; other values index constants.",
        "  const std::int8_t* dtype_parameter_indices;",
        "};",
    ]
    kinds = {
        "index": "TILEIR_CONSTANT_PARAMETER_INT",
        "!cuda_tile.preview$Datatype": "TILEIR_CONSTANT_PARAMETER_DTYPE",
        "!cuda_tile.preview$string": "TILEIR_CONSTANT_PARAMETER_STRING",
    }
    records = []
    for declaration in declarations:
        name = declaration.symbol.removeprefix("cuda_coop_tile_")
        indices = {name: index for index, (name, _) in enumerate(declaration.constants)}
        if len(declaration.constants) > 128:
            raise ValueError(f"{declaration.symbol}: constant indices exceed int8_t")
        lines.extend(
            [
                "",
                (f"inline constexpr ConstantParameterSchema {name}_constants[] = {{"),
            ]
        )
        for constant, kind in declaration.constants:
            lines.append(f"  {{{json.dumps(constant)}, {kinds[kind]}}},")
        lines.extend(
            ["};", f"inline constexpr TileIRLPAccessKind {name}_access[] = {{"]
        )
        lines.extend(
            f"  TILEIR_LP_ACCESS_{operand.access.upper()},"
            for operand in declaration.operands
        )
        dtype_indices = [
            str(indices.get(operand.type.dtype, -1)) for operand in declaration.operands
        ]
        lines.extend(
            [
                "};",
                (
                    "inline constexpr std::int8_t "
                    f"{name}_dtype_parameter_indices[] "
                    f"= {{{', '.join(dtype_indices)}}};"
                ),
            ]
        )
        records.extend(
            [
                "  {",
                f"    {json.dumps(declaration.symbol)},",
                f"    {name}_constants, {len(declaration.constants)},",
                (
                    f"    {name}_access, {len(declaration.operands)}, "
                    f"{len(declaration.results)},"
                ),
                f"    {name}_dtype_parameter_indices",
                "  },",
            ]
        )
    lines.extend(["", "inline constexpr DeclarationSchema declarations[] = {"])
    lines.extend(records)
    lines.extend(
        [
            "};",
            "",
            (
                "inline constexpr const DeclarationSchema* "
                "schema_for(std::string_view symbol) noexcept"
            ),
            "{",
            "  for (const auto& declaration : declarations)",
            "  {",
            "    if (symbol == declaration.symbol)",
            "    {",
            "      return &declaration;",
            "    }",
            "  }",
            "  return nullptr;",
            "}",
            "} // namespace cuda::coop::tile::detail::provider::generated",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, default=INTERFACE)
    parser.add_argument("--output", type=Path, help="write only the C++ header here")
    parser.add_argument(
        "--python-output", type=Path, help="write the Python schema here"
    )
    parser.add_argument(
        "--native-output",
        type=Path,
        help="write the native provider schema here",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the selected generated outputs are stale",
    )
    args = parser.parse_args()
    try:
        declarations = parse_declarations(args.interface.read_text(encoding="utf-8"))
        outputs = {}
        if (
            args.output is None
            and args.python_output is None
            and args.native_output is None
        ):
            outputs[HEADER] = render_header(declarations)
            outputs[PYTHON_SCHEMA] = render_python(declarations)
            outputs[NATIVE_SCHEMA] = render_native_schema(declarations)
        else:
            if args.output is not None:
                outputs[args.output] = render_header(declarations)
            if args.python_output is not None:
                outputs[args.python_output] = render_python(declarations)
            if args.native_output is not None:
                outputs[args.native_output] = render_native_schema(declarations)
    except ValueError as error:
        parser.error(str(error))
    for path, rendered in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                parser.exit(1, f"{path} is stale; run {Path(__file__).name}\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
