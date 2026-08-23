# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static typing contracts for the public cuda.coop package surfaces."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from ...support.cases.api_contracts import (
    PORTABLE_GROUP_FIRST_EXPORTS,
)
from ...support.cases.api_contracts import (
    PORTABLE_GROUP_PRIMITIVE_DEFAULTS as _PORTABLE_PRIMITIVE_DEFAULTS,
)
from ...support.cases.api_contracts import (
    PORTABLE_GROUP_PRIMITIVE_KEYWORDS as _PORTABLE_PRIMITIVE_KEYWORDS,
)
from ...support.cases.api_contracts import (
    PORTABLE_GROUP_PRIMITIVE_POSITIONALS as _PORTABLE_PRIMITIVE_POSITIONALS,
)
from ...support.cases.api_contracts import (
    QUALIFIED_GROUP_PRIMITIVE_SUFFIXES as _BACKEND_SUFFIXES,
)
from ...support.cases.api_contracts import (
    REQUIRED_PARAMETER as _REQUIRED,
)
from ...support.cases.api_contracts import (
    qualified_group_primitive_suffix_contract as _suffix_contract,
)
from ...support.paths import PACKAGE_ROOT, TESTS_ROOT

_COOP_ROOT = PACKAGE_ROOT / "cuda" / "coop"
_TYPING_ARTIFACTS = (
    Path("py.typed"),
    Path("__init__.pyi"),
    Path("_typing.pyi"),
    Path("cutlass/py.typed"),
    Path("cutlass/__init__.pyi"),
    Path("cutlass/_group_adjacent_difference.pyi"),
    Path("cutlass/_group_discontinuity.pyi"),
    Path("cutlass/_group_exchange.pyi"),
    Path("cutlass/_group_histogram.pyi"),
    Path("cutlass/_group_load_store.pyi"),
    Path("cutlass/_group_merge_sort.pyi"),
    Path("cutlass/_group_radix.pyi"),
    Path("cutlass/_group_reduce.pyi"),
    Path("cutlass/_group_run_length_decode.pyi"),
    Path("cutlass/_group_scan.pyi"),
    Path("cutlass/_group_shuffle.pyi"),
    Path("cutlass/_group_topk.pyi"),
    Path("cutlass/_types.pyi"),
    Path("cutlass/aot.pyi"),
    Path("cutlass/_block/__init__.pyi"),
    Path("cutlass/_warp/__init__.pyi"),
    Path("numba_mlir/py.typed"),
    Path("numba_mlir/__init__.pyi"),
    Path("numba_mlir/_block/__init__.pyi"),
    Path("numba_mlir/_warp/__init__.pyi"),
)
_CUTLASS_PRIMITIVE_STUB_EXPORTS = {
    Path("_group_adjacent_difference.pyi"): ("adjacent_difference",),
    Path("_group_discontinuity.pyi"): ("discontinuity",),
    Path("_group_exchange.pyi"): ("exchange",),
    Path("_group_histogram.pyi"): ("histogram",),
    Path("_group_load_store.pyi"): ("load", "store"),
    Path("_group_merge_sort.pyi"): ("merge_sort_keys", "merge_sort_pairs"),
    Path("_group_radix.pyi"): (
        "radix_rank",
        "radix_sort_keys",
        "radix_sort_pairs",
    ),
    Path("_group_reduce.pyi"): ("reduce", "sum"),
    Path("_group_run_length_decode.pyi"): ("run_length_decode",),
    Path("_group_scan.pyi"): (
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "scan",
    ),
    Path("_group_shuffle.pyi"): ("shuffle",),
    Path("_group_topk.pyi"): (
        "topk_max_keys",
        "topk_max_pairs",
        "topk_min_keys",
        "topk_min_pairs",
    ),
}
_STUB_TO_RUNTIME_EXPORTS = {
    Path("cutlass/__init__.pyi"): Path("cutlass/__init__.py"),
    Path("cutlass/aot.pyi"): Path("cutlass/aot.py"),
    Path("cutlass/_block/__init__.pyi"): Path("cutlass/_dsl/block/__init__.py"),
    Path("cutlass/_warp/__init__.pyi"): Path("cutlass/_dsl/warp/__init__.py"),
    Path("numba_mlir/__init__.pyi"): Path("numba_mlir/__init__.py"),
    Path("numba_mlir/_block/__init__.pyi"): Path("numba_mlir/_block/__init__.py"),
    Path("numba_mlir/_warp/__init__.pyi"): Path("numba_mlir/_warp/__init__.py"),
}
_CUTLASS_THREAD_DATA_HELPERS = {
    "from_values",
    "from_fn",
    "from_register_tensor",
    "from_vector",
    "from_payload",
    "load",
    "to_tensor_ssa",
    "to_register_tensor",
}
_COMMON_THREAD_GROUP_METHODS = {
    "rank",
    "count",
    "rank_as",
    "count_as",
    "sync",
    "sync_aligned",
    "group_by",
    "is_member",
}
_PRIMS_LOAD_CONTROLS = {
    "alignment",
    "bounds_check",
    "is_volatile",
    "is_nontemporal",
    "is_invariant",
    "is_invariant_group",
    "ordering",
    "syncscope",
    "loc",
    "ip",
}
_PRIMS_STORE_CONTROLS = _PRIMS_LOAD_CONTROLS - {
    "is_invariant",
    "is_invariant_group",
}
_CUTLASS_LAUNCH_METADATA_CONTROLS = {
    "launch_metadata",
    "launch_meta",
    "launch",
    "launch_config",
}


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _relative_import_path(
    path: Path,
    statement: ast.ImportFrom,
    *,
    imported_name: str | None = None,
) -> Path | None:
    if statement.level == 0:
        return None

    module_path = path.parent
    for _ in range(statement.level - 1):
        module_path = module_path.parent
    if statement.module is None:
        if imported_name is None:
            return None
        module_path /= imported_name
    else:
        module_path = module_path.joinpath(*statement.module.split("."))
    suffixes = (".pyi", ".py") if path.suffix == ".pyi" else (".py", ".pyi")
    candidates = (
        candidate
        for suffix in suffixes
        for candidate in (
            module_path.with_suffix(suffix),
            module_path / f"__init__{suffix}",
        )
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _reexport_target(path: Path, name: str) -> tuple[Path, str] | None:
    for statement in _module_tree(path).body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        for imported_name in statement.names:
            if (imported_name.asname or imported_name.name) == name:
                target_path = _relative_import_path(
                    path,
                    statement,
                    imported_name=imported_name.name,
                )
                if target_path is None:
                    continue
                return target_path, imported_name.name
    return None


def _stub_surface_source(path: Path) -> str:
    if path.parent.name != "cutlass":
        return path.read_text(encoding="utf-8")

    package_root = path.parent
    pending = [path]
    visited: set[Path] = set()
    sources: list[str] = []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        sources.append(current.read_text(encoding="utf-8"))
        for statement in _module_tree(current).body:
            if not isinstance(statement, ast.ImportFrom):
                continue
            for imported_name in statement.names:
                target = _relative_import_path(
                    current,
                    statement,
                    imported_name=imported_name.name,
                )
                if (
                    target is not None
                    and target.suffix == ".pyi"
                    and target.is_relative_to(package_root)
                ):
                    pending.append(target)
    return "\n".join(sources)


def _literal_exports(path: Path) -> tuple[str, ...]:
    tree = _module_tree(path)
    for statement in tree.body:
        value = None
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in statement.targets
            )
        ) or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
        ):
            value = statement.value
        if value is not None:
            exports = ast.literal_eval(value)
            assert isinstance(exports, (list, tuple)), path
            assert all(isinstance(name, str) for name in exports), path
            return tuple(exports)
    raise AssertionError(f"{path} must assign a literal __all__")


def _static_declarations(path: Path) -> set[str]:
    declarations: set[str] = set()
    for statement in _module_tree(path).body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            declarations.add(statement.name)
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            declarations.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            declarations.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(statement, ast.ImportFrom):
            assert all(alias.name != "*" for alias in statement.names), path
            declarations.update(alias.asname or alias.name for alias in statement.names)
    return declarations


def _top_level_declaration(
    path: Path,
    name: str,
    *,
    _visited: frozenset[tuple[Path, str]] = frozenset(),
) -> ast.stmt:
    key = (path, name)
    assert key not in _visited, f"cyclic stub reexport for {path}:{name}"
    for statement in _module_tree(path).body:
        if (
            isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            and statement.name == name
        ):
            return statement
    target = _reexport_target(path, name)
    if target is not None:
        target_path, target_name = target
        return _top_level_declaration(
            target_path,
            target_name,
            _visited=_visited | {key},
        )
    raise AssertionError(f"{path} must declare {name}")


def _named_assignment(
    path: Path,
    name: str,
    *,
    _visited: frozenset[tuple[Path, str]] = frozenset(),
) -> ast.expr:
    key = (path, name)
    assert key not in _visited, f"cyclic stub reexport for {path}:{name}"
    for statement in _module_tree(path).body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            assert statement.value is not None
            return statement.value
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return statement.value
    target = _reexport_target(path, name)
    if target is not None:
        target_path, target_name = target
        return _named_assignment(
            target_path,
            target_name,
            _visited=_visited | {key},
        )
    raise AssertionError(f"{path} must assign {name}")


def _union_terms(expression: ast.expr) -> tuple[str, ...]:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        return (*_union_terms(expression.left), *_union_terms(expression.right))
    return (ast.unparse(expression),)


def _typevar_bound(path: Path, name: str) -> str:
    declaration = _named_assignment(path, name)
    assert isinstance(declaration, ast.Call), (path, name)
    bound = next(
        keyword.value for keyword in declaration.keywords if keyword.arg == "bound"
    )
    return ast.unparse(bound)


def _class_methods(path: Path, class_name: str) -> set[str]:
    declaration = _top_level_declaration(path, class_name)
    assert isinstance(declaration, ast.ClassDef), path
    return {
        statement.name
        for statement in declaration.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_declarations(
    path: Path,
    name: str,
    *,
    _visited: frozenset[tuple[Path, str]] = frozenset(),
) -> tuple[ast.FunctionDef, ...]:
    declarations = tuple(
        statement
        for statement in _module_tree(path).body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    )
    if declarations:
        return declarations
    key = (path, name)
    assert key not in _visited, f"cyclic stub reexport for {path}:{name}"
    target = _reexport_target(path, name)
    if target is None:
        return ()
    target_path, target_name = target
    return _function_declarations(
        target_path,
        target_name,
        _visited=_visited | {key},
    )


def _method_declarations(
    path: Path,
    class_name: str,
    method_name: str,
) -> tuple[ast.FunctionDef, ...]:
    declaration = _top_level_declaration(path, class_name)
    assert isinstance(declaration, ast.ClassDef), path
    return tuple(
        statement
        for statement in declaration.body
        if isinstance(statement, ast.FunctionDef) and statement.name == method_name
    )


def _call_declarations(path: Path, name: str) -> tuple[ast.FunctionDef, ...]:
    declarations = _function_declarations(path, name)
    if declarations:
        return declarations
    return _method_declarations(path, name, "__init__")


def _parameter_layout(
    declaration: ast.FunctionDef,
    *,
    drop_self: bool = False,
) -> tuple[tuple[str, str, str], ...]:
    positional = (
        *(("positional_only", item) for item in declaration.args.posonlyargs),
        *(("positional", item) for item in declaration.args.args),
    )
    required_count = len(positional) - len(declaration.args.defaults)
    positional_defaults: tuple[ast.expr | None, ...] = (
        *(None for _ in range(required_count)),
        *declaration.args.defaults,
    )
    layout = [
        (kind, argument.arg, _default_source(default))
        for (kind, argument), default in zip(positional, positional_defaults)
    ]
    layout.extend(
        ("keyword_only", argument.arg, _default_source(default))
        for argument, default in zip(
            declaration.args.kwonlyargs,
            declaration.args.kw_defaults,
        )
    )
    if declaration.args.vararg is not None:
        layout.append(("var_positional", declaration.args.vararg.arg, _REQUIRED))
    if declaration.args.kwarg is not None:
        layout.append(("var_keyword", declaration.args.kwarg.arg, _REQUIRED))
    if drop_self:
        assert layout and layout[0][1] == "self"
        layout = layout[1:]
    return tuple(layout)


def _default_source(default: ast.expr | None) -> str:
    return _REQUIRED if default is None else ast.unparse(default)


def _expected_default_source(default: object) -> str:
    return _REQUIRED if default == _REQUIRED else repr(default)


def _active_cuda_coop_search_root() -> Path:
    spec = importlib.util.find_spec("cuda.coop")
    assert spec is not None
    assert spec.origin is not None
    package_init = Path(spec.origin).resolve()
    assert package_init.name == "__init__.py"
    assert package_init.parent.name == "coop"
    assert package_init.parent.parent.name == "cuda"
    return package_init.parents[2]


def _pyright_environment() -> dict[str, str]:
    environment = os.environ.copy()
    invocation_directory = Path.cwd()
    search_roots = [_active_cuda_coop_search_root()]
    for entry in environment.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (invocation_directory / path).resolve()
        )
        if resolved not in search_roots:
            search_roots.append(resolved)
    environment["PYTHONPATH"] = os.pathsep.join(map(str, search_roots))
    return environment


def _run_pyright_json(*args: str) -> dict[str, object]:
    environment = _pyright_environment()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            *args,
            "--pythonpath",
            sys.executable,
            "--outputjson",
        ],
        check=False,
        cwd=TESTS_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pytest.fail(
            "Pyright did not return JSON output:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    assert isinstance(report, dict)
    assert report.get("version") == "1.1.411"
    if completed.returncode:
        pytest.fail(json.dumps(report.get("generalDiagnostics"), indent=2))
    return report


def _import_bindings(path: Path) -> dict[str, tuple[int, str, str]]:
    bindings: dict[str, tuple[int, str, str]] = {}
    for statement in _module_tree(path).body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        if statement.module is None:
            continue
        for imported_name in statement.names:
            public_name = imported_name.asname or imported_name.name
            bindings[public_name] = (
                statement.level,
                statement.module,
                imported_name.name,
            )
    return bindings


def _lazy_export_names(test: ast.expr, *, path: Path) -> tuple[str, ...]:
    assert isinstance(test, ast.Compare), path
    assert isinstance(test.left, ast.Name) and test.left.id == "name", path
    assert len(test.ops) == 1 and isinstance(test.ops[0], ast.In), path
    assert len(test.comparators) == 1, path
    names = ast.literal_eval(test.comparators[0])
    assert isinstance(names, tuple), path
    assert all(isinstance(name, str) for name in names), path
    return names


def _lazy_relative_module(
    module_expression: ast.expr,
    *,
    path: Path,
) -> tuple[int, str]:
    assert isinstance(module_expression, ast.JoinedStr), path
    assert len(module_expression.values) == 2, path
    base, suffix = module_expression.values
    assert isinstance(base, ast.FormattedValue), path
    assert isinstance(suffix, ast.Constant) and isinstance(suffix.value, str), path
    assert suffix.value.startswith("."), path

    if isinstance(base.value, ast.Name) and base.value.id == "__name__":
        level = 1
    else:
        parent = base.value
        assert isinstance(parent, ast.Subscript), path
        assert isinstance(parent.value, ast.Call), path
        assert isinstance(parent.value.func, ast.Attribute), path
        assert isinstance(parent.value.func.value, ast.Name), path
        assert parent.value.func.value.id == "__name__", path
        assert parent.value.func.attr == "rsplit", path
        level = 2

    return level, suffix.value.removeprefix(".")


def _lazy_runtime_bindings(path: Path) -> dict[str, tuple[int, str, str]]:
    declaration = _top_level_declaration(path, "__getattr__")
    assert isinstance(declaration, ast.FunctionDef), path
    bindings: dict[str, tuple[int, str, str]] = {}

    for branch in declaration.body:
        if not isinstance(branch, ast.If):
            continue
        names = _lazy_export_names(branch.test, path=path)
        import_calls = [
            node
            for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
        ]
        assert len(import_calls) == 1, path
        level, module = _lazy_relative_module(import_calls[0].args[0], path=path)

        getattr_calls = [
            node
            for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
        ]
        assert len(getattr_calls) == 1, path
        source_expression = getattr_calls[0].args[1]
        if isinstance(source_expression, ast.Name):
            assert source_expression.id == "name", path
            strip_factory_prefix = False
        else:
            assert isinstance(source_expression, ast.Call), path
            assert isinstance(source_expression.func, ast.Name), path
            assert source_expression.func.id == "_factory_name", path
            strip_factory_prefix = True

        for name in names:
            assert name not in bindings, (path, name)
            source_name = name.removeprefix("make_") if strip_factory_prefix else name
            bindings[name] = (level, module, source_name)

    return bindings


def _expected_cutlass_wrapper_signature(
    scope: str,
    name: str,
) -> tuple[tuple[tuple[str, str], ...], set[str], bool]:
    group_shape = (
        {"threads_per_block", "dim"} if scope == "block" else {"threads_in_warp"}
    )
    launch_metadata_controls = (
        _CUTLASS_LAUNCH_METADATA_CONTROLS if scope == "block" else set()
    )
    if name == "load":
        return (
            (("positional_only", "source"), ("positional_only", "output")),
            {
                "items_per_thread",
                "valid_items",
                "num_valid_items",
                "oob_default",
                "offset",
                "algorithm",
                "dtype",
                "temp_storage",
                "payload",
                *group_shape,
                *_PRIMS_LOAD_CONTROLS,
                *launch_metadata_controls,
            },
            False,
        )
    if name == "store":
        return (
            (
                ("positional_only", "destination"),
                ("positional_only", "value"),
            ),
            {
                "items_per_thread",
                "valid_items",
                "num_valid_items",
                "algorithm",
                "offset",
                "dtype",
                "temp_storage",
                "payload",
                *group_shape,
                *_PRIMS_STORE_CONTROLS,
                *launch_metadata_controls,
            },
            False,
        )

    is_load = name == "make_load"
    controls = _PRIMS_LOAD_CONTROLS if is_load else _PRIMS_STORE_CONTROLS
    if scope == "block":
        positional = (
            ("positional", "dtype"),
            ("positional", "threads_per_block"),
            ("positional", "items_per_thread"),
            ("positional", "algorithm"),
        )
        keywords = {"dim", "valid_items", "num_valid_items"}
    else:
        positional_names = [
            "dtype",
            "items_per_thread",
            "threads_in_warp",
            "algorithm",
            "num_valid_items",
        ]
        if is_load:
            positional_names.append("oob_default")
        positional = tuple(("positional", item) for item in positional_names)
        keywords = {"valid_items"}
    if is_load and scope == "block":
        keywords.add("oob_default")
    keywords.update(
        {
            "payload",
            "offset",
            *controls,
            *launch_metadata_controls,
        }
    )
    return positional, keywords, False


def test_cuda_coop_declares_pep561_typing_artifacts() -> None:
    missing = [
        artifact.as_posix()
        for artifact in _TYPING_ARTIFACTS
        if not (_COOP_ROOT / artifact).is_file()
    ]

    assert missing == []
    assert not (_COOP_ROOT.parent / "py.typed").exists()


@pytest.mark.parametrize(
    "marker_relative_path",
    [artifact for artifact in _TYPING_ARTIFACTS if artifact.name == "py.typed"],
)
def test_pep561_markers_are_empty(marker_relative_path: Path) -> None:
    marker = _COOP_ROOT / marker_relative_path

    assert marker.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "stub_relative_path",
    [artifact for artifact in _TYPING_ARTIFACTS if artifact.suffix == ".pyi"],
)
def test_public_stub_explicitly_declares_every_export(
    stub_relative_path: Path,
) -> None:
    stub_path = _COOP_ROOT / stub_relative_path
    exports = set(_literal_exports(stub_path))

    assert exports
    assert exports <= _static_declarations(stub_path)


def test_cutlass_facade_reexports_each_typed_primitive_family() -> None:
    package = _COOP_ROOT / "cutlass"
    facade_bindings = _import_bindings(package / "__init__.pyi")

    for stub_relative_path, expected_exports in _CUTLASS_PRIMITIVE_STUB_EXPORTS.items():
        stub_path = package / stub_relative_path
        runtime_path = stub_path.with_suffix(".py")

        assert _literal_exports(stub_path) == expected_exports
        for name in expected_exports:
            assert facade_bindings[name] == (
                1,
                stub_path.stem,
                name,
            )
            assert isinstance(
                _top_level_declaration(runtime_path, name),
                ast.FunctionDef,
            )


def test_facade_includes_aot() -> None:
    source = _stub_surface_source(_COOP_ROOT / "cutlass" / "__init__.pyi")

    assert "class PackInfo:" in source
    assert "def capture(" in source


def test_common_stub_exposes_the_broad_v1_root_contract() -> None:
    expected = {
        "__version__",
        "TempStorage",
        "TempStorageLike",
        "ThreadData",
        "ThreadDataLike",
        *PORTABLE_GROUP_FIRST_EXPORTS,
    }

    assert set(_literal_exports(_COOP_ROOT / "__init__.pyi")) == expected


def test_common_thread_group_declares_the_neutral_method_intersection() -> None:
    path = _COOP_ROOT / "__init__.pyi"
    declaration = _top_level_declaration(path, "ThreadGroup")

    assert isinstance(declaration, ast.ClassDef)
    assert _COMMON_THREAD_GROUP_METHODS <= _class_methods(path, "ThreadGroup")


def test_common_thread_group_carries_a_private_covariant_kind_parameter() -> None:
    path = _COOP_ROOT / "__init__.pyi"
    declaration = _top_level_declaration(path, "ThreadGroup")

    assert isinstance(declaration, ast.ClassDef)
    assert "Generic[_GroupKindT_co]" in {
        ast.unparse(base) for base in declaration.bases
    }
    kind = _method_declarations(path, "ThreadGroup", "kind")
    assert len(kind) == 1
    assert ast.unparse(kind[0].returns) == "_GroupKindT_co"


def test_thread_group_query_results_match_backend_scalar_defaults() -> None:
    expected = {
        None: {
            "rank": "_IntegerValue",
            "count": "_IntegerValue",
            "is_member": "_IntegerValue",
        },
        "cutlass": {
            "rank": "_CompilerIntegerLike",
            "count": "_CompilerIntegerLike",
            "is_member": "_CompilerIntegerLike",
        },
        "numba_mlir": {
            "rank": "_NumpyInt32",
            "count": "_NumpyInt32",
            "is_member": "_NumpyUint8",
        },
    }

    for backend, methods in expected.items():
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        for method, result in methods.items():
            declarations = _method_declarations(path, "ThreadGroup", method)
            assert len(declarations) == 1
            assert ast.unparse(declarations[0].returns) == result


def test_thread_group_sync_typing_excludes_grid_except_cutlass() -> None:
    synchronizable = _named_assignment(
        _COOP_ROOT / "_typing.pyi",
        "_SynchronizableGroupKind",
    )
    assert ast.unparse(synchronizable) == (
        "Literal['thread', 'warp', 'block', 'cluster', "
        "'threads_within_warp', 'warps_within_block']"
    )

    for backend in (None, "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        for method in ("sync", "sync_aligned"):
            declarations = _method_declarations(path, "ThreadGroup", method)
            assert len(declarations) == 1
            self_argument = declarations[0].args.args[0]
            assert ast.unparse(self_argument.annotation) == (
                "ThreadGroup[_SynchronizableGroupKind]"
            )

    for method in ("sync", "sync_aligned"):
        declarations = _method_declarations(
            _COOP_ROOT / "cutlass" / "__init__.pyi",
            "ThreadGroup",
            method,
        )
        assert len(declarations) == 1
        assert declarations[0].args.args[0].annotation is None


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
@pytest.mark.parametrize(
    "name",
    ["this_thread", "this_warp", "this_block", "this_cluster", "this_grid"],
)
def test_group_constructor_stub_parameter_layout_is_identical(
    backend: str,
    name: str,
) -> None:
    common = _function_declarations(_COOP_ROOT / "__init__.pyi", name)
    qualified = _function_declarations(_COOP_ROOT / backend / "__init__.pyi", name)

    assert len(common) == len(qualified) == 1
    assert _parameter_layout(qualified[0]) == _parameter_layout(common[0])


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("this_thread", "thread"),
        ("this_warp", "warp"),
        ("this_block", "block"),
        ("this_cluster", "cluster"),
        ("this_grid", "grid"),
    ],
)
def test_group_constructors_return_the_exact_static_kind(
    name: str,
    kind: str,
) -> None:
    expected = f"ThreadGroup[Literal['{kind}']]"
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, name)
        assert len(declarations) == 1
        assert ast.unparse(declarations[0].returns) == expected


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
@pytest.mark.parametrize("name", sorted(_COMMON_THREAD_GROUP_METHODS))
def test_thread_group_stub_method_parameter_layout_is_identical(
    backend: str,
    name: str,
) -> None:
    common = _method_declarations(_COOP_ROOT / "__init__.pyi", "ThreadGroup", name)
    qualified = _method_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "ThreadGroup",
        name,
    )

    assert common
    assert len(qualified) == len(common)
    assert tuple(map(_parameter_layout, qualified)) == tuple(
        map(_parameter_layout, common)
    )


@pytest.mark.parametrize("backend", [None, "cutlass", "numba_mlir"])
def test_thread_group_group_by_is_limited_to_partitionable_receivers(
    backend: str | None,
) -> None:
    path = (
        _COOP_ROOT / "__init__.pyi"
        if backend is None
        else _COOP_ROOT / backend / "__init__.pyi"
    )
    declarations = _method_declarations(path, "ThreadGroup", "group_by")

    assert len(declarations) == 2
    assert tuple(ast.unparse(declaration.returns) for declaration in declarations) == (
        "ThreadGroup[Literal['threads_within_warp']]",
        "ThreadGroup[Literal['warps_within_block']]",
    )


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
def test_qualified_warp_group_aliases_include_logical_warps(backend: str) -> None:
    path = _COOP_ROOT / backend / "__init__.pyi"

    assert ast.unparse(_named_assignment(path, "_MemoryGroup")) == (
        "ThreadGroup[Literal['warp', 'threads_within_warp', 'block']]"
    )
    assert ast.unparse(_named_assignment(path, "_WarpGroup")) == (
        "ThreadGroup[Literal['warp', 'threads_within_warp']]"
    )


@pytest.mark.parametrize(
    ("name", "group_annotation"),
    [
        ("load", ("_BlockGroup", "_BlockGroup", "_WarpGroup", "_WarpGroup")),
        ("store", ("_BlockGroup", "_WarpGroup")),
        (
            "reduce",
            (
                "_ReductionGroup",
                "_ReductionGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
            ),
        ),
        (
            "sum",
            (
                "_ReductionGroup",
                "_ReductionGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
            ),
        ),
        (
            "scan",
            (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
                "_WarpGroup",
            ),
        ),
        ("exclusive_sum", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        ("inclusive_sum", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        (
            "exclusive_scan",
            (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
            ),
        ),
        ("inclusive_scan", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        ("exchange", "_MemoryGroup"),
        ("adjacent_difference", "_BlockGroup"),
        ("discontinuity", "_BlockGroup"),
        ("shuffle", "_BlockGroup"),
        ("merge_sort_keys", "_MemoryGroup"),
        ("radix_sort_keys", "_BlockGroup"),
        ("radix_rank", "_BlockGroup"),
        ("histogram", "_BlockGroup"),
        ("run_length_decode", "_BlockGroup"),
        ("topk_max_keys", "_BlockGroup"),
        ("topk_min_keys", "_BlockGroup"),
    ],
)
def test_common_primitives_encode_the_certified_group_matrix(
    name: str,
    group_annotation: str | tuple[str, ...],
) -> None:
    declarations = _function_declarations(_COOP_ROOT / "__init__.pyi", name)

    assert declarations
    actual = tuple(
        ast.unparse(declaration.args.posonlyargs[0].annotation)
        for declaration in declarations
    )
    if isinstance(group_annotation, str):
        assert all(annotation == group_annotation for annotation in actual)
    else:
        assert actual == group_annotation


@pytest.mark.parametrize(
    ("name", "group_annotation"),
    [
        ("load", ("_BlockGroup", "_BlockGroup", "_WarpGroup", "_WarpGroup")),
        ("store", ("_BlockGroup", "_WarpGroup")),
        (
            "reduce",
            (
                "_ReductionGroup",
                "_ReductionGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
            ),
        ),
        (
            "sum",
            (
                "_ReductionGroup",
                "_ReductionGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
            ),
        ),
        (
            "scan",
            (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
                "_WarpGroup",
            ),
        ),
        ("exclusive_sum", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        ("inclusive_sum", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        (
            "exclusive_scan",
            (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
            ),
        ),
        ("inclusive_scan", ("_BlockGroup", "_BlockGroup", "_WarpGroup")),
        ("exchange", "_MemoryGroup"),
        ("adjacent_difference", "_BlockGroup"),
        ("discontinuity", "_BlockGroup"),
        ("shuffle", "_BlockGroup"),
        ("merge_sort_keys", "_MemoryGroup"),
        ("merge_sort_pairs", "_MemoryGroup"),
        ("radix_sort_keys", "_BlockGroup"),
        ("radix_sort_pairs", "_BlockGroup"),
        ("radix_rank", "_BlockGroup"),
        ("histogram", "_BlockGroup"),
        ("run_length_decode", "_BlockGroup"),
        ("topk_max_keys", "_BlockGroup"),
        ("topk_min_keys", "_BlockGroup"),
        ("topk_max_pairs", "_BlockGroup"),
        ("topk_min_pairs", "_BlockGroup"),
    ],
)
def test_qualified_primitives_encode_the_certified_group_matrix(
    name: str,
    group_annotation: str | tuple[str, ...],
) -> None:
    for backend in ("cutlass", "numba_mlir"):
        declarations = _function_declarations(
            _COOP_ROOT / backend / "__init__.pyi",
            name,
        )

        assert declarations
        actual = tuple(
            ast.unparse(declaration.args.posonlyargs[0].annotation)
            for declaration in declarations
        )
        if name in {"reduce", "sum"} and backend == "cutlass":
            assert isinstance(group_annotation, tuple)
            assert actual == (*group_annotation, "_ReductionGroup")
        elif name == "scan" and backend == "cutlass":
            assert actual == (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
                "_WarpGroup",
            )
        elif name == "exclusive_scan" and backend == "cutlass":
            assert actual == (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
                "_WarpGroup",
            )
        elif (
            name in {"exclusive_sum", "inclusive_sum", "inclusive_scan"}
            and backend == "cutlass"
        ):
            assert actual == (
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_WarpGroup",
            )
        elif name == "merge_sort_keys" and backend == "cutlass":
            assert actual == (
                "_BlockGroup",
                "_BlockGroup",
                "_MergeSortWarpGroup",
                "_MergeSortWarpGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_MergeSortWarpGroup",
                "_MergeSortWarpGroup",
                "_BlockGroup",
                "_BlockGroup",
            )
        elif name == "merge_sort_pairs" and backend == "cutlass":
            assert actual == (
                "_BlockGroup",
                "_BlockGroup",
                "_MergeSortWarpGroup",
                "_MergeSortWarpGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_BlockGroup",
                "_MergeSortWarpGroup",
                "_MergeSortWarpGroup",
            )
        elif name == "exchange":
            assert actual == (
                ("_BlockGroup", "_WarpGroup", "_BlockGroup", "_WarpGroup")
                if backend == "cutlass"
                else ("_BlockGroup", "_WarpGroup")
            )
        elif isinstance(group_annotation, str):
            assert all(annotation == group_annotation for annotation in actual)
        else:
            assert actual == group_annotation


@pytest.mark.parametrize(
    "name",
    ["scan", "exclusive_sum", "inclusive_sum", "exclusive_scan", "inclusive_scan"],
)
def test_physical_warp_scan_overloads_reject_block_only_controls(name: str) -> None:
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, name)
        warp_declarations = [
            declaration
            for declaration in declarations
            if ast.unparse(declaration.args.posonlyargs[0].annotation) == "_WarpGroup"
        ]

        expected_count = {"scan": 3, "exclusive_scan": 2}.get(name, 1)
        assert len(warp_declarations) == expected_count
        for declaration in warp_declarations:
            annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in declaration.args.kwonlyargs
            }
            assert annotations["algorithm"] == "None"
            assert annotations["temp_storage"] == "None"


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
@pytest.mark.parametrize(
    "name",
    ["scan", "exclusive_sum", "inclusive_sum", "exclusive_scan", "inclusive_scan"],
)
def test_qualified_scan_valid_items_are_warp_only(
    backend: str,
    name: str,
) -> None:
    declarations = _function_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        name,
    )

    for declaration in declarations:
        group = ast.unparse(declaration.args.posonlyargs[0].annotation)
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        defaults = {
            argument.arg: _default_source(default)
            for argument, default in zip(
                declaration.args.kwonlyargs,
                declaration.args.kw_defaults,
            )
        }
        assert annotations["valid_items"] == (
            "None" if group == "_BlockGroup" else "_ValidItems | None"
        )
        assert defaults["valid_items"] == "None"


def test_scan_operator_aliases_partition_sum_from_non_sum_operations() -> None:
    path = _COOP_ROOT / "_typing.pyi"

    assert ast.unparse(_named_assignment(path, "_SumScanOperator")) == (
        "Literal['+', 'sum', 'add', 'plus']"
    )
    assert ast.unparse(_named_assignment(path, "_NonSumScanOperator")) == (
        "Literal['*', 'mul', 'multiply', 'multiplies', 'min', 'minimum', "
        "'max', 'maximum', '&', 'bit_and', '|', 'bit_or', '^', 'bit_xor']"
    )
    assert ast.unparse(_named_assignment(path, "ScanOperator")) == (
        "_SumScanOperator | _NonSumScanOperator"
    )
    exports = _literal_exports(path)
    assert "ScanOperator" in exports
    assert "_SumScanOperator" not in exports
    assert "_NonSumScanOperator" not in exports


@pytest.mark.parametrize(
    ("backend", "shape_count"),
    [(None, 3), ("cutlass", 4), ("numba_mlir", 3)],
)
def test_scan_overloads_correlate_mode_operator_and_initial_value(
    backend: str | None,
    shape_count: int,
) -> None:
    path = (
        _COOP_ROOT / "__init__.pyi"
        if backend is None
        else _COOP_ROOT / backend / "__init__.pyi"
    )

    scan_declarations = _function_declarations(path, "scan")
    assert len(scan_declarations) == shape_count * 3
    for index, declaration in enumerate(scan_declarations):
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        defaults = {
            argument.arg: _default_source(default)
            for argument, default in zip(
                declaration.args.kwonlyargs,
                declaration.args.kw_defaults,
            )
        }
        variant = index % 3
        if variant == 0:
            assert annotations["mode"] == "Literal['exclusive']"
            assert annotations["scan_op"] == "_SumScanOperator | None"
            assert defaults["mode"] == "'exclusive'"
            assert defaults["scan_op"] == "None"
            assert defaults["initial_value"] == "None"
        elif variant == 1:
            assert annotations["mode"] == "Literal['exclusive']"
            if backend == "numba_mlir":
                assert annotations["scan_op"].startswith(
                    "_NonSumScanOperator | Callable["
                )
            else:
                assert annotations["scan_op"] == "_NonSumScanOperator"
            assert defaults["mode"] == "'exclusive'"
            assert defaults["scan_op"] == _REQUIRED
            assert defaults["initial_value"] == _REQUIRED
        else:
            assert annotations["mode"] == "Literal['inclusive']"
            assert annotations["scan_op"].startswith("_ScanOperator")
            assert annotations["initial_value"] == "None"
            assert defaults["mode"] == _REQUIRED
            assert defaults["scan_op"] == "None"
            assert defaults["initial_value"] == "None"

    exclusive_declarations = _function_declarations(path, "exclusive_scan")
    assert len(exclusive_declarations) == shape_count * 2
    for index, declaration in enumerate(exclusive_declarations):
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        defaults = {
            argument.arg: _default_source(default)
            for argument, default in zip(
                declaration.args.kwonlyargs,
                declaration.args.kw_defaults,
            )
        }
        if index % 2 == 0:
            assert annotations["scan_op"] == "_SumScanOperator | None"
            assert defaults["scan_op"] == "None"
            assert defaults["initial_value"] == "None"
        else:
            if backend == "numba_mlir":
                assert annotations["scan_op"].startswith(
                    "_NonSumScanOperator | Callable["
                )
            else:
                assert annotations["scan_op"] == "_NonSumScanOperator"
            assert defaults["scan_op"] == _REQUIRED
            assert defaults["initial_value"] == _REQUIRED


@pytest.mark.parametrize("name", ["reduce", "sum"])
def test_reduce_stubs_encode_direct_cub_selector_routes(name: str) -> None:
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, name)
        assert len(declarations) == (6 if backend == "cutlass" else 5)

        full_payload, full_scalar, block_valid, block_algorithm, direct_warp = (
            declarations[:5]
        )
        assert ast.unparse(full_payload.args.posonlyargs[0].annotation) == (
            "_ReductionGroup"
        )
        assert ast.unparse(full_scalar.args.posonlyargs[0].annotation) == (
            "_ReductionGroup"
        )

        for declaration in (full_payload, full_scalar):
            annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in declaration.args.kwonlyargs
            }
            assert annotations["broadcast"] == "bool"
            assert annotations["valid_items"] == "None"
            assert annotations["algorithm"] == "None"

        valid_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in block_valid.args.kwonlyargs
        }
        assert ast.unparse(block_valid.args.posonlyargs[0].annotation) == "_BlockGroup"
        assert ast.unparse(block_valid.args.posonlyargs[1].annotation) == (
            "_ScalarValueT"
        )
        assert valid_annotations["broadcast"] == "Literal[False]"
        assert valid_annotations["valid_items"] == "_ValidItems"
        assert valid_annotations["algorithm"] == "_ReduceAlgorithm | None"

        algorithm_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in block_algorithm.args.kwonlyargs
        }
        assert ast.unparse(block_algorithm.args.posonlyargs[0].annotation) == (
            "_BlockGroup"
        )
        assert ast.unparse(block_algorithm.args.posonlyargs[1].annotation) == (
            "_ScalarValueT"
        )
        assert algorithm_annotations["broadcast"] == "Literal[False]"
        assert algorithm_annotations["valid_items"] == "None"
        assert algorithm_annotations["algorithm"] == "_ReduceAlgorithm"

        warp_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in direct_warp.args.kwonlyargs
        }
        assert ast.unparse(direct_warp.args.posonlyargs[0].annotation) == "_WarpGroup"
        assert ast.unparse(direct_warp.args.posonlyargs[1].annotation) == (
            "_ScalarValueT"
        )
        assert warp_annotations["broadcast"] == "Literal[False]"
        assert warp_annotations["valid_items"] == "_ValidItems"
        assert warp_annotations["algorithm"] == "None"

        if backend == "cutlass":
            tensor = declarations[5]
            assert ast.unparse(tensor.args.posonlyargs[0].annotation) == (
                "_ReductionGroup"
            )
            assert ast.unparse(tensor.args.posonlyargs[1].annotation) == (
                "_CutlassTensorSample | _CutlassTensorSSASample"
            )
            assert ast.unparse(tensor.returns) == "Any"


def test_merge_sort_stubs_encode_integral_full_and_partial_routes() -> None:
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, "merge_sort_keys")
        assert len(declarations) == (
            10 if backend == "cutlass" else 4 if backend == "numba_mlir" else 2
        )

        payload_offset = 4 if backend == "cutlass" else 0
        full_payload, partial_payload = declarations[
            payload_offset : payload_offset + 2
        ]
        expected_payload = (
            "ThreadData[_IntegerKeyT]"
            if backend == "cutlass"
            else (
                "ThreadDataLike[_IntegerKeyT]"
                if backend is None
                else "_ThreadDataLike[_NumbaMergeSortKeyT]"
            )
        )
        expected_group = "_BlockGroup" if backend == "cutlass" else "_MemoryGroup"

        for declaration in (full_payload, partial_payload):
            assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
                expected_group
            )
            assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
                expected_payload
            )
            assert ast.unparse(declaration.returns) == expected_payload

        full_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in full_payload.args.kwonlyargs
        }
        partial_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in partial_payload.args.kwonlyargs
        }
        assert full_annotations["valid_items"] == "None"
        assert full_annotations["oob_default"] == "None"
        assert partial_annotations["valid_items"] == "_ValidItems"
        assert partial_annotations["oob_default"] == (
            "_NumbaMergeSortKeyT" if backend == "numba_mlir" else "_IntegerKeyT"
        )

        if backend == "numba_mlir":
            for scalar in declarations[2:]:
                assert ast.unparse(scalar.args.posonlyargs[0].annotation) == (
                    "_MemoryGroup"
                )
                assert ast.unparse(scalar.args.posonlyargs[1].annotation) == (
                    "_NumbaMergeSortKeyT"
                )
                assert ast.unparse(scalar.returns) == "_NumbaMergeSortKeyT"

    cutlass_declarations = _function_declarations(
        _COOP_ROOT / "cutlass" / "__init__.pyi",
        "merge_sort_keys",
    )
    (
        block_tensor_full,
        block_tensor_partial,
        warp_tensor_full,
        warp_tensor_partial,
        _,
        _,
        warp_payload_full,
        warp_payload_partial,
        scalar_full,
        scalar_partial,
    ) = cutlass_declarations
    for declaration in (scalar_full, scalar_partial):
        assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
            "_BlockGroup"
        )
        assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
            "_IntegerKeyT"
        )
        assert ast.unparse(declaration.returns) == "_IntegerKeyT"
    for declaration in (block_tensor_full, block_tensor_partial):
        assert ast.unparse(declaration.args.posonlyargs[0].annotation) == "_BlockGroup"
        assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
            "_CutlassTensorSample | _CutlassTensorSSASample"
        )
        assert ast.unparse(declaration.returns) == "ThreadData[Any]"
    for declaration in (warp_tensor_full, warp_tensor_partial):
        assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
            "_MergeSortWarpGroup"
        )
        assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
            "_CutlassTensorSample | _CutlassTensorSSASample"
        )
        assert ast.unparse(declaration.returns) == "ThreadData[Any]"
    for declaration in (warp_payload_full, warp_payload_partial):
        assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
            "_MergeSortWarpGroup"
        )
        assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
            "ThreadData[_WarpMergeSortKeyT]"
        )
        assert ast.unparse(declaration.returns) == ("ThreadData[_WarpMergeSortKeyT]")


def test_radix_sort_stubs_encode_portable_and_qualified_key_routes() -> None:
    paths = {
        None: _COOP_ROOT / "__init__.pyi",
        "cutlass": _COOP_ROOT / "cutlass" / "__init__.pyi",
        "numba_mlir": _COOP_ROOT / "numba_mlir" / "__init__.pyi",
    }
    expected_routes = {
        None: (("ThreadDataLike[_IntegerKeyT]", "ThreadDataLike[_IntegerKeyT]"),),
        "cutlass": (
            ("ThreadData[_IntegerKeyT]", "ThreadData[_IntegerKeyT]"),
            ("_IntegerKeyT", "_IntegerKeyT"),
            (
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "ThreadData[Any]",
            ),
        ),
        "numba_mlir": (
            ("_ThreadDataLike[_IntegerKeyT]", "_ThreadDataLike[_IntegerKeyT]"),
            ("_IntegerKeyT", "_IntegerKeyT"),
        ),
    }

    for backend, path in paths.items():
        declarations = _function_declarations(path, "radix_sort_keys")
        assert len(declarations) == len(expected_routes[backend])

        if backend is not None:
            assert all(
                any(
                    isinstance(decorator, ast.Name) and decorator.id == "overload"
                    for decorator in declaration.decorator_list
                )
                for declaration in declarations
            )

        for declaration, (keys_annotation, return_annotation) in zip(
            declarations,
            expected_routes[backend],
        ):
            assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
                "_BlockGroup"
            )
            assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
                keys_annotation
            )
            assert ast.unparse(declaration.returns) == return_annotation

            annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in declaration.args.kwonlyargs
            }
            assert annotations["begin_bit"] == "_IntegerValue"
            assert annotations["end_bit"] == "_IntegerValue | None"
            assert annotations["descending"] == "bool"
            assert annotations["temp_storage"] == (
                "TempStorageLike | None" if backend is None else "TempStorage | None"
            )

            docstring = ast.get_docstring(declaration) or ""
            for parameter in (
                *declaration.args.posonlyargs,
                *declaration.args.kwonlyargs,
            ):
                assert f"``{parameter.arg}``" in docstring


def test_qualified_pair_sort_stubs_encode_evidenced_payload_routes() -> None:
    cutlass_path = _COOP_ROOT / "cutlass" / "__init__.pyi"
    numba_path = _COOP_ROOT / "numba_mlir" / "__init__.pyi"

    cutlass_merge = _function_declarations(cutlass_path, "merge_sort_pairs")
    assert len(cutlass_merge) == 10
    assert tuple(
        (
            ast.unparse(declaration.args.posonlyargs[0].annotation),
            ast.unparse(declaration.args.posonlyargs[1].annotation),
            ast.unparse(declaration.args.posonlyargs[2].annotation),
            ast.unparse(declaration.returns),
        )
        for declaration in cutlass_merge
    ) == (
        (
            "_BlockGroup",
            "ThreadData[_IntegerKeyT]",
            "ThreadData[_CutlassPairValueT]",
            "tuple[ThreadData[_IntegerKeyT], ThreadData[_CutlassPairValueT]]",
        ),
        (
            "_BlockGroup",
            "ThreadData[_IntegerKeyT]",
            "ThreadData[_CutlassPairValueT]",
            "tuple[ThreadData[_IntegerKeyT], ThreadData[_CutlassPairValueT]]",
        ),
        (
            "_MergeSortWarpGroup",
            "ThreadData[_WarpMergeSortKeyT]",
            "ThreadData[_CutlassPairValueT]",
            "tuple[ThreadData[_WarpMergeSortKeyT], ThreadData[_CutlassPairValueT]]",
        ),
        (
            "_MergeSortWarpGroup",
            "ThreadData[_WarpMergeSortKeyT]",
            "ThreadData[_CutlassPairValueT]",
            "tuple[ThreadData[_WarpMergeSortKeyT], ThreadData[_CutlassPairValueT]]",
        ),
        (
            "_BlockGroup",
            "_IntegerKeyT",
            "_CutlassPairValueT",
            "tuple[_IntegerKeyT, _CutlassPairValueT]",
        ),
        (
            "_BlockGroup",
            "_IntegerKeyT",
            "_CutlassPairValueT",
            "tuple[_IntegerKeyT, _CutlassPairValueT]",
        ),
        *(
            (
                group,
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "tuple[ThreadData[Any], ThreadData[Any]]",
            )
            for group in (
                "_BlockGroup",
                "_BlockGroup",
                "_MergeSortWarpGroup",
                "_MergeSortWarpGroup",
            )
        ),
    )

    numba_merge = _function_declarations(numba_path, "merge_sort_pairs")
    assert len(numba_merge) == 4
    assert tuple(
        (
            ast.unparse(declaration.args.posonlyargs[1].annotation),
            ast.unparse(declaration.args.posonlyargs[2].annotation),
            ast.unparse(declaration.returns),
        )
        for declaration in numba_merge
    ) == (
        (
            "_ThreadDataLike[_NumbaMergeSortKeyT]",
            "_ThreadDataLike[_NumbaPairValueT]",
            "tuple[_ThreadDataLike[_NumbaMergeSortKeyT], _ThreadDataLike[_NumbaPairValueT]]",
        ),
        (
            "_ThreadDataLike[_NumbaMergeSortKeyT]",
            "_ThreadDataLike[_NumbaPairValueT]",
            "tuple[_ThreadDataLike[_NumbaMergeSortKeyT], _ThreadDataLike[_NumbaPairValueT]]",
        ),
        (
            "_NumbaMergeSortKeyT",
            "_NumbaPairValueT",
            "tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]",
        ),
        (
            "_NumbaMergeSortKeyT",
            "_NumbaPairValueT",
            "tuple[_NumbaMergeSortKeyT, _NumbaPairValueT]",
        ),
    )

    for declarations in (cutlass_merge, numba_merge):
        assert all(
            any(
                isinstance(decorator, ast.Name) and decorator.id == "overload"
                for decorator in declaration.decorator_list
            )
            for declaration in declarations
        )
        for full, partial in zip(declarations[::2], declarations[1::2]):
            full_annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in full.args.kwonlyargs
            }
            partial_annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in partial.args.kwonlyargs
            }
            assert full_annotations["valid_items"] == "None"
            assert full_annotations["oob_default"] == "None"
            assert partial_annotations["valid_items"] == "_ValidItems"
            assert partial_annotations["oob_default"] != "object"

    cutlass_radix = _function_declarations(cutlass_path, "radix_sort_pairs")
    numba_radix = _function_declarations(numba_path, "radix_sort_pairs")
    assert len(cutlass_radix) == 3
    assert len(numba_radix) == 2
    assert all(
        any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in declaration.decorator_list
        )
        for declaration in cutlass_radix
    )
    assert tuple(
        (
            ast.unparse(declaration.args.posonlyargs[1].annotation),
            ast.unparse(declaration.args.posonlyargs[2].annotation),
            ast.unparse(declaration.returns),
        )
        for declaration in cutlass_radix
    ) == (
        (
            "ThreadData[_IntegerKeyT]",
            "ThreadData[_CutlassPairValueT]",
            "tuple[ThreadData[_IntegerKeyT], ThreadData[_CutlassPairValueT]]",
        ),
        (
            "_IntegerKeyT",
            "_CutlassPairValueT",
            "tuple[_IntegerKeyT, _CutlassPairValueT]",
        ),
        (
            "_CutlassTensorSample | _CutlassTensorSSASample",
            "_CutlassTensorSample | _CutlassTensorSSASample",
            "tuple[ThreadData[Any], ThreadData[Any]]",
        ),
    )
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[1].annotation)
        for declaration in numba_radix
    ) == ("_ThreadDataLike[_IntegerKeyT]", "_IntegerKeyT")
    for declaration in (*cutlass_radix, *numba_radix):
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        assert annotations["begin_bit"] == "_IntegerValue"
        assert annotations["end_bit"] == "_IntegerValue | None"


@pytest.mark.parametrize(
    ("backend", "operation", "runtime_relative_path"),
    [
        ("cutlass", "merge_sort_pairs", Path("cutlass/_group_merge_sort.py")),
        ("cutlass", "radix_sort_pairs", Path("cutlass/_group_radix.py")),
        ("numba_mlir", "merge_sort_pairs", Path("numba_mlir/_group_ops.py")),
        ("numba_mlir", "radix_sort_pairs", Path("numba_mlir/_group_ops.py")),
    ],
)
def test_qualified_pair_stub_layout_matches_runtime_parameter_order(
    backend: str,
    operation: str,
    runtime_relative_path: Path,
) -> None:
    runtime = _function_declarations(_COOP_ROOT / runtime_relative_path, operation)
    stubs = _function_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        operation,
    )

    assert len(runtime) == 1
    runtime_layout = _parameter_layout(runtime[0])
    runtime_shape = tuple((kind, name) for kind, name, _ in runtime_layout)
    for stub in stubs:
        stub_layout = _parameter_layout(stub)
        assert tuple((kind, name) for kind, name, _ in stub_layout) == runtime_shape
        for (_, name, runtime_default), (_, _, stub_default) in zip(
            runtime_layout,
            stub_layout,
        ):
            if operation == "merge_sort_pairs" and name in {
                "valid_items",
                "oob_default",
            }:
                assert stub_default in {"None", _REQUIRED}
            else:
                assert stub_default == runtime_default


def test_topk_key_stubs_encode_portable_and_qualified_payload_routes() -> None:
    paths = {
        None: _COOP_ROOT / "__init__.pyi",
        "cutlass": _COOP_ROOT / "cutlass" / "__init__.pyi",
        "numba_mlir": _COOP_ROOT / "numba_mlir" / "__init__.pyi",
    }
    expected_key_routes = {
        None: (
            (
                "ThreadDataLike[_IntegerKeyT]",
                "ThreadDataLike[_IntegerKeyT]",
            ),
        ),
        "cutlass": (
            (
                "ThreadData[_CutlassTopKKeyT]",
                "ThreadData[_CutlassTopKKeyT]",
            ),
            ("_CutlassTopKKeyT", "_CutlassTopKKeyT"),
            (
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "ThreadData[Any]",
            ),
        ),
        "numba_mlir": (
            (
                "_ThreadDataLike[_NumbaTopKKeyT]",
                "_ThreadDataLike[_NumbaTopKKeyT]",
            ),
        ),
    }
    for operation in ("topk_max_keys", "topk_min_keys"):
        for backend, path in paths.items():
            declarations = _function_declarations(path, operation)
            assert len(declarations) == len(expected_key_routes[backend])

            for declaration, (keys_annotation, return_annotation) in zip(
                declarations,
                expected_key_routes[backend],
            ):
                assert ast.unparse(declaration.args.posonlyargs[0].annotation) == (
                    "_BlockGroup"
                )
                assert ast.unparse(declaration.args.posonlyargs[1].annotation) == (
                    keys_annotation
                )
                assert ast.unparse(declaration.args.posonlyargs[2].annotation) == (
                    "_IntegerValue"
                )
                assert ast.unparse(declaration.returns) == return_annotation

                annotations = {
                    argument.arg: ast.unparse(argument.annotation)
                    for argument in declaration.args.kwonlyargs
                }
                assert annotations["valid_items"] == "_ValidItems | None"
                assert annotations["begin_bit"] == "_IntegerValue"
                assert annotations["end_bit"] == "_IntegerValue | None"

                docstring = ast.get_docstring(declaration) or ""
                for parameter in (
                    *declaration.args.posonlyargs,
                    *declaration.args.kwonlyargs,
                ):
                    assert f"``{parameter.arg}``" in docstring


def test_common_thread_data_is_a_factory_without_cutlass_class_helpers() -> None:
    declaration = _top_level_declaration(_COOP_ROOT / "__init__.pyi", "ThreadData")

    assert isinstance(declaration, ast.FunctionDef)
    assert not isinstance(declaration, ast.ClassDef)


def test_common_thread_data_uses_the_shared_mutable_payload_protocol() -> None:
    protocol_path = _COOP_ROOT / "_typing.pyi"

    assert {"__getitem__", "__setitem__"} <= _class_methods(
        protocol_path,
        "ThreadDataLike",
    )
    assert _CUTLASS_THREAD_DATA_HELPERS.isdisjoint(
        _class_methods(protocol_path, "ThreadDataLike")
    )


@pytest.mark.parametrize(
    ("backend", "suffix"),
    [("cutlass", ("values", "None")), ("numba_mlir", ("alignas", "8"))],
)
def test_thread_data_stub_preserves_the_exact_common_prefix(
    backend: str,
    suffix: tuple[str, str],
) -> None:
    common = _call_declarations(_COOP_ROOT / "__init__.pyi", "ThreadData")
    qualified = _call_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "ThreadData",
    )

    assert len(common) == len(qualified) == 2
    for common_declaration, qualified_declaration in zip(common, qualified):
        common_layout = _parameter_layout(common_declaration)
        qualified_layout = _parameter_layout(
            qualified_declaration,
            drop_self=backend == "cutlass",
        )
        assert qualified_layout == (
            *common_layout,
            ("keyword_only", *suffix),
        )


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
def test_temp_storage_stub_parameter_layout_is_identical(backend: str) -> None:
    common = _call_declarations(_COOP_ROOT / "__init__.pyi", "TempStorage")
    qualified = _call_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "TempStorage",
    )

    assert len(common) == len(qualified) == 1
    assert _parameter_layout(qualified[0], drop_self=True) == _parameter_layout(
        common[0]
    )
    assert _parameter_layout(common[0]) == (
        ("positional", "size_in_bytes", "None"),
        ("positional", "alignment", "None"),
        ("positional", "auto_sync", "None"),
        ("positional", "sharing", "'shared'"),
    )


@pytest.mark.parametrize(
    ("name", "keywords"),
    _PORTABLE_PRIMITIVE_KEYWORDS.items(),
)
def test_common_primitive_stubs_accept_only_portable_parameters(
    name: str,
    keywords: tuple[str, ...],
) -> None:
    declarations = [
        statement
        for statement in _module_tree(_COOP_ROOT / "__init__.pyi").body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    ]

    assert declarations
    counter_dtype_defaults = []
    reduction_defaults = []
    scan_defaults = []
    merge_sort_defaults = []
    load_defaults = []
    adjacent_difference_defaults = []
    discontinuity_defaults = []
    for declaration in declarations:
        assert declaration.args.vararg is None
        assert declaration.args.kwarg is None
        assert (
            tuple(item.arg for item in declaration.args.posonlyargs)
            == (_PORTABLE_PRIMITIVE_POSITIONALS[name])
        )
        assert declaration.args.args == []
        assert tuple(item.arg for item in declaration.args.kwonlyargs) == keywords
        actual_defaults = tuple(
            _default_source(default) for default in declaration.args.kw_defaults
        )
        expected_defaults = tuple(
            _expected_default_source(default)
            for default in _PORTABLE_PRIMITIVE_DEFAULTS[name]
        )
        if name == "histogram":
            counter_index = keywords.index("counter_dtype")
            counter_dtype_defaults.append(actual_defaults[counter_index])
            assert actual_defaults[counter_index] in {_REQUIRED, "None"}
            assert (
                actual_defaults[:counter_index] + actual_defaults[counter_index + 1 :]
                == expected_defaults[:counter_index]
                + expected_defaults[counter_index + 1 :]
            )
        elif name in {"reduce", "sum"}:
            reduction_defaults.append(actual_defaults)
        elif name in {"scan", "exclusive_scan"}:
            scan_defaults.append(actual_defaults)
        elif name in {"merge_sort_keys", "merge_sort_pairs"}:
            merge_sort_defaults.append(actual_defaults)
        elif name == "load":
            load_defaults.append(actual_defaults)
        elif name == "adjacent_difference":
            adjacent_difference_defaults.append(actual_defaults)
        elif name == "discontinuity":
            discontinuity_defaults.append(actual_defaults)
        else:
            assert actual_defaults == expected_defaults

    if name == "histogram":
        assert "None" in counter_dtype_defaults
    elif name == "reduce":
        assert reduction_defaults == [
            ("None", "True", "None", "None"),
            ("None", "True", "None", "None"),
            ("None", _REQUIRED, _REQUIRED, "None"),
            ("None", _REQUIRED, "None", _REQUIRED),
            ("None", _REQUIRED, _REQUIRED, "None"),
        ]
    elif name == "sum":
        assert reduction_defaults == [
            ("True", "None", "None"),
            ("True", "None", "None"),
            (_REQUIRED, _REQUIRED, "None"),
            (_REQUIRED, "None", _REQUIRED),
            (_REQUIRED, _REQUIRED, "None"),
        ]
    elif name == "scan":
        assert scan_defaults == 3 * [
            ("'exclusive'", "None", "None", "None", "None"),
            ("'exclusive'", _REQUIRED, _REQUIRED, "None", "None"),
            (_REQUIRED, "None", "None", "None", "None"),
        ]
    elif name == "exclusive_scan":
        assert scan_defaults == 3 * [
            ("None", "None", "None", "None"),
            (_REQUIRED, _REQUIRED, "None", "None"),
        ]
    elif name in {"merge_sort_keys", "merge_sort_pairs"}:
        assert merge_sort_defaults == [
            ("False", "None", "None", "None"),
            ("False", _REQUIRED, _REQUIRED, "None"),
        ]
    elif name == "load":
        assert load_defaults == 2 * [
            ("'direct'", "None", "None", "None", "None"),
            ("'direct'", _REQUIRED, _REQUIRED, "None", "None"),
        ]
    elif name == "adjacent_difference":
        assert adjacent_difference_defaults == [
            ("'left'", "None", "None", "None", "None"),
            (_REQUIRED, "None", "None", "None", "None"),
            (_REQUIRED, "None", "None", _REQUIRED, "None"),
        ]
    elif name == "discontinuity":
        assert discontinuity_defaults == [
            ("'heads'", "None", "None", "None"),
            (_REQUIRED, "None", "None", "None"),
        ]


@pytest.mark.parametrize("name", _PORTABLE_PRIMITIVE_KEYWORDS)
def test_qualified_primitive_stubs_preserve_the_exact_common_prefix(
    name: str,
) -> None:
    common_declarations = _function_declarations(_COOP_ROOT / "__init__.pyi", name)

    assert common_declarations
    for backend in ("cutlass", "numba_mlir"):
        backend_declarations = _function_declarations(
            _COOP_ROOT / backend / "__init__.pyi",
            name,
        )
        expected_count = len(common_declarations)
        if name == "scan" and backend == "cutlass":
            expected_count += 3
        elif name == "exclusive_scan" and backend == "cutlass":
            expected_count += 2
        elif (
            name in {"exclusive_sum", "inclusive_sum", "inclusive_scan"}
            and backend == "cutlass"
        ):
            expected_count += 1
        elif name == "exchange":
            expected_count += 3 if backend == "cutlass" else 1
        elif name == "adjacent_difference":
            expected_count += 6 if backend == "cutlass" else 3
        elif name == "discontinuity":
            expected_count += 7 if backend == "cutlass" else 4
        elif name == "shuffle":
            expected_count += 4 if backend == "cutlass" else 1
        elif name in {"reduce", "sum"} and backend == "cutlass":
            expected_count += 1
        elif name in {"merge_sort_keys", "merge_sort_pairs"} and backend == "cutlass":
            expected_count += 8
        elif (
            name in {"merge_sort_keys", "merge_sort_pairs"} and backend == "numba_mlir"
        ):
            expected_count += 2
        elif name in {"radix_sort_keys", "radix_sort_pairs"} and backend == "cutlass":
            expected_count += 2
        elif (
            name in {"radix_sort_keys", "radix_sort_pairs"} and backend == "numba_mlir"
        ):
            expected_count += 1
        elif name == "radix_rank" and backend == "cutlass":
            expected_count += 2
        elif name == "radix_rank" and backend == "numba_mlir":
            expected_count += 1
        elif name == "run_length_decode" and backend == "cutlass":
            expected_count += 2
        elif name == "run_length_decode" and backend == "numba_mlir":
            expected_count += 1
        elif name == "histogram" and backend == "numba_mlir":
            expected_count += 1
        elif (
            name
            in {
                "topk_max_keys",
                "topk_max_pairs",
                "topk_min_keys",
                "topk_min_pairs",
            }
            and backend == "cutlass"
        ):
            expected_count += 2
        assert len(backend_declarations) == expected_count
        expected_suffix = _BACKEND_SUFFIXES[backend].get(name, ())
        if name in {
            "exchange",
            "radix_rank",
            "radix_sort_keys",
            "radix_sort_pairs",
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
        }:
            declaration_pairs = (
                (common_declarations[0], qualified)
                for qualified in backend_declarations
            )
        else:
            declaration_pairs = zip(common_declarations, backend_declarations)
        for common, qualified in declaration_pairs:
            common_layout = _parameter_layout(common)
            qualified_layout = _parameter_layout(qualified)
            assert qualified_layout[: len(common_layout)] == common_layout
            assert qualified_layout[len(common_layout) :] == tuple(
                ("keyword_only", parameter, repr(default))
                for parameter, default in _suffix_contract(expected_suffix)
            )


def test_common_discontinuity_mode_remains_single_output_only() -> None:
    typing_tree = _module_tree(_COOP_ROOT / "_typing.pyi")
    mode_alias = next(
        statement
        for statement in typing_tree.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "DiscontinuityMode"
    )
    assert mode_alias.value is not None
    assert ast.unparse(mode_alias.value) == "Literal['heads', 'tails']"

    declarations = _function_declarations(
        _COOP_ROOT / "__init__.pyi",
        "discontinuity",
    )
    assert len(declarations) == 2
    assert tuple(
        ast.unparse(declaration.args.kwonlyargs[0].annotation)
        for declaration in declarations
    ) == ("Literal['heads']", "Literal['tails']")
    assert tuple(ast.unparse(declaration.returns) for declaration in declarations) == (
        "ThreadDataLike[int]",
        "ThreadDataLike[int]",
    )


@pytest.mark.parametrize(
    ("backend", "single_return", "pair_return"),
    [
        (
            "cutlass",
            "ThreadData[int]",
            "tuple[ThreadData[int], ThreadData[int]]",
        ),
        (
            "numba_mlir",
            "_ThreadDataLike[int]",
            "tuple[_ThreadDataLike[int], _ThreadDataLike[int]]",
        ),
    ],
)
def test_qualified_discontinuity_overloads_type_single_and_pair_results(
    backend: str,
    single_return: str,
    pair_return: str,
) -> None:
    declarations = _function_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "discontinuity",
    )
    assert len(declarations) == (9 if backend == "cutlass" else 6)
    payload_declarations = [
        declaration
        for declaration in declarations
        if ast.unparse(declaration.args.posonlyargs[1].annotation)
        in {
            "ThreadData[_CutlassNumericT]",
            "_ThreadDataLike[_ItemT]",
        }
    ]
    assert len(payload_declarations) == 3
    heads, tails, pair = payload_declarations
    assert all(
        any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in declaration.decorator_list
        )
        for declaration in declarations
    )

    assert tuple(
        ast.unparse(declaration.args.kwonlyargs[0].annotation)
        for declaration in payload_declarations
    ) == (
        "Literal['heads']",
        "Literal['tails']",
        "Literal['heads_and_tails']",
    )
    assert tuple(
        ast.unparse(declaration.returns) for declaration in payload_declarations
    ) == (single_return, single_return, pair_return)

    layouts = tuple(
        _parameter_layout(declaration) for declaration in (heads, tails, pair)
    )
    assert (
        tuple(parameter[1] for parameter in layouts[0])
        == tuple(parameter[1] for parameter in layouts[1])
        == tuple(parameter[1] for parameter in layouts[2])
    )
    mode_index = next(
        index for index, parameter in enumerate(layouts[0]) if parameter[1] == "mode"
    )
    assert tuple(layout[mode_index][2] for layout in layouts) == (
        "'heads'",
        _REQUIRED,
        _REQUIRED,
    )


@pytest.mark.parametrize(
    ("backend", "scalar_annotation"),
    [("cutlass", "_ScalarValueT"), ("numba_mlir", "_ScalarT")],
)
def test_qualified_comparison_scalar_overloads_preserve_scalar_shape(
    backend: str,
    scalar_annotation: str,
) -> None:
    adjacent_declarations = _function_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "adjacent_difference",
    )
    adjacent_scalar = next(
        declaration
        for declaration in adjacent_declarations
        if ast.unparse(declaration.args.posonlyargs[1].annotation) == scalar_annotation
    )
    assert ast.unparse(adjacent_scalar.returns) == scalar_annotation

    discontinuity_declarations = _function_declarations(
        _COOP_ROOT / backend / "__init__.pyi",
        "discontinuity",
    )
    scalar_declarations = [
        declaration
        for declaration in discontinuity_declarations
        if ast.unparse(declaration.args.posonlyargs[1].annotation) == scalar_annotation
    ]
    assert len(scalar_declarations) == 3
    assert tuple(
        ast.unparse(declaration.args.kwonlyargs[0].annotation)
        for declaration in scalar_declarations
    ) == (
        "Literal['heads']",
        "Literal['tails']",
        "Literal['heads_and_tails']",
    )
    assert tuple(
        ast.unparse(declaration.returns) for declaration in scalar_declarations
    ) == ("int", "int", "tuple[int, int]")


@pytest.mark.parametrize("name", ["adjacent_difference", "discontinuity"])
@pytest.mark.parametrize("backend", [None, "cutlass", "numba_mlir"])
def test_segmentation_stub_docstrings_describe_every_parameter(
    backend: str | None,
    name: str,
) -> None:
    path = (
        _COOP_ROOT / "__init__.pyi"
        if backend is None
        else _COOP_ROOT / backend / "__init__.pyi"
    )
    declarations = _function_declarations(path, name)
    assert declarations

    for declaration in declarations:
        docstring = ast.get_docstring(declaration) or ""
        assert docstring
        parameters = (
            *declaration.args.posonlyargs,
            *declaration.args.kwonlyargs,
        )
        for parameter in parameters:
            assert f"``{parameter.arg}``" in docstring


@pytest.mark.parametrize("name", ["topk_max_pairs", "topk_min_pairs"])
def test_cutlass_pair_topk_stubs_are_explicit_and_payload_preserving(name: str) -> None:
    declarations = _function_declarations(
        _COOP_ROOT / "cutlass" / "__init__.pyi",
        name,
    )
    expected = (
        ("positional_only", "group", _REQUIRED),
        ("positional_only", "keys", _REQUIRED),
        ("positional_only", "values", _REQUIRED),
        ("positional_only", "k", _REQUIRED),
        ("keyword_only", "valid_items", "None"),
        ("keyword_only", "begin_bit", "0"),
        ("keyword_only", "end_bit", "None"),
        ("keyword_only", "temp_storage", "None"),
    )

    assert len(declarations) == 3
    assert all(
        _parameter_layout(declaration) == expected for declaration in declarations
    )
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[1].annotation)
        for declaration in declarations
    ) == (
        "ThreadData[_CutlassTopKKeyT]",
        "_CutlassTopKKeyT",
        "_CutlassTensorSample | _CutlassTensorSSASample",
    )
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[2].annotation)
        for declaration in declarations
    ) == (
        "ThreadData[_CutlassTopKValueT]",
        "_CutlassTopKValueT",
        "_CutlassTensorSample | _CutlassTensorSSASample",
    )
    assert tuple(ast.unparse(declaration.returns) for declaration in declarations) == (
        ("tuple[ThreadData[_CutlassTopKKeyT], ThreadData[_CutlassTopKValueT]]"),
        "tuple[_CutlassTopKKeyT, _CutlassTopKValueT]",
        "tuple[ThreadData[Any], ThreadData[Any]]",
    )
    for declaration in declarations:
        assert ast.unparse(declaration.args.posonlyargs[3].annotation) == (
            "_IntegerValue"
        )
        controls = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        assert controls["valid_items"] == "_ValidItems | None"
        assert controls["begin_bit"] == "_IntegerValue"
        assert controls["end_bit"] == "_IntegerValue | None"
        assert controls["temp_storage"] == "TempStorage | None"


@pytest.mark.parametrize("name", ["topk_max_pairs", "topk_min_pairs"])
def test_numba_pair_topk_stubs_are_explicit_and_payload_preserving(name: str) -> None:
    declarations = _function_declarations(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        name,
    )
    expected = (
        ("positional_only", "group", _REQUIRED),
        ("positional_only", "keys", _REQUIRED),
        ("positional_only", "values", _REQUIRED),
        ("positional_only", "k", _REQUIRED),
        ("keyword_only", "valid_items", "None"),
        ("keyword_only", "begin_bit", "0"),
        ("keyword_only", "end_bit", "None"),
        ("keyword_only", "temp_storage", "None"),
    )

    assert len(declarations) == 1
    assert _parameter_layout(declarations[0]) == expected
    result = ast.unparse(declarations[0].returns)
    assert ast.unparse(declarations[0].args.posonlyargs[1].annotation) == (
        "_ThreadDataLike[_NumbaTopKKeyT]"
    )
    assert ast.unparse(declarations[0].args.posonlyargs[2].annotation) == (
        "_ThreadDataLike[_NumbaTopKValueT]"
    )
    assert ast.unparse(declarations[0].args.posonlyargs[3].annotation) == (
        "_IntegerValue"
    )
    assert result == (
        "tuple[_ThreadDataLike[_NumbaTopKKeyT], _ThreadDataLike[_NumbaTopKValueT]]"
    )
    controls = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in declarations[0].args.kwonlyargs
    }
    assert controls["valid_items"] == "_ValidItems | None"
    assert controls["begin_bit"] == "_IntegerValue"
    assert controls["end_bit"] == "_IntegerValue | None"
    assert controls["temp_storage"] == "TempStorage | None"


def test_radix_rank_stubs_encode_integer_inputs_and_int32_result_shapes() -> None:
    common = _function_declarations(_COOP_ROOT / "__init__.pyi", "radix_rank")
    cutlass = _function_declarations(
        _COOP_ROOT / "cutlass" / "__init__.pyi",
        "radix_rank",
    )
    numba = _function_declarations(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        "radix_rank",
    )

    assert tuple(
        ast.unparse(item.args.posonlyargs[1].annotation) for item in common
    ) == ("ThreadDataLike[_IntegerKeyT]",)
    assert tuple(ast.unparse(item.returns) for item in common) == (
        "ThreadDataLike[int]",
    )
    assert tuple(
        ast.unparse(item.args.posonlyargs[1].annotation) for item in cutlass
    ) == (
        "ThreadData[_IntegerKeyT]",
        "_IntegerKeyT",
        "_CutlassTensorSample | _CutlassTensorSSASample",
    )
    assert tuple(ast.unparse(item.returns) for item in cutlass) == (
        "ThreadData[int]",
        "int",
        "ThreadData[int]",
    )
    assert tuple(
        ast.unparse(item.args.posonlyargs[1].annotation) for item in numba
    ) == (
        "_ThreadDataLike[_IntegerKeyT]",
        "_IntegerKeyT",
    )
    assert tuple(ast.unparse(item.returns) for item in numba) == (
        "_ThreadDataLike[int]",
        "int",
    )

    for declaration in (*common, *cutlass, *numba):
        controls = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        assert controls["begin_bit"] == "_TraceInteger"
        assert controls["end_bit"] == "_TraceInteger | None"
        assert controls["radix_bits"] == "_TraceInteger | None"


def test_run_length_decode_stubs_share_dtype_controls_and_preserve_values() -> None:
    common = _function_declarations(_COOP_ROOT / "__init__.pyi", "run_length_decode")
    cutlass = _function_declarations(
        _COOP_ROOT / "cutlass" / "__init__.pyi",
        "run_length_decode",
    )
    numba = _function_declarations(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        "run_length_decode",
    )

    assert len(common) == 1
    assert len(cutlass) == 3
    assert len(numba) == 2
    assert ast.unparse(common[0].args.posonlyargs[1].annotation) == (
        "ThreadDataLike[_RunValueT]"
    )
    assert ast.unparse(common[0].args.posonlyargs[2].annotation) == (
        "ThreadDataLike[_RunLengthT]"
    )
    assert ast.unparse(common[0].returns) == "ThreadDataLike[_RunValueT]"
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[1].annotation)
        for declaration in cutlass
    ) == (
        "ThreadData[_RunValueT]",
        "_CutlassRunTensor",
        "_RunValueT",
    )
    assert tuple(ast.unparse(declaration.returns) for declaration in cutlass) == (
        "ThreadData[_RunValueT]",
        "ThreadData[Any]",
        "ThreadData[_RunValueT]",
    )
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[1].annotation) for declaration in numba
    ) == (
        "_ThreadDataLike[_RunValueT]",
        "_ThreadDataLike[_NumbaRunValueT]",
    )
    assert tuple(
        ast.unparse(declaration.args.posonlyargs[2].annotation) for declaration in numba
    ) == (
        "_ThreadDataLike[_RunLengthT]",
        "_ThreadDataLike[_NumbaRunLengthT]",
    )
    assert tuple(ast.unparse(declaration.returns) for declaration in numba) == (
        "_ThreadDataLike[_RunValueT]",
        "_ThreadDataLike[_NumbaRunValueT]",
    )

    for declaration in (*common, *cutlass, *numba):
        controls = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in declaration.args.kwonlyargs
        }
        assert controls["decoded_items_per_thread"] == "_TraceInteger"
        assert controls["decoded_window_offset"] == "_IntegerValue"


def test_run_length_decode_dtype_aliases_have_one_authoritative_definition() -> None:
    typing_source = (_COOP_ROOT / "_typing.pyi").read_text(encoding="utf-8")
    stub_sources = [
        _stub_surface_source(path)
        for path in (
            _COOP_ROOT / "__init__.pyi",
            _COOP_ROOT / "cutlass" / "__init__.pyi",
            _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        )
    ]

    assert (
        "_PortableIntegerValue: TypeAlias = _PortableIntegerKey | _NumpyUint8"
        in typing_source
    )
    assert "_PortableRunValue: TypeAlias = _PortableIntegerValue" in typing_source
    assert "_PortableRunLength: TypeAlias = _PortableIntegerKey" in typing_source
    assert all("_PortableRunValue" in source for source in stub_sources)
    assert all("_PortableRunLength" in source for source in stub_sources)
    assert all(
        "_PortableIntegerKey | _NumpyUint8" not in source for source in stub_sources
    )


def test_common_and_cutlass_histogram_stubs_close_integer_dtype_families() -> None:
    common_path = _COOP_ROOT / "__init__.pyi"
    cutlass_path = _COOP_ROOT / "cutlass" / "__init__.pyi"

    assert _typevar_bound(common_path, "_CounterT") == "_PortableIntegerKey"
    assert _typevar_bound(common_path, "_HistogramSampleT") == ("_PortableIntegerValue")
    assert _typevar_bound(cutlass_path, "_CounterT") == "_PortableIntegerKey"
    assert _typevar_bound(cutlass_path, "_HistogramSampleT") == (
        "_PortableIntegerValue"
    )

    expected_samples = {
        common_path: "ThreadDataLike[_HistogramSampleT]",
        cutlass_path: (
            "ThreadData[_HistogramSampleT] | _HistogramSampleT | "
            "_CutlassHistogramOpaqueSamples"
        ),
    }
    expected_returns = {
        common_path: ("ThreadDataLike[_CounterT]", "ThreadDataLike[int]"),
        cutlass_path: ("ThreadData[_CounterT]", "ThreadData[int]"),
    }
    for path in (common_path, cutlass_path):
        declarations = _function_declarations(path, "histogram")
        assert len(declarations) == 2
        assert tuple(
            ast.unparse(declaration.args.posonlyargs[1].annotation)
            for declaration in declarations
        ) == (expected_samples[path], expected_samples[path])
        assert tuple(
            ast.unparse(
                next(
                    argument.annotation
                    for argument in declaration.args.kwonlyargs
                    if argument.arg == "counter_dtype"
                )
            )
            for declaration in declarations
        ) == ("type[_CounterT]", "None")
        assert (
            tuple(ast.unparse(declaration.returns) for declaration in declarations)
            == (expected_returns[path])
        )


def test_portable_numeric_scalar_alias_is_closed_and_authoritative() -> None:
    typing_path = _COOP_ROOT / "_typing.pyi"
    common_path = _COOP_ROOT / "__init__.pyi"
    cutlass_path = _COOP_ROOT / "cutlass" / "__init__.pyi"
    numba_path = _COOP_ROOT / "numba_mlir" / "__init__.pyi"

    assert _union_terms(_named_assignment(typing_path, "_PortableNumericScalar")) == (
        "int",
        "float",
        "_NumpyUint8",
        "_NumpyInt32",
        "_NumpyUint32",
        "_NumpyInt64",
        "_NumpyUint64",
        "_NumpyFloat32",
        "_NumpyFloat64",
        "_CompilerScalarLike",
    )
    assert _typevar_bound(common_path, "_ItemT") == "_PortableNumericScalar"
    assert _typevar_bound(common_path, "_PortableNumericT") == "_PortableNumericScalar"
    assert _typevar_bound(common_path, "_ScalarT") == "_PortableNumericScalar"
    assert _typevar_bound(common_path, "_ScalarValueT") == "_PortableNumericScalar"
    for name in ("_ScalarT", "_ScalarValueT", "_CutlassNumericT"):
        assert _typevar_bound(cutlass_path, name) == "_PortableNumericScalar"
    assert _typevar_bound(cutlass_path, "_DtypeT") == "_OrdinaryNumericScalar"
    for name in ("_ItemT", "_SourceT_co", "_ValueT"):
        declaration = _named_assignment(cutlass_path, name)
        assert isinstance(declaration, ast.Call)
        assert not any(keyword.arg == "bound" for keyword in declaration.keywords)
    cutlass_source = _stub_surface_source(cutlass_path)
    assert "_SourceValueT" not in cutlass_source
    assert (
        ast.unparse(_named_assignment(cutlass_path, "_CutlassOrderedItem"))
        == "_PortableNumericScalar"
    )
    assert "_CutlassNumericSample" not in cutlass_source

    # Numba's qualified surface deliberately remains broader than the common
    # intersection and includes Python complex plus all NumPy scalar classes.
    assert _typevar_bound(numba_path, "_ScalarT") == (
        "bool | int | float | complex | _NumpyScalar"
    )
    assert _typevar_bound(numba_path, "_ScalarValueT") == "_ScalarValue"


def test_thread_data_dtype_overloads_match_common_and_qualified_domains() -> None:
    common_path = _COOP_ROOT / "__init__.pyi"
    common = _function_declarations(common_path, "ThreadData")
    assert tuple(
        ast.unparse(declaration.args.args[1].annotation) for declaration in common
    ) == ("type[_PortableNumericT]", "None")
    assert tuple(ast.unparse(declaration.returns) for declaration in common) == (
        "ThreadDataLike[_PortableNumericT]",
        "ThreadDataLike[Any]",
    )

    cutlass_path = _COOP_ROOT / "cutlass" / "__init__.pyi"
    constructors = _method_declarations(cutlass_path, "ThreadData", "__init__")
    assert tuple(
        ast.unparse(declaration.args.args[2].annotation) for declaration in constructors
    ) == ("type[_ItemT]", "None")

    from_values = _method_declarations(cutlass_path, "ThreadData", "from_values")[0]
    assert ast.unparse(from_values.args.args[1].annotation) == "_ValueT"
    assert from_values.args.vararg is not None
    assert ast.unparse(from_values.args.vararg.annotation) == "_ValueT"
    assert ast.unparse(from_values.args.kwonlyargs[0].annotation) == "type[Any] | None"
    assert ast.unparse(from_values.returns) == "ThreadData[_ValueT]"
    assert "without casting" in (ast.get_docstring(from_values) or "")

    from_fn = _method_declarations(cutlass_path, "ThreadData", "from_fn")
    assert tuple(
        ast.unparse(declaration.args.args[2].annotation) for declaration in from_fn
    ) == (
        "Callable[[int], object]",
        "Callable[[int], _ValueT]",
        "Callable[[int], object]",
    )
    assert tuple(
        ast.unparse(declaration.args.kwonlyargs[0].annotation)
        for declaration in from_fn
    ) == (
        "type[_DtypeT]",
        "None",
        "object",
    )
    assert tuple(ast.unparse(declaration.returns) for declaration in from_fn) == (
        "ThreadData[_DtypeT]",
        "ThreadData[_ValueT]",
        "ThreadData[Any]",
    )

    expected_helper_routes = {
        "from_register_tensor": (
            (
                "_ThreadDataSizedRegisterTensorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataRegisterTensorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int",
                _REQUIRED,
            ),
            (
                "_ThreadDataSizedRegisterTensorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataRegisterTensorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int",
                _REQUIRED,
            ),
            (
                "_ThreadDataSizedRegisterTensorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataRegisterTensorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int",
                _REQUIRED,
            ),
        ),
        "from_vector": (
            (
                "_ThreadDataStaticallySizedVectorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int",
                _REQUIRED,
            ),
            (
                "_ThreadDataStaticallySizedVectorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int",
                _REQUIRED,
            ),
            (
                "_ThreadDataStaticallySizedVectorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int",
                _REQUIRED,
            ),
        ),
        "from_payload": (
            (
                "ThreadData[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataStaticallySizedVectorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int",
                _REQUIRED,
            ),
            (
                "ThreadData[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataStaticallySizedVectorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int",
                _REQUIRED,
            ),
            (
                "ThreadData[Any]",
                "object",
                "ThreadData[Any]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataStaticallySizedVectorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataVectorSource[Any]",
                "object",
                "ThreadData[Any]",
                "int",
                _REQUIRED,
            ),
        ),
        "load": (
            (
                "ThreadDataLoadSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataIndexableLoadSource[Any]",
                "type[_DtypeT]",
                "ThreadData[_DtypeT]",
                "int",
                _REQUIRED,
            ),
            (
                "ThreadDataLoadSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataIndexableLoadSource[_ValueT]",
                "None",
                "ThreadData[_ValueT]",
                "int",
                _REQUIRED,
            ),
            (
                "ThreadDataLoadSource[Any]",
                "object",
                "ThreadData[Any]",
                "int | None",
                "None",
            ),
            (
                "_ThreadDataIndexableLoadSource[Any]",
                "object",
                "ThreadData[Any]",
                "int",
                _REQUIRED,
            ),
        ),
    }
    for name, expected_routes in expected_helper_routes.items():
        declarations = _method_declarations(cutlass_path, "ThreadData", name)
        actual_routes = []
        for declaration in declarations:
            keyword_annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in declaration.args.kwonlyargs
            }
            keyword_defaults = {
                argument.arg: _default_source(default)
                for argument, default in zip(
                    declaration.args.kwonlyargs,
                    declaration.args.kw_defaults,
                )
            }
            actual_routes.append(
                (
                    ast.unparse(declaration.args.args[1].annotation),
                    keyword_annotations["dtype"],
                    ast.unparse(declaration.returns),
                    keyword_annotations["items_per_thread"],
                    keyword_defaults["items_per_thread"],
                )
            )
        assert tuple(actual_routes) == expected_routes

    numba = _function_declarations(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        "ThreadData",
    )
    assert tuple(
        ast.unparse(declaration.args.args[1].annotation) for declaration in numba
    ) == ("type[_ItemT]", "object")


def test_integer_key_alias_is_collective_neutral_without_compatibility_spelling() -> (
    None
):
    typing_source = (_COOP_ROOT / "_typing.pyi").read_text(encoding="utf-8")
    stub_sources = [
        _stub_surface_source(path)
        for path in (
            _COOP_ROOT / "__init__.pyi",
            _COOP_ROOT / "cutlass" / "__init__.pyi",
            _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        )
    ]

    assert "_PortableIntegerKey: TypeAlias" in typing_source
    assert all("_PortableIntegerKey" in source for source in stub_sources)
    assert "_MergeSortKey" not in typing_source
    assert all("_MergeSortKey" not in source for source in stub_sources)


def test_common_modes_use_literal_static_contracts() -> None:
    path = _COOP_ROOT / "_typing.pyi"
    aliases = {
        statement.target.id: {
            element.value
            for element in statement.value.slice.elts
            if isinstance(element, ast.Constant)
        }
        for statement in _module_tree(path).body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.annotation is not None
        and ast.unparse(statement.annotation) == "TypeAlias"
        and isinstance(statement.value, ast.Subscript)
        and isinstance(statement.value.value, ast.Name)
        and statement.value.value.id == "Literal"
        and isinstance(statement.value.slice, ast.Tuple)
    }

    assert aliases["ThreadLevel"] == {
        "thread",
        "gpu_thread",
        "warp",
        "block",
        "cluster",
        "grid",
    }
    block_load_store_algorithms = {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
    warp_load_store_algorithms = {
        "direct",
        "striped",
        "vectorize",
        "transpose",
    }
    assert aliases["_BlockLoadStoreAlgorithm"] == block_load_store_algorithms
    assert aliases["_WarpLoadStoreAlgorithm"] == warp_load_store_algorithms
    assert ast.unparse(_named_assignment(path, "LoadStoreAlgorithm")) == (
        "_BlockLoadStoreAlgorithm"
    )
    assert aliases["ReduceAlgorithm"] == {
        "raking_commutative_only",
        "raking",
        "warp_reductions",
    }
    assert aliases["ReduceOperator"] == {
        "+",
        "sum",
        "add",
        "plus",
        "*",
        "mul",
        "multiply",
        "multiplies",
        "min",
        "minimum",
        "max",
        "maximum",
        "&",
        "bit_and",
        "|",
        "bit_or",
        "^",
        "bit_xor",
    }
    assert aliases["ExchangeMode"] == {
        "striped_to_blocked",
        "blocked_to_striped",
    }
    assert aliases["PortableShuffleMode"] == {"down", "up"}
    assert aliases["ShuffleMode"] == {"down", "up", "offset", "rotate"}
    assert aliases["TempStorageSharing"] == {"shared", "exclusive"}


def test_load_store_stubs_correlate_group_and_algorithm_selectors() -> None:
    expected_annotations = {
        None: {
            "load": (
                "_BlockLoadStoreAlgorithm",
                "_BlockLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
            ),
            "store": (
                "_BlockLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
            ),
        },
        "cutlass": {
            "load": (
                "_BlockLoadStoreAlgorithm",
                "_BlockLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
            ),
            "store": (
                "_BlockLoadStoreAlgorithm",
                "_WarpLoadStoreAlgorithm",
            ),
        },
        "numba_mlir": {
            "load": (
                "_BlockLoadStoreAlgorithm | BlockLoadAlgorithm",
                "_BlockLoadStoreAlgorithm | BlockLoadAlgorithm",
                "_WarpLoadStoreAlgorithm | WarpLoadAlgorithm",
                "_WarpLoadStoreAlgorithm | WarpLoadAlgorithm",
            ),
            "store": (
                "_BlockLoadStoreAlgorithm | BlockStoreAlgorithm",
                "_WarpLoadStoreAlgorithm | WarpStoreAlgorithm",
            ),
        },
    }

    for backend, operations in expected_annotations.items():
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        for name, expected_algorithms in operations.items():
            declarations = _function_declarations(path, name)
            expected_groups = (
                ("_BlockGroup", "_BlockGroup", "_WarpGroup", "_WarpGroup")
                if name == "load"
                else ("_BlockGroup", "_WarpGroup")
            )
            assert len(declarations) == len(expected_groups)
            assert (
                tuple(
                    ast.unparse(declaration.args.posonlyargs[0].annotation)
                    for declaration in declarations
                )
                == expected_groups
            )
            assert (
                tuple(
                    ast.unparse(
                        next(
                            argument.annotation
                            for argument in declaration.args.kwonlyargs
                            if argument.arg == "algorithm"
                        )
                    )
                    for declaration in declarations
                )
                == expected_algorithms
            )
            assert tuple(
                _default_source(
                    declaration.args.kw_defaults[
                        next(
                            index
                            for index, argument in enumerate(
                                declaration.args.kwonlyargs
                            )
                            if argument.arg == "algorithm"
                        )
                    ]
                )
                for declaration in declarations
            ) == tuple("'direct'" for _ in declarations)


def test_load_store_stubs_type_partial_tile_controls_consistently() -> None:
    fill_types = {
        None: "_PortableNumericT",
        "cutlass": "_CutlassNumericT",
        "numba_mlir": "_ItemT",
    }

    for backend, fill_type in fill_types.items():
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        loads = _function_declarations(path, "load")
        assert len(loads) == 4
        for index, declaration in enumerate(loads):
            keyword_arguments = {
                argument.arg: (argument, declaration.args.kw_defaults[position])
                for position, argument in enumerate(declaration.args.kwonlyargs)
            }
            valid_argument, valid_default = keyword_arguments["valid_items"]
            fill_argument, fill_default = keyword_arguments["oob_default"]
            offset_argument, offset_default = keyword_arguments["offset"]
            assert ast.unparse(offset_argument.annotation) == "_IntegerValue | None"
            assert _default_source(offset_default) == "None"

            if index in {0, 2}:
                assert ast.unparse(valid_argument.annotation) == "_ValidItems | None"
                assert _default_source(valid_default) == "None"
                assert ast.unparse(fill_argument.annotation) == "None"
                assert _default_source(fill_default) == "None"
            else:
                assert ast.unparse(valid_argument.annotation) == "_ValidItems"
                assert valid_default is None
                assert ast.unparse(fill_argument.annotation) == fill_type
                assert fill_default is None

        stores = _function_declarations(path, "store")
        assert len(stores) == 2
        for declaration in stores:
            keyword_arguments = {
                argument.arg: (argument, declaration.args.kw_defaults[position])
                for position, argument in enumerate(declaration.args.kwonlyargs)
            }
            valid_argument, valid_default = keyword_arguments["valid_items"]
            offset_argument, offset_default = keyword_arguments["offset"]
            assert ast.unparse(valid_argument.annotation) == "_ValidItems | None"
            assert _default_source(valid_default) == "None"
            assert ast.unparse(offset_argument.annotation) == "_IntegerValue | None"
            assert _default_source(offset_default) == "None"


def test_adjacent_difference_stubs_correlate_direction_and_boundaries() -> None:
    payload_cohorts = {
        None: (("ThreadDataLike[_PortableNumericT]", "_PortableNumericT"),),
        "cutlass": (
            ("ThreadData[_CutlassNumericT]", "_CutlassNumericT"),
            (
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "_PortableNumericScalar",
            ),
            ("_ScalarValueT", "_ScalarValueT"),
        ),
        "numba_mlir": (
            ("_ThreadDataLike[_ItemT]", "_ItemT"),
            ("_ScalarT", "_ScalarT"),
        ),
    }

    for backend, expected_cohorts in payload_cohorts.items():
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, "adjacent_difference")
        assert len(declarations) == 3 * len(expected_cohorts)

        for cohort_index, (payload_type, boundary_type) in enumerate(expected_cohorts):
            cohort = declarations[cohort_index * 3 : cohort_index * 3 + 3]
            assert tuple(
                ast.unparse(declaration.args.posonlyargs[1].annotation)
                for declaration in cohort
            ) == (payload_type, payload_type, payload_type)

            annotations = [
                {
                    argument.arg: ast.unparse(argument.annotation)
                    for argument in declaration.args.kwonlyargs
                }
                for declaration in cohort
            ]
            assert tuple(item["direction"] for item in annotations) == (
                "Literal['left']",
                "Literal['right']",
                "Literal['right']",
            )
            assert tuple(item["valid_items"] for item in annotations) == (
                "_ValidItems | None",
                "_ValidItems | None",
                "None",
            )
            assert tuple(item["tile_predecessor_item"] for item in annotations) == (
                f"{boundary_type} | None",
                "None",
                "None",
            )
            assert tuple(item["tile_successor_item"] for item in annotations) == (
                "None",
                "None",
                boundary_type,
            )

            layouts = tuple(_parameter_layout(declaration) for declaration in cohort)
            direction_index = next(
                index
                for index, parameter in enumerate(layouts[0])
                if parameter[1] == "direction"
            )
            successor_index = next(
                index
                for index, parameter in enumerate(layouts[0])
                if parameter[1] == "tile_successor_item"
            )
            assert tuple(layout[direction_index][2] for layout in layouts) == (
                "'left'",
                _REQUIRED,
                _REQUIRED,
            )
            assert tuple(layout[successor_index][2] for layout in layouts) == (
                "None",
                "None",
                _REQUIRED,
            )


def test_discontinuity_stubs_correlate_mode_and_boundaries() -> None:
    payload_cohorts = {
        None: (("ThreadDataLike[_PortableNumericT]", "_PortableNumericT"),),
        "cutlass": (
            ("ThreadData[_CutlassNumericT]", "_CutlassNumericT"),
            (
                "_CutlassTensorSample | _CutlassTensorSSASample",
                "_PortableNumericScalar",
            ),
            ("_ScalarValueT", "_ScalarValueT"),
        ),
        "numba_mlir": (
            ("_ThreadDataLike[_ItemT]", "_ItemT"),
            ("_ScalarT", "_ScalarT"),
        ),
    }

    for backend, expected_cohorts in payload_cohorts.items():
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, "discontinuity")
        modes_per_cohort = 2 if backend is None else 3
        assert len(declarations) == modes_per_cohort * len(expected_cohorts)

        for cohort_index, (payload_type, boundary_type) in enumerate(expected_cohorts):
            start = cohort_index * modes_per_cohort
            cohort = declarations[start : start + modes_per_cohort]
            assert (
                tuple(
                    ast.unparse(declaration.args.posonlyargs[1].annotation)
                    for declaration in cohort
                )
                == (payload_type,) * modes_per_cohort
            )

            annotations = [
                {
                    argument.arg: ast.unparse(argument.annotation)
                    for argument in declaration.args.kwonlyargs
                }
                for declaration in cohort
            ]
            expected_modes = ["Literal['heads']", "Literal['tails']"]
            expected_predecessors = [f"{boundary_type} | None", "None"]
            expected_successors = ["None", f"{boundary_type} | None"]
            if backend is not None:
                expected_modes.append("Literal['heads_and_tails']")
                expected_predecessors.append(f"{boundary_type} | None")
                expected_successors.append(f"{boundary_type} | None")
            assert tuple(item["mode"] for item in annotations) == tuple(expected_modes)
            assert tuple(
                item["tile_predecessor_item"] for item in annotations
            ) == tuple(expected_predecessors)
            assert tuple(item["tile_successor_item"] for item in annotations) == tuple(
                expected_successors
            )

            layouts = tuple(_parameter_layout(declaration) for declaration in cohort)
            mode_index = next(
                index
                for index, parameter in enumerate(layouts[0])
                if parameter[1] == "mode"
            )
            assert tuple(layout[mode_index][2] for layout in layouts) == (
                "'heads'",
                *(_REQUIRED for _ in range(modes_per_cohort - 1)),
            )


def test_histogram_stub_docs_state_the_input_range_precondition() -> None:
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, "histogram")
        assert declarations
        for declaration in declarations:
            docstring = ast.get_docstring(declaration)
            assert docstring is not None
            assert "0 <= sample < bins" in docstring
            assert "undefined behavior" in docstring


def test_numba_stateful_function_stub_matches_the_runtime_value() -> None:
    declaration = _top_level_declaration(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        "StatefulFunction",
    )
    assert isinstance(declaration, ast.ClassDef)
    assert tuple(ast.unparse(base) for base in declaration.bases) == ("Generic[_OpT]",)

    attributes = {
        statement.target.id: ast.unparse(statement.annotation)
        for statement in declaration.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }
    assert attributes == {
        "op": "_OpT",
        "dtype": "object",
        "name": "str | None",
    }
    initializer = next(
        statement
        for statement in declaration.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "__init__"
    )
    assert _parameter_layout(initializer, drop_self=True) == (
        ("positional", "op", _REQUIRED),
        ("positional", "dtype", _REQUIRED),
        ("positional", "name", "None"),
    )


def test_shuffle_stubs_separate_portable_payload_and_qualified_scalar_forms() -> None:
    common_declarations = _function_declarations(
        _COOP_ROOT / "__init__.pyi",
        "shuffle",
    )
    assert len(common_declarations) == 1
    common = common_declarations[0]
    assert ast.unparse(common.args.posonlyargs[1].annotation) == (
        "ThreadDataLike[_PortableNumericT]"
    )
    assert ast.unparse(common.returns) == "ThreadDataLike[_PortableNumericT]"
    common_annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in common.args.kwonlyargs
    }
    assert common_annotations == {
        "mode": "_PortableShuffleMode",
        "distance": "Literal[1]",
    }

    for backend, payload_annotation, payload_return, expected_count in (
        (
            "cutlass",
            "ThreadData[_CutlassNumericT]",
            "ThreadData[_CutlassNumericT]",
            5,
        ),
        ("numba_mlir", "_PayloadT", "_PayloadT", 2),
    ):
        declarations = _function_declarations(
            _COOP_ROOT / backend / "__init__.pyi",
            "shuffle",
        )
        assert len(declarations) == expected_count
        payload = declarations[0]
        scalar = declarations[-1]
        assert all(
            any(
                isinstance(decorator, ast.Name) and decorator.id == "overload"
                for decorator in declaration.decorator_list
            )
            for declaration in declarations
        )
        assert ast.unparse(payload.args.posonlyargs[1].annotation) == (
            payload_annotation
        )
        assert ast.unparse(payload.returns) == payload_return
        payload_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in payload.args.kwonlyargs
        }
        assert payload_annotations["mode"] == (
            "Literal['down']" if backend == "cutlass" else "_PortableShuffleMode"
        )
        assert payload_annotations["distance"] == "Literal[1]"

        if backend == "cutlass":
            up = declarations[1]
            up_annotations = {
                argument.arg: ast.unparse(argument.annotation)
                for argument in up.args.kwonlyargs
            }
            assert ast.unparse(up.args.posonlyargs[1].annotation) == (
                "ThreadData[_CutlassNumericT]"
            )
            assert ast.unparse(up.returns) == "ThreadData[_CutlassNumericT]"
            assert up_annotations["mode"] == "Literal['up']"
            assert up_annotations["distance"] == "Literal[1]"
            assert up_annotations["block_prefix"] == "None"
            assert (
                up_annotations["block_suffix"] == "ThreadData[_CutlassNumericT] | None"
            )
            tensor_down, tensor_up = declarations[2:4]
            assert all(
                ast.unparse(declaration.args.posonlyargs[1].annotation)
                == "_CutlassTensorSample | _CutlassTensorSSASample"
                for declaration in (tensor_down, tensor_up)
            )
            assert all(
                ast.unparse(declaration.returns) == "ThreadData[Any]"
                for declaration in (tensor_down, tensor_up)
            )

        assert ast.unparse(scalar.args.posonlyargs[1].annotation) == "_ScalarT"
        assert ast.unparse(scalar.returns) == "_ScalarT"
        scalar_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in scalar.args.kwonlyargs
        }
        assert scalar_annotations["mode"] == (
            "Literal['offset', 'rotate']" if backend == "cutlass" else "_ShuffleMode"
        )
        assert scalar_annotations["distance"] == "int"
        assert scalar_annotations["block_prefix"] == "None"
        assert scalar_annotations["block_suffix"] == "None"
        scalar_defaults = {
            argument.arg: _default_source(default)
            for argument, default in zip(
                scalar.args.kwonlyargs,
                scalar.args.kw_defaults,
            )
        }
        assert scalar_defaults["mode"] == (
            _REQUIRED if backend == "cutlass" else "'down'"
        )

        for declaration in declarations:
            docstring = " ".join((ast.get_docstring(declaration) or "").split())
            for parameter in (
                "group",
                "value",
                "mode",
                "distance",
                "block_prefix",
                "block_suffix",
            ):
                assert f"``{parameter}``" in docstring


def test_exchange_stubs_preserve_payload_type_and_document_portable_boundary() -> None:
    for backend in (None, "cutlass", "numba_mlir"):
        path = (
            _COOP_ROOT / "__init__.pyi"
            if backend is None
            else _COOP_ROOT / backend / "__init__.pyi"
        )
        declarations = _function_declarations(path, "exchange")

        assert len(declarations) == {None: 1, "cutlass": 4, "numba_mlir": 2}[backend]
        declaration = declarations[0]
        expected = {
            None: (
                "ThreadDataLike[_PortableNumericT]",
                "ThreadDataLike[_PortableNumericT]",
            ),
            "cutlass": (
                "ThreadData[_CutlassNumericT]",
                "ThreadData[_CutlassNumericT]",
            ),
            "numba_mlir": ("_PayloadT", "_PayloadT"),
        }[backend]
        assert ast.unparse(declaration.args.posonlyargs[1].annotation) == expected[0]
        assert ast.unparse(declaration.returns) == expected[1]

        docstring = " ".join((ast.get_docstring(declaration) or "").split())
        assert "ThreadData" in docstring
        if backend == "numba_mlir":
            assert "portability is guaranteed for one through five" in docstring
            assert "larger fixed-size payloads" in docstring
        else:
            assert "one through five" in docstring
        assert "scalar inputs are not supported" in docstring
        assert "without mutation" in docstring
        assert "striped_to_blocked" in docstring
        assert "blocked_to_striped" in docstring

        if backend == "cutlass":
            register_declarations = declarations[2:]
            assert all(
                ast.unparse(declaration.args.posonlyargs[1].annotation)
                == "_CutlassTensorSample | _CutlassTensorSSASample"
                for declaration in register_declarations
            )
            assert all(
                ast.unparse(declaration.returns) == "ThreadData[Any]"
                for declaration in register_declarations
            )


def test_cutlass_register_collective_overloads_are_explicit_and_documented() -> None:
    expected_returns = {
        "scan": ("ThreadData[Any]",) * 3,
        "exclusive_sum": ("ThreadData[Any]",),
        "inclusive_sum": ("ThreadData[Any]",),
        "exclusive_scan": ("ThreadData[Any]",) * 2,
        "inclusive_scan": ("ThreadData[Any]",),
        "exchange": ("ThreadData[Any]",) * 2,
        "adjacent_difference": ("ThreadData[Any]",) * 3,
        "discontinuity": (
            "ThreadData[int]",
            "ThreadData[int]",
            "tuple[ThreadData[int], ThreadData[int]]",
        ),
        "shuffle": ("ThreadData[Any]", "ThreadData[Any]"),
    }

    for name, returns in expected_returns.items():
        declarations = _function_declarations(
            _COOP_ROOT / "cutlass" / "__init__.pyi",
            name,
        )
        register_declarations = [
            declaration
            for declaration in declarations
            if ast.unparse(declaration.args.posonlyargs[1].annotation)
            == "_CutlassTensorSample | _CutlassTensorSSASample"
        ]
        assert (
            tuple(
                ast.unparse(declaration.returns)
                for declaration in register_declarations
            )
            == returns
        )
        for declaration in register_declarations:
            docstring = ast.get_docstring(declaration) or ""
            assert docstring
            for parameter in (
                *declaration.args.posonlyargs,
                *declaration.args.kwonlyargs,
            ):
                assert f"``{parameter.arg}``" in docstring


def test_cutlass_thread_data_is_the_concrete_helper_class() -> None:
    declaration = _top_level_declaration(
        _COOP_ROOT / "cutlass" / "__init__.pyi",
        "ThreadData",
    )

    assert isinstance(declaration, ast.ClassDef)
    assert "Generic[_ItemT]" in {ast.unparse(base) for base in declaration.bases}
    assert _CUTLASS_THREAD_DATA_HELPERS <= _class_methods(
        _COOP_ROOT / "cutlass" / "__init__.pyi", "ThreadData"
    )


def test_numba_mlir_thread_data_is_a_distinct_factory() -> None:
    common_declaration = _top_level_declaration(
        _COOP_ROOT / "__init__.pyi",
        "ThreadData",
    )
    numba_declaration = _top_level_declaration(
        _COOP_ROOT / "numba_mlir" / "__init__.pyi",
        "ThreadData",
    )

    assert isinstance(common_declaration, ast.FunctionDef)
    assert isinstance(numba_declaration, ast.FunctionDef)
    assert {argument.arg for argument in numba_declaration.args.kwonlyargs} == {
        "alignas"
    }
    assert {argument.arg for argument in common_declaration.args.kwonlyargs}.isdisjoint(
        {"values", "alignas"}
    )


def test_numba_mlir_thread_data_overloads_use_one_canonical_extent() -> None:
    path = _COOP_ROOT / "numba_mlir" / "__init__.pyi"
    declarations = [
        statement
        for statement in _module_tree(path).body
        if isinstance(statement, ast.FunctionDef) and statement.name == "ThreadData"
    ]

    assert len(declarations) == 2
    assert all(
        any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in declaration.decorator_list
        )
        for declaration in declarations
    )

    assert all(
        declaration.args.args[0].arg == "items_per_thread"
        for declaration in declarations
    )
    assert all(
        [argument.arg for argument in declaration.args.kwonlyargs] == ["alignas"]
        for declaration in declarations
    )


def test_numba_mlir_local_stub_declarations_have_editor_docstrings() -> None:
    path = _COOP_ROOT / "numba_mlir" / "__init__.pyi"
    exports = set(_literal_exports(path))
    local_declarations = [
        statement
        for statement in _module_tree(path).body
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef))
        and statement.name in exports
    ]

    assert local_declarations
    assert all(ast.get_docstring(declaration) for declaration in local_declarations)


def test_numba_mlir_stub_documents_extension_dtype_storage_boundary() -> None:
    path = _COOP_ROOT / "numba_mlir" / "__init__.pyi"
    tree = _module_tree(path)
    thread_data_docs = [
        ast.get_docstring(statement) or ""
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "ThreadData"
    ]
    temp_storage = _top_level_declaration(path, "TempStorage")
    shared_memory = _top_level_declaration(path, "_SharedMemory")

    assert thread_data_docs
    assert all(
        "thread-local" in doc and "extension dtype" in doc for doc in thread_data_docs
    )
    assert all(
        "matching CUDA and MLIR models" in doc and "fail specialization" in doc
        for doc in thread_data_docs
    )
    assert isinstance(temp_storage, ast.ClassDef)
    assert "opaque byte" in (ast.get_docstring(temp_storage) or "")
    assert isinstance(shared_memory, ast.ClassDef)
    shared_array = next(
        statement
        for statement in shared_memory.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "array"
    )
    shared_doc = ast.get_docstring(shared_array) or ""
    assert "LLVM-backed extension aggregate dtypes" in shared_doc
    assert "must fail compilation" in shared_doc

    for namespace_name in ("_LocalMemory", "_SharedMemory"):
        namespace = _top_level_declaration(path, namespace_name)
        assert isinstance(namespace, ast.ClassDef)
        array = next(
            statement
            for statement in namespace.body
            if isinstance(statement, ast.FunctionDef) and statement.name == "array"
        )
        assert [argument.arg for argument in array.args.args] == [
            "self",
            "shape",
            "dtype",
            "alignas",
        ]
        assert [argument.arg for argument in array.args.kwonlyargs] == ["alignment"]
        assert array.args.vararg is None
        assert array.args.kwarg is None


@pytest.mark.parametrize("scope", ["block", "warp"])
def test_numba_internal_scoped_stubs_match_runtime_exports(scope: str) -> None:
    package = _COOP_ROOT / "numba_mlir" / f"_{scope}"
    runtime_path = package / "__init__.py"
    stub_path = package / "__init__.pyi"
    runtime_bindings = _lazy_runtime_bindings(runtime_path)
    exports = set(_literal_exports(runtime_path))

    assert set(runtime_bindings) == exports
    assert set(_literal_exports(stub_path)) == exports
    assert exports <= _static_declarations(stub_path)
    assert not any(
        isinstance(statement, ast.ImportFrom)
        and statement.module is not None
        and statement.module.startswith("_")
        and statement.module != "_typing"
        for statement in _module_tree(stub_path).body
    )

    factory_declarations = [
        statement
        for statement in _module_tree(stub_path).body
        if isinstance(statement, ast.FunctionDef) and statement.name.startswith("make_")
    ]
    protocol_names = {
        statement.name
        for statement in _module_tree(stub_path).body
        if isinstance(statement, ast.ClassDef)
        and any("Protocol" in ast.unparse(base) for base in statement.bases)
    }

    assert factory_declarations
    for declaration in factory_declarations:
        assert declaration.returns is not None
        return_annotation = ast.unparse(declaration.returns)
        assert "Callable" not in return_annotation
        assert return_annotation.split("[", 1)[0] in protocol_names


@pytest.mark.parametrize("scope", ["block", "warp"])
def test_numba_scoped_factory_overloads_match_exact_runtime_signatures(
    scope: str,
) -> None:
    """Every lazy factory spelling preserves its established Python signature."""

    package = _COOP_ROOT / "numba_mlir" / f"_{scope}"
    runtime_bindings = _lazy_runtime_bindings(package / "__init__.py")
    stub_path = package / "__init__.pyi"

    for public_name, (level, module, source_name) in runtime_bindings.items():
        if level != 1:
            continue
        source_path = package / f"{module}.py"
        source_declarations = _function_declarations(source_path, source_name)
        if not source_declarations:
            continue

        assert len(source_declarations) == 1, (scope, public_name, source_path)
        expected = _parameter_layout(source_declarations[0])
        stub_declarations = _function_declarations(stub_path, public_name)
        assert stub_declarations, (scope, public_name)
        matching = [
            declaration
            for declaration in stub_declarations
            if _parameter_layout(declaration) == expected
        ]

        assert matching, (
            scope,
            public_name,
            expected,
            tuple(map(_parameter_layout, stub_declarations)),
        )
        assert all(
            declaration.args.vararg is None and declaration.args.kwarg is None
            for declaration in stub_declarations
        ), (scope, public_name)

        is_dual_use = not public_name.startswith(("make_", "warp_"))
        if is_dual_use:
            assert any(
                _parameter_layout(declaration) != expected
                for declaration in stub_declarations
            ), (scope, public_name)
            assert all(
                any(
                    isinstance(decorator, ast.Name) and decorator.id == "overload"
                    for decorator in declaration.decorator_list
                )
                for declaration in stub_declarations
            ), (scope, public_name)
        else:
            assert all(
                _parameter_layout(declaration) == expected
                for declaration in stub_declarations
            ), (scope, public_name)


@pytest.mark.parametrize("scope", ["block", "warp"])
@pytest.mark.parametrize("name", ["load", "store", "make_load", "make_store"])
def test_cutlass_internal_scoped_load_store_stubs_are_explicit_and_documented(
    scope: str,
    name: str,
) -> None:
    package = _COOP_ROOT / "cutlass" / f"_{scope}"
    runtime_declaration = _top_level_declaration(package / "__init__.py", name)
    stub_declaration = _top_level_declaration(package / "__init__.pyi", name)

    assert isinstance(runtime_declaration, ast.FunctionDef)
    assert isinstance(stub_declaration, ast.FunctionDef)
    expected_leading, expected_keywords, expected_varargs = (
        _expected_cutlass_wrapper_signature(scope, name)
    )
    actual_leading = (
        *(("positional_only", item.arg) for item in stub_declaration.args.posonlyargs),
        *(("positional", item.arg) for item in stub_declaration.args.args),
    )
    actual_keywords = {item.arg for item in stub_declaration.args.kwonlyargs}

    assert runtime_declaration.args.vararg is not None
    assert runtime_declaration.args.kwarg is not None
    assert actual_leading == expected_leading
    assert actual_keywords == expected_keywords
    assert (stub_declaration.args.vararg is not None) is expected_varargs
    assert stub_declaration.args.kwarg is None
    payload = next(
        item for item in stub_declaration.args.kwonlyargs if item.arg == "payload"
    )
    payload_annotation = ast.unparse(payload.annotation)
    assert "_Payload" in payload_annotation
    assert "Literal['prims']" in payload_annotation
    assert ast.get_docstring(stub_declaration)


@pytest.mark.parametrize("scope", ["block", "warp"])
@pytest.mark.parametrize("protocol", ["_DeferredLoad", "_DeferredStore"])
def test_cutlass_deferred_load_store_protocols_reject_splats(
    scope: str,
    protocol: str,
) -> None:
    path = _COOP_ROOT / "cutlass" / f"_{scope}" / "__init__.pyi"
    declarations = _method_declarations(path, protocol, "__call__")

    assert len(declarations) == 1
    assert declarations[0].args.vararg is None
    assert declarations[0].args.kwarg is None
    keyword_names = {argument.arg for argument in declarations[0].args.kwonlyargs}
    if scope == "block":
        assert _CUTLASS_LAUNCH_METADATA_CONTROLS <= keyword_names
    else:
        assert _CUTLASS_LAUNCH_METADATA_CONTROLS.isdisjoint(keyword_names)


@pytest.mark.parametrize(
    ("stub_relative_path", "runtime_relative_path"),
    _STUB_TO_RUNTIME_EXPORTS.items(),
)
def test_backend_stub_matches_runtime_public_exports(
    stub_relative_path: Path,
    runtime_relative_path: Path,
) -> None:
    stub_exports = set(_literal_exports(_COOP_ROOT / stub_relative_path))
    runtime_exports = set(_literal_exports(_COOP_ROOT / runtime_relative_path))

    assert stub_exports == runtime_exports


def test_common_and_qualified_typing_fixture_passes_strict_pyright() -> None:
    probe = TESTS_ROOT / "support" / "fixtures" / "typing_public_surfaces.py"
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "typing_adjacent_discontinuity_positive.py",
        "typing_adjacent_discontinuity_negative.py",
        "typing_shuffle_positive.py",
        "typing_shuffle_negative.py",
        "typing_histogram_positive.py",
        "typing_histogram_negative.py",
        "typing_reduce_positive.py",
        "typing_reduce_negative.py",
        "typing_scan_positive.py",
        "typing_scan_negative.py",
        "typing_merge_sort_positive.py",
        "typing_merge_sort_negative.py",
        "typing_radix_sort_positive.py",
        "typing_radix_sort_negative.py",
        "typing_radix_rank_positive.py",
        "typing_radix_rank_negative.py",
        "typing_run_length_decode_positive.py",
        "typing_run_length_decode_negative.py",
        "typing_topk_positive.py",
        "typing_topk_negative.py",
        "typing_numeric_dtype_positive.py",
        "typing_numeric_dtype_negative.py",
        "typing_load_store_algorithms_positive.py",
        "typing_load_store_algorithms_negative.py",
    ],
)
def test_group_collective_typing_fixtures_pass_strict_pyright(
    fixture_name: str,
) -> None:
    probe = TESTS_ROOT / "support" / "fixtures" / fixture_name
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


def _xfail_without_cutlass_typing(module_name: str) -> None:
    """Keep optional compiler probes visible in the backend-free contract lane."""

    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError as error:
        if error.name != "cutlass" and not str(error.name).startswith("cutlass."):
            raise
        spec = None
    if spec is None or spec.origin is None:
        pytest.xfail(
            f"{module_name} is qualified only in the pinned CUTLASS compiler lane"
        )

    visible_roots = {
        Path(entry).resolve()
        for entry in _pyright_environment()["PYTHONPATH"].split(os.pathsep)
        if entry
    }
    visible_roots.update(
        Path(path).resolve()
        for key in ("purelib", "platlib")
        if (path := sysconfig.get_path(key))
    )
    module_path = Path(spec.origin).resolve()
    if not any(module_path.is_relative_to(root) for root in visible_roots):
        pytest.xfail(f"{module_name} is outside the module roots visible to Pyright")


def test_cutlass_histogram_compiler_types_pass_strict_pyright() -> None:
    _xfail_without_cutlass_typing("cutlass.base_dsl.typing")
    probe = TESTS_ROOT / "support" / "fixtures" / "typing_histogram_cutlass_external.py"
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


def test_cutlass_reduce_compiler_types_pass_strict_pyright() -> None:
    _xfail_without_cutlass_typing("cutlass.base_dsl.typing")
    probe = TESTS_ROOT / "support" / "fixtures" / "typing_reduce_cutlass_external.py"
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


def test_cutlass_register_collective_types_pass_strict_pyright() -> None:
    _xfail_without_cutlass_typing("cutlass.cute")
    probe = (
        TESTS_ROOT
        / "support"
        / "fixtures"
        / "typing_cutlass_register_collectives_external.py"
    )
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


def test_cutlass_merge_sort_compiler_types_pass_strict_pyright() -> None:
    _xfail_without_cutlass_typing("cutlass.base_dsl.typing")
    probe = (
        TESTS_ROOT / "support" / "fixtures" / "typing_merge_sort_cutlass_external.py"
    )
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "typing_group_kinds_positive.py",
        "typing_group_kinds_negative.py",
    ],
)
def test_common_group_kind_typing_fixtures_pass_strict_pyright(
    fixture_name: str,
) -> None:
    probe = TESTS_ROOT / "support" / "fixtures" / fixture_name
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "typing_cutlass_scoped_positive.py",
        "typing_cutlass_scoped_negative.py",
    ],
)
def test_cutlass_scoped_typing_fixtures_pass_strict_pyright(
    fixture_name: str,
) -> None:
    probe = TESTS_ROOT / "support" / "fixtures" / fixture_name
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


@pytest.mark.parametrize(
    "fixture_name",
    [
        "typing_numba_scoped_positive.py",
        "typing_numba_scoped_negative.py",
    ],
)
def test_numba_scoped_typing_fixtures_pass_strict_pyright(
    fixture_name: str,
) -> None:
    probe = TESTS_ROOT / "support" / "fixtures" / fixture_name
    report = _run_pyright_json(str(probe), "--pythonversion", "3.10")
    summary = report["summary"]

    assert isinstance(summary, dict)
    assert summary["errorCount"] == 0
    assert summary["warningCount"] == 0


@pytest.mark.parametrize("backend", ["cutlass", "numba_mlir"])
def test_qualified_backends_have_complete_public_typing(backend: str) -> None:
    report = _run_pyright_json(
        "--verifytypes",
        f"cuda.coop.{backend}",
        "--ignoreexternal",
    )
    completeness = report["typeCompleteness"]

    assert isinstance(completeness, dict)
    expected_module_root = _active_cuda_coop_search_root() / "cuda" / "coop" / backend
    assert (
        Path(str(completeness["moduleRootDirectory"])).resolve() == expected_module_root
    )
    assert Path(str(completeness["pyTypedPath"])).resolve() == (
        expected_module_root / "py.typed"
    )
    assert completeness["completenessScore"] == 1
    exported = completeness["exportedSymbolCounts"]
    assert isinstance(exported, dict)
    assert exported["withAmbiguousType"] == 0
    assert exported["withUnknownType"] == 0
    assert completeness["missingFunctionDocStringCount"] == 0
    assert completeness["missingClassDocStringCount"] == 0
