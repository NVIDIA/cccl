#!/usr/bin/env python3
"""Detect same-mangled-name ABI breaks in CCCL header-emitted symbols between two revisions.

Each revision (git rev or directory) is extracted, its public headers are compiled into an
object with every inline entity emitted (-O0 -g -fkeep-inline-functions), and symbols plus
DWARF are compared. A finding is a mangled name present in both revisions whose definition
(function return/parameter types, object size/type, or the layout of a reachable class) differs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ARCHIVE_PATHS = ["libcudacxx/include", "cub", "thrust", "cudax/include"]
INCLUDE_DIRS = ["libcudacxx/include", "cub", "thrust", "cudax/include"]
HEADER_ROOTS = [
    ("libcudacxx/include", "cuda"),
    ("cub", "cub"),
    ("thrust", "thrust"),
    ("cudax/include", "cuda/experimental"),
]
DEFAULT_DEFINES = [
    "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP",
    "CCCL_IGNORE_DEPRECATED_CPP_DIALECT",
    "CCCL_IGNORE_DEPRECATED_API",
]
SYMBOL_PREFIXES = (
    "_ZN4cuda",
    "_ZNK4cuda",
    "_ZN3cub",
    "_ZNK3cub",
    "_ZN6thrust",
    "_ZNK6thrust",
)
LOCAL_STATIC_RE = re.compile(r"^_ZZ(.*)E(\d+)(\w+?)(?:_\d+)?$")
SCRIPT_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def log(msg: str) -> None:
    print(f"[cccl-abi-diff] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------------------- source trees


def resolve_tree(spec: str, repo: Path, cache: Path) -> tuple[str, Path]:
    """Return (label, extracted source dir) for a git rev or an existing directory."""
    p = Path(spec).expanduser()
    if p.is_dir():
        h = hashlib.sha256()
        files = []
        for rel in ARCHIVE_PATHS:
            if (p / rel).is_dir():
                files += sorted(f for f in (p / rel).rglob("*") if f.is_file())
        for f in files:
            h.update(str(f.relative_to(p)).encode())
            h.update(f.read_bytes())
        key = "dir-" + h.hexdigest()[:16]
        dest = cache / "src" / key
        if not dest.exists():
            tmp = Path(tempfile.mkdtemp(dir=cache / "src"))
            for rel in ARCHIVE_PATHS:
                if (p / rel).is_dir():
                    shutil.copytree(p / rel, tmp / rel, symlinks=True)
            tmp.rename(dest)
        return f"{spec} ({key})", dest
    r = run(["git", "-C", str(repo), "rev-parse", "--verify", f"{spec}^{{commit}}"])
    if r.returncode != 0:
        sys.exit(f"cannot resolve revision {spec!r} in {repo}: {r.stderr.strip()}")
    sha = r.stdout.strip()
    dest = cache / "src" / sha
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(dir=cache / "src"))
        present = [
            a
            for a in ARCHIVE_PATHS
            if run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}:{a}"]).returncode
            == 0
        ]
        arc = subprocess.Popen(
            ["git", "-C", str(repo), "archive", sha, *present], stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(tmp)], stdin=arc.stdout, check=True)
        if arc.wait() != 0:
            sys.exit(f"git archive failed for {spec}")
        tmp.rename(dest)
    return f"{spec} ({sha[:12]})", dest


def public_headers(src: Path) -> list[str]:
    """Public (non-detail) headers, as include names."""
    out = []
    for root, sub in HEADER_ROOTS:
        base = src / root
        if not (base / sub).is_dir():
            continue
        for f in sorted((base / sub).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(base).as_posix()
            parts = rel.split("/")
            if any(
                x.startswith("__") or x in ("detail", "experimental")
                for x in parts[1:-1]
            ):
                if not (root == "cudax/include" and parts[1] == "experimental"):
                    continue
            if any(x.startswith("__") or x == "detail" for x in parts):
                continue
            if root == "libcudacxx/include" and parts[1] == "experimental":
                continue
            if f.suffix not in ("", ".h", ".cuh", ".hpp"):
                continue
            if root == "cudax/include" and len(parts) < 3:
                continue
            out.append(rel)
    return out


# ----------------------------------------------------------------------------- compilation


@dataclass
class Toolchain:
    cxx: str
    nvcc: str | None
    std: str
    cuda_include: str
    extra: list[str]

    def flags(self, src: Path, lang: str) -> list[str]:
        inc = [f"-I{src / d}" for d in INCLUDE_DIRS if (src / d).is_dir()]
        common = [
            f"-std={self.std}",
            *inc,
            f"-I{self.cuda_include}",
            *[f"-D{d}" for d in DEFAULT_DEFINES],
            "-w",
        ]
        if lang == "cu":
            return [
                self.nvcc,
                "-x",
                "cu",
                *common,
                "--extended-lambda",
                *[f"-Xcompiler={x}" for x in self.extra],
            ]
        return [self.cxx, "-x", "c++", *common, *self.extra]

    def compile_flags(self, lang: str) -> list[str]:
        if lang == "cu":
            return [
                "-O0",
                "-g",
                "-Xcompiler=-fkeep-inline-functions",
                f"-ccbin={self.cxx}",
            ]
        return ["-O0", "-g", "-fkeep-inline-functions"]

    def ident(self) -> str:
        v = [run([self.cxx, "--version"]).stdout.splitlines()[0]]
        if self.nvcc:
            v.append(run([self.nvcc, "--version"]).stdout.strip().splitlines()[-1])
        return hashlib.sha256(
            json.dumps(
                [v, self.std, self.cuda_include, self.extra, DEFAULT_DEFINES]
            ).encode()
        ).hexdigest()[:12]


def probe_header(tc: Toolchain, src: Path, header: str) -> str | None:
    """Return the language ('cpp' or 'cu') in which the header compiles standalone, if any."""
    code = f"#include <{header}>\n"
    if run([*tc.flags(src, "cpp"), "-fsyntax-only", "-"], input=code).returncode == 0:
        return "cpp"
    if tc.nvcc:
        with tempfile.TemporaryDirectory() as d:
            tu = Path(d) / "probe.cu"
            tu.write_text(code)
            cmd = [
                *tc.flags(src, "cu"),
                f"-ccbin={tc.cxx}",
                "-c",
                str(tu),
                "-o",
                str(Path(d) / "p.o"),
            ]
            if run(cmd).returncode == 0:
                return "cu"
    return None


def build_object(
    tc: Toolchain, src: Path, headers: list[str], obj: Path, lang: str
) -> bool:
    tu = obj.with_suffix(".cu" if lang == "cu" else ".cpp")
    tu.write_text("".join(f"#include <{h}>\n" for h in headers))
    r = run(
        [*tc.flags(src, lang), *tc.compile_flags(lang), "-c", str(tu), "-o", str(obj)]
    )
    if r.returncode != 0:
        obj.with_suffix(".err").write_text(r.stderr)
    return r.returncode == 0


def build_objects(
    tc: Toolchain, src: Path, work: Path, jobs: int, chunk: int
) -> list[Path]:
    """Compile all standalone-compilable public headers; idempotent via the cache."""
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "objects.json"
    if manifest.exists():
        objs = [Path(p) for p in json.loads(manifest.read_text())]
        if all(o.exists() for o in objs):
            return objs
    headers = public_headers(src)
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        langs = list(ex.map(lambda h: probe_header(tc, src, h), headers))
    by_lang = {
        lang: [h for h, hl in zip(headers, langs) if hl == lang]
        for lang in ("cpp", "cu")
    }
    bad = [h for h, hl in zip(headers, langs) if hl is None]
    (work / "headers.json").write_text(
        json.dumps({**by_lang, "skipped": bad}, indent=1)
    )
    log(
        f"{src.name}: {len(by_lang['cpp'])} headers compile as C++, {len(by_lang['cu'])} only as CUDA,"
        f" {len(bad)} skipped"
    )
    groups = [
        (lang, hs[i : i + chunk])
        for lang, hs in by_lang.items()
        for i in range(0, len(hs), chunk)
    ]

    def build_group(i_g):
        i, (lang, g) = i_g
        obj = work / f"{lang}{i:03d}.o"
        if build_object(tc, src, g, obj, lang):
            return [obj]
        # Header combination conflicts: fall back to one object per header.
        res = []
        for j, h in enumerate(g):
            o = work / f"{lang}{i:03d}_{j:03d}.o"
            if build_object(tc, src, [h], o, lang):
                res.append(o)
        return res

    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        objs = [o for lst in ex.map(build_group, enumerate(groups)) for o in lst]
    manifest.write_text(json.dumps([str(o) for o in objs]))
    return objs


# ----------------------------------------------------------------------------- DWARF parsing

DIE_RE = re.compile(r"^\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: (\d+)(?: \((\w+)\))?")
ATTR_RE = re.compile(r"^\s*<[0-9a-f]+>\s+(DW_AT_\w+)\s*: ?(.*)$")
FORM_RE = re.compile(
    r"^\((?:strp|string|line_strp|strx\d?|data\d|udata|sdata|implicit_const|flag_present|flag|ref\w*|exprloc|sec_offset|block\d?)\)\s*"
)
STR_RE = re.compile(r"^\((?:indirect (?:line )?string|indexed string|offset)[^)]*\): ")
REF_RE = re.compile(r"<0x([0-9a-f]+)>")
KEEP_ATTRS = {
    "DW_AT_name",
    "DW_AT_linkage_name",
    "DW_AT_MIPS_linkage_name",
    "DW_AT_type",
    "DW_AT_byte_size",
    "DW_AT_data_member_location",
    "DW_AT_data_bit_offset",
    "DW_AT_bit_size",
    "DW_AT_declaration",
    "DW_AT_specification",
    "DW_AT_abstract_origin",
    "DW_AT_upper_bound",
    "DW_AT_count",
    "DW_AT_artificial",
    "DW_AT_external",
    "DW_AT_alignment",
    "DW_AT_calling_convention",
    "DW_AT_location",
    "DW_AT_const_value",
    "DW_AT_virtuality",
    "DW_AT_export_symbols",
}


@dataclass
class Die:
    off: int
    tag: str
    parent: int | None
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


def parse_dwarf(obj: Path) -> dict[int, Die]:
    p = subprocess.Popen(
        ["readelf", "--wide", "--debug-dump=info", str(obj)],
        stdout=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    dies: dict[int, Die] = {}
    stack: list[int] = []
    cur: Die | None = None
    for line in p.stdout:
        m = DIE_RE.match(line)
        if m:
            depth, off, abbrev, tag = int(m[1]), int(m[2], 16), m[3], m[4]
            if abbrev == "0":
                cur = None
                continue
            del stack[depth:]
            parent = stack[-1] if stack else None
            cur = Die(off, tag, parent)
            dies[off] = cur
            if parent is not None:
                dies[parent].children.append(off)
            stack.append(off)
            continue
        if cur is None:
            continue
        m = ATTR_RE.match(line)
        if not m or m[1] not in KEEP_ATTRS:
            continue
        val = FORM_RE.sub("", m[2].strip())
        if m[1] in ("DW_AT_name", "DW_AT_linkage_name", "DW_AT_MIPS_linkage_name"):
            val = STR_RE.sub("", val)
        cur.attrs[m[1]] = val
    p.wait()
    return dies


def ref(d: Die, attr: str) -> int | None:
    v = d.attrs.get(attr)
    if v is None:
        return None
    m = REF_RE.search(v)
    return int(m[1], 16) if m else None


def num(v: str | None) -> int | None:
    if v is None:
        return None
    m = re.match(r"(0x[0-9a-f]+|-?\d+)", v.strip())
    return int(m[1], 0) if m else None


CLASS_TAGS = {"DW_TAG_structure_type", "DW_TAG_class_type", "DW_TAG_union_type"}
SCOPE_TAGS = CLASS_TAGS | {"DW_TAG_namespace", "DW_TAG_enumeration_type"}


class Dwarf:
    """Per-object view: qualified names, canonical type names, class layouts."""

    def __init__(self, dies: dict[int, Die]):
        self.d = dies
        self._qn: dict[int, str] = {}
        self._tn: dict[int, str] = {}
        self.defs: dict[str, int] = {}
        for off, die in dies.items():
            if (
                die.tag in CLASS_TAGS | {"DW_TAG_enumeration_type"}
                and "DW_AT_declaration" not in die.attrs
                and "DW_AT_name" in die.attrs
            ):
                self.defs.setdefault(self.qualname(off), off)

    def qualname(self, off: int) -> str:
        if off in self._qn:
            return self._qn[off]
        die = self.d[off]
        spec = ref(die, "DW_AT_specification") or ref(die, "DW_AT_abstract_origin")
        if spec is not None and spec in self.d and "DW_AT_name" not in die.attrs:
            q = self.qualname(spec)
        else:
            name = die.attrs.get("DW_AT_name", "")
            scope = []
            p = die.parent
            while p is not None:
                pd = self.d[p]
                if pd.tag in SCOPE_TAGS:
                    scope.append(pd.attrs.get("DW_AT_name", "(anonymous)"))
                elif pd.tag == "DW_TAG_subprogram":
                    scope.append(self.qualname(p) + "()")
                    break
                p = pd.parent
            q = "::".join(list(reversed(scope)) + [name]) if name else ""
        self._qn[off] = q
        return q

    def strip(self, off: int | None) -> int | None:
        """Follow cv-qualifiers and typedefs to the underlying type DIE."""
        seen = 0
        while off is not None and seen < 64:
            die = self.d.get(off)
            if die is None:
                return None
            if die.tag in (
                "DW_TAG_typedef",
                "DW_TAG_const_type",
                "DW_TAG_volatile_type",
                "DW_TAG_restrict_type",
                "DW_TAG_atomic_type",
            ):
                off = ref(die, "DW_AT_type")
                seen += 1
                continue
            if (
                die.tag in CLASS_TAGS | {"DW_TAG_enumeration_type"}
                and "DW_AT_declaration" in die.attrs
            ):
                return self.defs.get(self.qualname(off), off)
            return off
        return off

    def type_name(self, off: int | None) -> str:
        """Canonical (typedef-resolved) type spelling."""
        if off is None:
            return "void"
        if off in self._tn:
            return self._tn[off]
        self._tn[off] = "<recursive>"
        die = self.d.get(off)
        if die is None:
            return "<missing>"
        t, inner = die.tag, ref(die, "DW_AT_type")
        if t == "DW_TAG_base_type" or t == "DW_TAG_unspecified_type":
            s = die.attrs.get("DW_AT_name", "?")
        elif t in ("DW_TAG_typedef", "DW_TAG_restrict_type", "DW_TAG_atomic_type"):
            s = self.type_name(inner)
        elif t == "DW_TAG_const_type":
            s = self.type_name(inner) + " const"
        elif t == "DW_TAG_volatile_type":
            s = self.type_name(inner) + " volatile"
        elif t == "DW_TAG_pointer_type":
            s = self.type_name(inner) + "*"
        elif t == "DW_TAG_reference_type":
            s = self.type_name(inner) + "&"
        elif t == "DW_TAG_rvalue_reference_type":
            s = self.type_name(inner) + "&&"
        elif t == "DW_TAG_array_type":
            dims = []
            for c in die.children:
                cd = self.d[c]
                if cd.tag == "DW_TAG_subrange_type":
                    ub, cnt = (
                        num(cd.attrs.get("DW_AT_upper_bound")),
                        num(cd.attrs.get("DW_AT_count")),
                    )
                    dims.append(
                        str(
                            cnt
                            if cnt is not None
                            else (ub + 1 if ub is not None else "")
                        )
                    )
            s = self.type_name(inner) + "".join(f"[{x}]" for x in dims)
        elif t == "DW_TAG_subroutine_type":
            ps = [
                self.type_name(ref(self.d[c], "DW_AT_type"))
                for c in die.children
                if self.d[c].tag == "DW_TAG_formal_parameter"
            ]
            s = f"{self.type_name(inner)}({', '.join(ps)})"
        elif t == "DW_TAG_ptr_to_member_type":
            s = f"{self.type_name(inner)} {self.type_name(ref(die, 'DW_AT_containing_type'))}::*"
        else:
            s = self.qualname(off) or f"<anon {t}>"
        self._tn[off] = s
        return s

    def size(self, off: int | None) -> int | None:
        off = self.strip(off)
        if off is None:
            return 0
        die = self.d[off]
        if die.tag in (
            "DW_TAG_pointer_type",
            "DW_TAG_reference_type",
            "DW_TAG_rvalue_reference_type",
            "DW_TAG_ptr_to_member_type",
        ):
            return num(die.attrs.get("DW_AT_byte_size")) or (
                16 if die.tag == "DW_TAG_ptr_to_member_type" else 8
            )
        if die.tag == "DW_TAG_array_type":
            n = 1
            for c in die.children:
                cd = self.d[c]
                if cd.tag == "DW_TAG_subrange_type":
                    ub, cnt = (
                        num(cd.attrs.get("DW_AT_upper_bound")),
                        num(cd.attrs.get("DW_AT_count")),
                    )
                    n *= cnt if cnt is not None else (ub + 1 if ub is not None else 0)
            es = self.size(ref(die, "DW_AT_type"))
            return None if es is None else n * es
        return num(die.attrs.get("DW_AT_byte_size"))

    def kind(self, off: int | None) -> str:
        off = self.strip(off)
        if off is None:
            return "void"
        return self.d[off].tag.removeprefix("DW_TAG_").removesuffix("_type")

    def type_desc(self, off: int | None) -> dict:
        s = self.strip(off)
        desc = {
            "name": self.type_name(off),
            "kind": self.kind(off),
            "size": self.size(off),
        }
        if s is not None and self.d[s].tag in CLASS_TAGS:
            cc = self.d[s].attrs.get("DW_AT_calling_convention", "")
            desc["pass_by"] = (
                "reference"
                if "reference" in cc
                else "value"
                if "value" in cc
                else self._guess_pass_by(s)
            )
        return desc

    def _guess_pass_by(self, off: int) -> str:
        """Itanium: non-trivial copy/move/dtor => passed indirectly (approximation from DWARF)."""
        for c in self.d[off].children:
            cd = self.d[c]
            if cd.tag == "DW_TAG_subprogram" and "DW_AT_artificial" not in cd.attrs:
                n = cd.attrs.get("DW_AT_name", "")
                if n.startswith("~") or "DW_AT_virtuality" in cd.attrs:
                    return "reference"
        return "value?"

    def class_layout(self, off: int) -> dict:
        die = self.d[off]
        members, bases = [], []
        for c in die.children:
            cd = self.d[c]
            if (
                cd.tag == "DW_TAG_member"
                and "DW_AT_external" not in cd.attrs
                and "DW_AT_declaration" not in cd.attrs
            ):
                t = ref(cd, "DW_AT_type")
                loc = num(cd.attrs.get("DW_AT_data_member_location"))
                bit = num(cd.attrs.get("DW_AT_data_bit_offset"))
                members.append(
                    {
                        "name": cd.attrs.get("DW_AT_name", ""),
                        "offset": (
                            loc
                            if loc is not None
                            else (bit // 8 if bit is not None else 0)
                        ),
                        "bits": num(cd.attrs.get("DW_AT_bit_size")),
                        "type": self.type_name(t),
                        "size": self.size(t),
                        "_ref": t,
                    }
                )
            elif cd.tag == "DW_TAG_inheritance":
                t = ref(cd, "DW_AT_type")
                bases.append(
                    {
                        "type": self.type_name(t),
                        "offset": num(cd.attrs.get("DW_AT_data_member_location")),
                        "size": self.size(t),
                        "_ref": t,
                    }
                )
        return {
            "size": num(die.attrs.get("DW_AT_byte_size")),
            "align": num(die.attrs.get("DW_AT_alignment")),
            "members": members,
            "bases": bases,
        }

    def referenced_classes(self, off: int | None) -> list[tuple[str, int]]:
        """Class DIEs reached by a type through cv/typedef/pointer/ref/array."""
        seen = 0
        while off is not None and seen < 64:
            seen += 1
            die = self.d.get(off)
            if die is None:
                return []
            if die.tag in CLASS_TAGS:
                s = self.strip(off)
                return [(self.qualname(s), s)] if s is not None else []
            if die.tag == "DW_TAG_subroutine_type":
                out = []
                for c in die.children:
                    out += self.referenced_classes(ref(self.d[c], "DW_AT_type"))
                return out + self.referenced_classes(ref(die, "DW_AT_type"))
            off = ref(die, "DW_AT_type")
        return []


@dataclass
class Snapshot:
    label: str
    functions: dict = field(default_factory=dict)  # mangled -> {ret, params, binding}
    objects: dict = field(default_factory=dict)  # mangled -> {size, type, binding}
    classes: dict = field(default_factory=dict)  # qualname -> layout
    symbol_classes: dict = field(default_factory=dict)  # mangled -> [qualnames]
    class_refs: dict = field(default_factory=dict)  # qualname -> [(via, qualname)]


def read_symbols(obj: Path) -> list[tuple[str, str, int, str, str]]:
    out = []
    for line in run(["readelf", "-Ws", str(obj)]).stdout.splitlines():
        f = line.split()
        if len(f) >= 8 and f[0].endswith(":") and f[3] in ("FUNC", "OBJECT"):
            if f[4] in ("WEAK", "UNIQUE", "GLOBAL") and f[6] != "UND":
                out.append((f[7], f[3], int(f[2], 0), f[4], f[5]))
    return out


def find_local_static(
    dw: Dwarf, by_link: dict[str, list[int]], mangled: str
) -> int | None:
    m = LOCAL_STATIC_RE.match(mangled)
    if not m:
        return None
    func, var = "_Z" + m[1], m[3]
    for fo in by_link.get(func, []):
        # The variable lives under the definition (which may point at this declaration).
        cands = [fo] + [
            o
            for o, d in dw.d.items()
            if d.tag == "DW_TAG_subprogram" and ref(d, "DW_AT_specification") == fo
        ]
        for c in cands:
            stack = list(dw.d[c].children)
            while stack:
                x = stack.pop()
                xd = dw.d[x]
                if xd.tag == "DW_TAG_variable" and xd.attrs.get("DW_AT_name") == var:
                    return x
                if xd.tag == "DW_TAG_lexical_block":
                    stack += xd.children
    return None


def snapshot_object(obj: Path) -> dict:
    """Extract ABI facts from one object (cached as JSON next to it)."""
    cache = obj.with_suffix(f".abi-{SCRIPT_HASH}.json")
    if cache.exists() and cache.stat().st_mtime >= obj.stat().st_mtime:
        return json.loads(cache.read_text())
    syms = [
        s
        for s in read_symbols(obj)
        if s[0].startswith(SYMBOL_PREFIXES)
        or s[0].startswith("_ZZ")
        or s[0].startswith("_ZTV")
        or s[0].startswith("_ZGV")
    ]
    dw = Dwarf(parse_dwarf(obj))
    by_link: dict[str, list[int]] = {}
    for off, die in dw.d.items():
        ln = die.attrs.get("DW_AT_linkage_name") or die.attrs.get(
            "DW_AT_MIPS_linkage_name"
        )
        if ln:
            by_link.setdefault(ln, []).append(off)
    funcs, objs, symcls, classes_needed = {}, {}, {}, {}
    for name, typ, size, bind, vis in syms:
        if typ == "FUNC":
            offs = by_link.get(name)
            if not offs:
                continue
            fo = offs[0]
            fd = dw.d[fo]
            src = fo
            while "DW_AT_type" not in dw.d[src].attrs and (
                ref(dw.d[src], "DW_AT_specification")
                or ref(dw.d[src], "DW_AT_abstract_origin")
            ):
                src = ref(dw.d[src], "DW_AT_specification") or ref(
                    dw.d[src], "DW_AT_abstract_origin"
                )
            rt = ref(dw.d[src], "DW_AT_type")
            params = [
                ref(dw.d[c], "DW_AT_type")
                for c in fd.children
                if dw.d[c].tag == "DW_TAG_formal_parameter"
            ]
            refs = dw.referenced_classes(rt)
            for p in params:
                refs += dw.referenced_classes(p)
            funcs[name] = {
                "qualname": dw.qualname(fo),
                "binding": f"{bind}/{vis}",
                "ret": dw.type_desc(rt),
                "params": [dw.type_desc(p) for p in params],
            }
            symcls[name] = sorted({q for q, _ in refs if q})
            for q, o in refs:
                classes_needed.setdefault(q, o)
        else:
            vo = (
                by_link.get(name, [None])[0]
                if name in by_link
                else find_local_static(dw, by_link, name)
            )
            ent = {"size": size, "binding": f"{bind}/{vis}", "type": None}
            if vo is not None:
                t = ref(dw.d[vo], "DW_AT_type")
                if t is None and ref(dw.d[vo], "DW_AT_specification"):
                    t = ref(dw.d[ref(dw.d[vo], "DW_AT_specification")], "DW_AT_type")
                ent["type"] = dw.type_desc(t)
                refs = dw.referenced_classes(t)
                symcls[name] = sorted({q for q, _ in refs if q})
                for q, o in refs:
                    classes_needed.setdefault(q, o)
            objs[name] = ent
    # Transitive closure over members/bases/pointees for layout comparison.
    classes, class_refs = {}, {}
    work = list(classes_needed.items())
    while work:
        q, o = work.pop()
        if q in classes or not q or o is None:
            continue
        o = dw.strip(o)
        if o is None or "DW_AT_declaration" in dw.d[o].attrs:
            continue
        lay = dw.class_layout(o)
        for m in lay["members"] + lay["bases"]:
            for cq, co in dw.referenced_classes(m.pop("_ref")):
                class_refs.setdefault(cq, []).append([q, m.get("name", "<base>")])
                if cq not in classes:
                    work.append((cq, co))
        classes[q] = lay
    res = {
        "functions": funcs,
        "objects": objs,
        "classes": classes,
        "symbol_classes": symcls,
        "class_refs": class_refs,
    }
    cache.write_text(json.dumps(res))
    return res


def build_snapshot(label: str, objs: list[Path], jobs: int) -> Snapshot:
    snap = Snapshot(label)
    with concurrent.futures.ProcessPoolExecutor(jobs) as ex:
        parts = list(ex.map(snapshot_object, objs))
    for part in parts:
        for k in ("functions", "objects", "classes", "symbol_classes"):
            getattr(snap, k).update(
                {n: v for n, v in part[k].items() if n not in getattr(snap, k)}
            )
        for q, lst in part["class_refs"].items():
            cur = snap.class_refs.setdefault(q, [])
            cur += [x for x in lst if x not in cur]
    return snap


# ----------------------------------------------------------------------------- comparison


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    r = run(["c++filt"], input="\n".join(names) + "\n")
    return dict(zip(names, r.stdout.splitlines()))


def strip_cv(name: str) -> str:
    return re.sub(r"( const| volatile)+$", "", name)


def norm_kind(k: str) -> str:
    return "class" if k in ("structure", "class") else k


def worst(*sevs: str) -> str:
    return min(sevs, key=SEVERITY_RANK.get)


def cmp_type(a: dict, b: dict) -> tuple[str, str] | None:
    """Return (severity, reason) if two signature type descriptors are ABI-incompatible."""
    if a["size"] != b["size"]:
        return "critical", f"size {a['size']} -> {b['size']}"
    if norm_kind(a["kind"]) != norm_kind(b["kind"]):
        return "critical", f"kind {a['kind']} -> {b['kind']}"
    pa, pb = a.get("pass_by"), b.get("pass_by")
    if pa != pb and "reference" in (pa, pb):
        return "critical", f"passing convention {pa} -> {pb}"
    if strip_cv(a["name"]) != strip_cv(b["name"]):
        return "high", "type changed with same size"
    return None


def layout_diff(a: dict, b: dict) -> tuple[str, list[str]]:
    """Severity and description of a class layout change ('' severity if unchanged)."""
    diffs, sev = [], "low"
    if a["size"] != b["size"]:
        diffs.append(f"sizeof {a['size']} -> {b['size']}")
        sev = "critical"
    if a["align"] != b["align"]:
        diffs.append(f"alignas {a['align']} -> {b['align']}")
        sev = worst(
            sev, "critical" if None not in (a["align"], b["align"]) else "medium"
        )

    def full(m):
        return (
            m.get("name", "<base>"),
            m["offset"],
            m.get("bits"),
            m["type"],
            m["size"],
        )

    def shape(ms):
        return [(m["offset"], m.get("bits"), m["size"]) for m in ms]

    def typed(ms):
        return [(m["offset"], m.get("bits"), m["size"], m["type"]) for m in ms]

    ma, mb = [full(m) for m in a["members"]], [full(m) for m in b["members"]]
    if ma != mb:
        sa, sb = set(ma), set(mb)
        diffs += [
            f"- member {m[0]} @{m[1]} : {m[3]} ({m[4]}B)" for m in ma if m not in sb
        ]
        diffs += [
            f"+ member {m[0]} @{m[1]} : {m[3]} ({m[4]}B)" for m in mb if m not in sa
        ]
        if shape(a["members"]) != shape(b["members"]):
            sev = "critical"
        elif typed(a["members"]) != typed(b["members"]):
            sev = worst(sev, "high")
    ba, bb = [full(m) for m in a["bases"]], [full(m) for m in b["bases"]]
    if ba != bb:
        diffs.append(
            f"bases {[(x[3], x[1]) for x in ba]} -> {[(x[3], x[1]) for x in bb]}"
        )

        def nonempty(ms):
            return [
                (m["offset"], m["size"], m["type"])
                for m in ms
                if m["size"] not in (0, 1)
            ]

        sev = worst(
            sev, "high" if nonempty(a["bases"]) != nonempty(b["bases"]) else "medium"
        )
    return (sev if diffs else ""), diffs


def layout_brief(snap: Snapshot, type_name: str) -> str | None:
    lay = snap.classes.get(strip_cv(type_name))
    if lay is None:
        return None
    ms = [
        f"{m.get('name', '<base>')}@{m['offset']}:{m['type']}"
        for m in lay["bases"] + lay["members"]
    ]
    return f"{{{', '.join(ms)}}}"


def type_change_evidence(a: Snapshot, b: Snapshot, ta: dict, tb: dict) -> list[str]:
    if strip_cv(ta["name"]) == strip_cv(tb["name"]):
        return []
    la, lb = layout_brief(a, ta["name"]), layout_brief(b, tb["name"])
    return [f"  a layout {la}", f"  b layout {lb}"] if la and lb else []


def compare(a: Snapshot, b: Snapshot) -> dict:
    findings = []
    common_f = sorted(set(a.functions) & set(b.functions))
    common_o = sorted(set(a.objects) & set(b.objects))
    dm = demangle(sorted(set(common_f) | set(common_o)))

    for n in common_f:
        fa, fb = a.functions[n], b.functions[n]
        reasons, sev = [], "low"
        r = cmp_type(fa["ret"], fb["ret"])
        if r:
            reasons.append(f"return {fa['ret']['name']} -> {fb['ret']['name']}: {r[1]}")
            reasons += type_change_evidence(a, b, fa["ret"], fb["ret"])
            sev = min(sev, r[0], key=SEVERITY_RANK.get)
        if len(fa["params"]) != len(fb["params"]):
            reasons.append(
                f"parameter count {len(fa['params'])} -> {len(fb['params'])}"
            )
            sev = "critical"
        else:
            for i, (pa, pb) in enumerate(zip(fa["params"], fb["params"])):
                r = cmp_type(pa, pb)
                if r:
                    reasons.append(f"param {i} {pa['name']} -> {pb['name']}: {r[1]}")
                    sev = min(sev, r[0], key=SEVERITY_RANK.get)
        if reasons:
            findings.append(
                {
                    "category": "A_function_signature",
                    "severity": sev,
                    "symbol": n,
                    "demangled": dm.get(n, n),
                    "details": reasons,
                    "a": fa,
                    "b": fb,
                }
            )

    for n in common_o:
        oa, ob = a.objects[n], b.objects[n]
        reasons, sev = [], "low"
        if oa["size"] != ob["size"]:
            reasons.append(f"st_size {oa['size']} -> {ob['size']}")
            sev = "critical"
        ta, tb = oa.get("type"), ob.get("type")
        if ta and tb and strip_cv(ta["name"]) != strip_cv(tb["name"]):
            reasons.append(f"type {ta['name']} -> {tb['name']}")
            reasons += type_change_evidence(a, b, ta, tb)
            sev = worst(sev, "high")
        if reasons:
            findings.append(
                {
                    "category": "B_object",
                    "severity": sev,
                    "symbol": n,
                    "demangled": dm.get(n, n),
                    "details": reasons,
                    "a": oa,
                    "b": ob,
                }
            )

    # C: layouts of classes reachable from symbols common to both revisions.
    users: dict[str, list[str]] = {}
    for n in common_f + common_o:
        for q in set(a.symbol_classes.get(n, [])) & set(b.symbol_classes.get(n, [])):
            users.setdefault(q, []).append(n)
    reach: dict[str, list] = {q: [["symbol", s] for s in v] for q, v in users.items()}

    def children(snap: Snapshot) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for child, lst in snap.class_refs.items():
            for parent, _ in lst:
                out.setdefault(parent, set()).add(child)
        return out

    ca, cb = children(a), children(b)
    frontier = list(reach)
    while frontier:
        q = frontier.pop()
        for child in ca.get(q, set()) & cb.get(q, set()):
            if child not in reach:
                reach[child] = []
                frontier.append(child)
            if ["via", q] not in reach[child]:
                reach[child].append(["via", q])
    for q in sorted(reach):
        if q not in a.classes or q not in b.classes:
            continue
        sev, diffs = layout_diff(a.classes[q], b.classes[q])
        if not diffs:
            continue
        direct = [s for k, s in reach[q] if k == "symbol"]
        if not direct:
            sev = {"critical": "high", "high": "medium", "medium": "low", "low": "low"}[
                sev
            ]
        findings.append(
            {
                "category": "C_class_layout",
                "severity": sev,
                "symbol": q,
                "demangled": q,
                "details": diffs,
                "referenced_by": sorted(dm.get(s, s) for s in direct)[:20],
                "referenced_by_count": len(direct),
                "reached_via": sorted({s for k, s in reach[q] if k == "via"})[:20],
            }
        )

    findings.sort(
        key=lambda f: (SEVERITY_RANK[f["severity"]], f["category"], f["demangled"])
    )
    counts: dict[str, dict[str, int]] = {}
    for f in findings:
        counts.setdefault(f["category"], {}).setdefault(f["severity"], 0)
        counts[f["category"]][f["severity"]] += 1
    return {
        "a": a.label,
        "b": b.label,
        "stats": {
            "functions": [len(a.functions), len(b.functions), len(common_f)],
            "objects": [len(a.objects), len(b.objects), len(common_o)],
            "classes": [len(a.classes), len(b.classes)],
        },
        "counts": counts,
        "findings": findings,
    }


def summarize(rep: dict, limit: int) -> str:
    s = rep["stats"]
    lines = [
        f"CCCL ABI diff: {rep['a']}  ->  {rep['b']}",
        f"  emitted functions: {s['functions'][0]} / {s['functions'][1]} ({s['functions'][2]} common)",
        f"  emitted objects:   {s['objects'][0]} / {s['objects'][1]} ({s['objects'][2]} common)",
        f"  findings: {len(rep['findings'])}  " + json.dumps(rep["counts"]),
    ]
    for f in rep["findings"][:limit]:
        lines.append(f"[{f['severity'].upper():8}] {f['category']}: {f['demangled']}")
        if f["category"] != "C_class_layout":
            lines.append(f"      {f['symbol']}")
        for d in f["details"][:8]:
            lines.append(f"      {d}")
        if f.get("referenced_by"):
            lines.append(
                f"      referenced by {f['referenced_by_count']} common symbol(s), e.g. "
                + "; ".join(f["referenced_by"][:3])
            )
        elif f.get("reached_via"):
            lines.append(f"      reached via {', '.join(f['reached_via'][:3])}")
    if len(rep["findings"]) > limit:
        lines.append(f"... {len(rep['findings']) - limit} more in JSON report")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="cccl-abi-diff", description=__doc__)
    ap.add_argument(
        "rev_a", help="git revision or directory containing a CCCL source tree"
    )
    ap.add_argument("rev_b")
    ap.add_argument(
        "--repo",
        default=os.environ.get("CCCL_REPO", str(Path(__file__).resolve().parents[2])),
    )
    ap.add_argument(
        "--cache-dir",
        default=os.environ.get("CCCL_ABI_DIFF_CACHE", "~/.cache/cccl-abi-diff"),
    )
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument(
        "--nvcc",
        default=os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
        help="nvcc for headers that only compile as CUDA ('' to disable)",
    )
    ap.add_argument("--std", default="c++17")
    ap.add_argument("--cuda-include", default="/usr/local/cuda/include")
    ap.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="extra compiler flag (repeatable)",
    )
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument(
        "--chunk", type=int, default=24, help="headers per translation unit"
    )
    ap.add_argument(
        "--exclude", help="regex; drop findings whose demangled name matches"
    )
    ap.add_argument(
        "--json", help="write full JSON report to this path ('-' for stdout)"
    )
    ap.add_argument(
        "--limit", type=int, default=40, help="findings shown in the text summary"
    )
    ap.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium", "low", "never"],
        default="high",
    )
    args = ap.parse_args()

    repo = Path(args.repo).expanduser()
    cache = Path(args.cache_dir).expanduser()
    (cache / "src").mkdir(parents=True, exist_ok=True)
    tc = Toolchain(
        args.cxx, args.nvcc or None, args.std, args.cuda_include, args.extra_flag
    )
    tcid = tc.ident()
    snaps = []
    for spec in (args.rev_a, args.rev_b):
        label, src = resolve_tree(spec, repo, cache)
        log(f"{label}: compiling")
        objs = build_objects(
            tc, src, cache / "obj" / tcid / src.name, args.jobs, args.chunk
        )
        log(f"{label}: {len(objs)} objects; extracting DWARF")
        snaps.append(build_snapshot(label, objs, args.jobs))
    rep = compare(*snaps)
    if args.exclude:
        rx = re.compile(args.exclude)
        rep["findings"] = [f for f in rep["findings"] if not rx.search(f["demangled"])]
        rep["counts"] = {}
        for f in rep["findings"]:
            c = rep["counts"].setdefault(f["category"], {})
            c[f["severity"]] = c.get(f["severity"], 0) + 1
    rep["toolchain"] = {
        "cxx": run([tc.cxx, "--version"]).stdout.splitlines()[0],
        "std": tc.std,
        "machine": os.uname().machine,
    }
    if args.json == "-":
        print(json.dumps(rep, indent=1))
    else:
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(rep, indent=1))
        print(summarize(rep, args.limit))
    if args.fail_on == "never":
        return 0
    return int(
        any(
            SEVERITY_RANK[f["severity"]] <= SEVERITY_RANK[args.fail_on]
            for f in rep["findings"]
        )
    )


if __name__ == "__main__":
    sys.exit(main())
