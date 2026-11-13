# Sphinx configuration for the CCCL Python documentation.
#
# The Python libraries ship on their own release line (cuda-cccl 1.x) while the
# C++ libraries are on 3.x, so they are published as two independent products:
# C++ at /cccl/cpp/<version>/ and Python at /cccl/python/<version>/. This file
# configures the Python product only; it needs no Doxygen or breathe.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Sphinx extensions live in docs/_ext; this build uses coop_visualization.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_ext"))
)

import registry  # noqa: E402

# cuda-cccl, cuda-coop and cuda-stf are separate distributions that contribute to
# the shared ``cuda`` namespace; Autodoc imports from whichever are present.
for _pkg in (
    "../../python/cuda_cccl",
    "../../python/cuda_coop",
    "../../python/cuda_stf",
):
    _path = os.path.abspath(os.path.join(os.path.dirname(__file__), _pkg))
    if os.path.exists(_path):
        sys.path.insert(0, _path)

# -- Project information -----------------------------------------------------

project = "CCCL Python Libraries"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# The directory this build will be served from: "unstable" or an exact
# MAJOR.MINOR.PATCH. gen_python_docs.bash validates the value and exports it;
# falling back to "unstable" keeps a bare local build coherent.
release = os.environ.get("SPHINX_CCCL_VER", "unstable")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "coop_visualization",  # Interactive cooperative primitives with RST fallbacks
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Shared with the C++ build: search.html and other theme overrides.
templates_path = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_templates"))
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.pytest_cache",
    "**/__pycache__",
    "*.pyc",
    "*.pyo",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

_static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_static"))
html_static_path = [_static_dir] if os.path.exists(_static_dir) else []

_logo = os.path.join(_static_dir, "nvidia-logo.png")
if os.path.exists(_logo):
    html_logo = _logo

html_title = "CCCL Python Libraries"

# Where this site is served from. The workflow sets it from the repository, so a
# fork's switcher fetches the fork's manifest rather than production's.
_site_root = os.environ.get(
    "CCCL_DOCS_SITE_URL", registry.load()["site"]["base_url"]
).rstrip("/")

# This product's namespace, which holds its switcher manifest and versions.
_component_root = f"{_site_root}/python/"

# Cross-references such as :doc:`cpp:cudax/stf` resolve against the C++
# inventory, so a renamed or removed C++ page fails this build instead of
# leaving a dead link. See registry.intersphinx_mapping.
intersphinx_mapping = registry.intersphinx_mapping("python", _site_root)

# Pages are served from <component root>/<version>/, so the version is part of
# the canonical URL; without it every version claims the same address.
html_baseurl = f"{_component_root}{release}/"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/cccl",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "sidebar_includehidden": True,
    "collapse_navigation": False,
    # A reader landing on unstable/ must not mistake development documentation
    # for a release.
    **(
        {
            "announcement": (
                "This is the <strong>development</strong> documentation, built "
                "from the latest commit on <code>main</code>."
            )
        }
        if release == "unstable"
        else {}
    ),
    "switcher": {
        # The component root, not html_baseurl: the manifest lists every version,
        # so it cannot live inside one of them.
        "json_url": f"{_component_root}nv-versions.json",
        # Must equal the directory this build is served from, so the switcher
        # highlights the entry the reader is actually on.
        "version_match": release,
    },
}

# -- Options for extensions --------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_type_hints = "description"
autodoc_type_aliases = {
    "Operator": "Operator",
    **{
        name: name
        for name in (
            "BlockExchangeMode",
            "BlockGroup",
            "BlockLoadStoreAlgorithm",
            "CommonNumericScalar",
            "CommonShuffleMode",
            "CommonThreadDataLike",
            "IntegerValue",
            "IntegralScalar",
            "MemoryGroup",
            "PortableNumericScalar",
            "PortableThreadDataLike",
            "ReduceAlgorithm",
            "ReduceOperator",
            "ReductionGroup",
            "ScalarShuffleMode",
            "ScanAlgorithm",
            "ScanOperator",
            "SignedIntegerScalar",
            "ThreadDataLike",
            "ValidItems",
            "WarpGroup",
            "WarpLoadStoreAlgorithm",
        )
    },
}

primary_domain = "py"

# Dependencies unavailable on the documentation runner. A mock stands in for a
# dependency; the documented module itself must stay real, or Autodoc renders an
# empty API page and the build still succeeds.
autodoc_mock_imports = [
    # Render real cooperative markers without initializing an optional compiler.
    "numba_cuda_mlir",
    "cuda.coop.numba_mlir._compiler._activation",
    "numba",
    "numba.core",
    "numba.core.cgutils",
    "numba.core.extending",
    "numba.core.typing",
    "numba.core.typing.ctypes_utils",
    "numba.core.typing.templates",
    "numba.cuda",
    "numba.cuda.cudadecl",
    "numba.cuda.dispatcher",
    "numba.extending",
    "numba.types",
    "cuda.bindings",
    "cuda.bindings.driver",
    "cuda.bindings.runtime",
    "cuda.core",
    "cuda.core.experimental",
    "cuda.core.experimental._utils",
    "cuda.core.experimental._utils.cuda_utils",
    "cuda.pathfinder",
    "llvmlite",
    "llvmlite.ir",
    # numpy is installed as a real dependency (see requirements.txt)
    "numpydoc_test_module",  # Mock to avoid import errors
    "cupy",
    "cuda.compute._bindings",
    "cuda.compute._bindings_impl",
    # STF's public API lives in a compiled Cython extension that is not built at
    # docs time; mock it so the pure-Python helper layers in stf_api.rst
    # (task_graph, interop.numba, interop.pytorch) can still be imported.
    "cuda.stf._experimental._stf_bindings",
    "cuda.stf._experimental._stf_bindings_impl",
]

autosummary_imported_members = False
autosummary_generate = True
autoclass_content = "class"

extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}

copybutton_prompt_text = ">>> |$ |# "


def setup(app):
    custom_css = os.path.join(_static_dir, "custom.css")
    if os.path.exists(custom_css):
        app.add_css_file("custom.css")
