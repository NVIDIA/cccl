# Sphinx configuration for the CCCL Python documentation.

import os
import sys
from datetime import datetime

# Add extension directory to path, and docs/ for registry.py
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.abspath(".."))

import registry  # noqa: E402

# Add Python CCCL distributions to the shared ``cuda`` namespace for autodoc.
for _pkg in (
    "../../python/cuda_cccl",
    "../../python/cuda_coop_cutlass",
    "../../python/cuda_stf",
):
    python_package_path = os.path.abspath(_pkg)
    if os.path.exists(python_package_path):
        sys.path.insert(0, python_package_path)

# Note: numpy is installed as a real dependency (see requirements.txt)
# This avoids issues with type annotations using union syntax (ndarray | type)

# -- Project information -----------------------------------------------------

project = "CCCL Python Libraries"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# gen_python_docs.bash sets this to the version being built.
release = os.environ.get("SPHINX_CCCL_VER", "unstable")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "myst_parser",  # MyST parser for markdown support
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "coop_visualization",  # Interactive cooperative primitives with RST fallbacks
]

# Add support for .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["../common/_templates"]

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

html_logo = "../common/_static/nvidia-logo.png"

# Each version is published at <site>/python/<version>/.
html_baseurl = f"{registry.site_url()}/python/{release}/"

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
    "switcher": {
        "json_url": f"{registry.site_url()}/python/nv-versions.json",
        "version_match": release,
    },
}

# Tell readers that unstable is not a release.
if release == "unstable":
    html_theme_options["announcement"] = (
        "This is the <strong>development</strong> documentation, built from "
        "the latest commit on <code>main</code>."
    )

html_static_path = ["_static", "../common/_static"]

html_title = "CCCL Python Libraries"

# -- Options for extensions --------------------------------------------------

# Intersphinx mapping. Also links to the C++ docs, as in :doc:`cpp:cudax/stf`.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    **registry.intersphinx_mapping("python", registry.site_url()),
}

# MyST parser configuration
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

# Enable type hints to be shown in the documentation
autodoc_type_hints = "description"
autodoc_type_aliases = {
    "Operator": "Operator",
}

# Set Python domain primary for intersphinx
primary_domain = "py"

# Mock imports for Python documentation - these modules may not be installed
autodoc_mock_imports = [
    "cutlass",
    "cuda.coop.cutlass._compiler._runtime",
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
    # STF's public API lives in a compiled Cython extension that is not built
    # at docs time; mock it so the pure-Python helper layers in stf_api.rst
    # (task_graph, interop.numba, interop.pytorch) can still be imported by autodoc.
    "cuda.stf._experimental._stf_bindings",
    "cuda.stf._experimental._stf_bindings_impl",
]

# External links configuration
extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}

copybutton_prompt_text = ">>> |$ |# "
autosummary_imported_members = False
autosummary_generate = True
autoclass_content = "class"
