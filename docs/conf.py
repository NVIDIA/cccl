# CCCL Documentation Configuration File
# Generated to replace repo-docs with direct Sphinx usage

import os
import sys
from datetime import datetime

# Add extension directory to path
sys.path.insert(0, os.path.abspath("_ext"))

# Add Python CCCL package to path for autodoc
python_package_path = os.path.abspath("../python/cuda_cccl")
if os.path.exists(python_package_path):
    sys.path.insert(0, python_package_path)

# Note: numpy is installed as a real dependency (see requirements.txt)
# This avoids issues with type annotations using union syntax (ndarray | type)

# -- Project information -----------------------------------------------------

project = "CUDA Core Compute Libraries"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# Version information
try:
    with open("VERSION.md", "r") as f:
        version = f.read().strip()
except Exception:
    version = "latest"

release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "myst_parser",  # MyST parser for markdown support
    "breathe",  # For Doxygen integration - has built-in embed:rst support
    # "exhale",  # Disabled - causing build timeouts, API docs handled by breathe
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "nbsphinx",
    # "rst_processor",  # Disabled - breathe handles embed:rst natively
    "auto_api_generator",  # Automatically generate API reference pages from Doxygen XML
]

# Breathe configuration for Doxygen integration
breathe_projects = {
    "cub": "_build/doxygen/cub/xml",
    "thrust": "_build/doxygen/thrust/xml",
    "libcudacxx": "_build/doxygen/libcudacxx/xml",
    "cudax": "_build/doxygen/cudax/xml",
}

breathe_default_project = "cub"
breathe_default_members = ("members", "undoc-members")
breathe_show_enumvalue_initializer = True
breathe_domain_by_extension = {"cuh": "cpp", "h": "cpp", "hpp": "cpp"}

# Configure cpp domain to handle cub namespace
cpp_index_common_prefix = ["cub::"]

# Preprocessor definitions for Breathe to handle CCCL macros
cpp_id_attributes = [
    "__device__",
    "__host__",
    "__global__",
    "__forceinline__",
    "_CCCL_HOST_DEVICE",
    "_CCCL_DEVICE",
    "_CCCL_HOST",
    "_CCCL_FORCEINLINE",
    "_CCCL_API",
    "_CCCL_HOST_API",
    "_CCCL_DEVICE_API",
    "_CCCL_NODEBUG_API",
    "_CCCL_NODEBUG_HOST_API",
    "_CCCL_NODEBUG_DEVICE_API",
    "_CCCL_TRIVIAL_API",
    "_CCCL_TRIVIAL_HOST_API",
    "_CCCL_TRIVIAL_DEVICE_API",
]
cpp_paren_attributes = ["__declspec", "__align__"]

# Add support for .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = [
    "_build",
    "_repo",
    "tools",
    "VERSION.md",
    "Thumbs.db",
    ".DS_Store",
    "env/**",  # Virtual environment
    "**/.pytest_cache",
    "**/__pycache__",
    "*.pyc",
    "*.pyo",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_logo = "_static/nvidia-logo.png"
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
}

html_static_path = ["_static"] if os.path.exists("_static") else []

# Images directory
if os.path.exists("img"):
    html_static_path.append("img")

html_title = "CUDA Core Compute Libraries"

# -- Options for extensions --------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
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
autodoc_typehints = "description"
autodoc_type_aliases = {}

# Set Python domain primary for intersphinx
primary_domain = "py"

# Mock imports for Python documentation - these modules may not be installed
autodoc_mock_imports = [
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
    "pynvjitlink",
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
    "cuda.cccl.parallel.experimental._bindings",
    "cuda.cccl.parallel.experimental._bindings_impl",
]

# External links configuration
extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}


# Exhale not used - API documentation is handled directly through breathe directives

# Napoleon configuration (handles NumPy-style docstrings)
# Note: numpydoc settings removed as Napoleon is used instead

# Config copybutton
copybutton_prompt_text = ">>> |$ |# "
autosummary_imported_members = False
autosummary_generate = True
autoclass_content = "class"


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
