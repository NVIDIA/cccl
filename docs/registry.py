"""Read docs/registry.yml, the list of products whose docs are published."""

import os
import pathlib
import re

import yaml

HERE = pathlib.Path(__file__).resolve().parent
REGISTRY = HERE / "registry.yml"

# Where build_site.py puts the site together. It passes the path to the product
# builds in CCCL_DOCS_SITE_DIR.
SITE = pathlib.Path(os.environ.get("CCCL_DOCS_SITE_DIR", HERE / "_site"))

# Where the product build scripts write their output.
ARTIFACTS = HERE / "_build" / "artifacts" / "docs"

# Docs are published as "unstable" (built from main) or under an exact release
# version. There is no MAJOR.MINOR folder whose contents would keep changing.
DEVELOPMENT = "unstable"
VERSION = re.compile(r"^(unstable|[0-9]+\.[0-9]+\.[0-9]+)\Z")


def load():
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def products():
    return load()["products"]


def product(key):
    for entry in products():
        if entry["key"] == key:
            return entry
    known = ", ".join(entry["key"] for entry in products())
    raise SystemExit(f"error: no product {key!r} in registry.yml (it has {known})")


def site_url():
    """The URL the site is served from. CI sets it, so a fork links to itself."""
    url = os.environ.get("CCCL_DOCS_SITE_URL") or load()["site"]["base_url"]
    return url.rstrip("/")


def resolve_tag(tag):
    """Map a release tag to (product, version), e.g. "v3.5.0" to ("cpp", "3.5.0")."""
    tag = (tag or "").strip()
    # Longest prefix first, in case one product's prefix starts with another's.
    by_prefix = sorted(products(), key=lambda e: len(e["tag_prefix"]), reverse=True)
    for entry in by_prefix:
        if tag.startswith(entry["tag_prefix"]):
            version = tag[len(entry["tag_prefix"]) :]
            if VERSION.match(version) and version != DEVELOPMENT:
                return entry["key"], version
            break
    expected = " or ".join(f"{e['tag_prefix']}MAJOR.MINOR.PATCH" for e in products())
    raise SystemExit(f"error: {tag!r} is not a release tag; expected {expected}")


def intersphinx_mapping(key, base_url):
    """Intersphinx entries for the products built before `key`.

    Products are built in registry order, so only the ones listed above `key`
    have an objects.inv yet. Pass key=None to map every product.
    """
    if key is not None:
        product(key)  # an unknown key is an error
    mapping = {}
    for entry in products():
        name = entry["key"]
        if name == key:
            break
        # Link to the unstable docs. The products are released separately, so
        # there is no matching version to link to.
        url = f"{base_url}/{name}/{DEVELOPMENT}/"
        # Read objects.inv from this build if it is there, so that links to new
        # pages work before they are published. Otherwise download it from url.
        inventory = None
        for root in (SITE, ARTIFACTS):
            candidate = root / name / DEVELOPMENT / "objects.inv"
            if candidate.is_file():
                inventory = str(candidate)
                break
        mapping[name] = (url, inventory)
        print(f"  intersphinx {name}: {inventory or url}")
    return mapping
