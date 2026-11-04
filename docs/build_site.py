#!/usr/bin/env python3
"""Build the root project, build each product, and assemble the site.

    python3 docs/build_site.py                      everything, into docs/_site/
    python3 docs/build_site.py --product python     one product, plus the root
    python3 docs/build_site.py --label 3.5.0 --product cpp      a release
    python3 docs/build_site.py --skip-build         re-assemble existing output

A root Sphinx project indexes the products, and an assembly step composes the
published site from each product's output. The root knows which products exist
and where they go, not how any of them is built.

The assembled tree is what the deploy publishes:

    _site/
      index.html             the root project: the landing index
      _static/               its theme assets
      cpp/
        index.html           redirect to unstable/
        nv-versions.json     the versions the switcher offers
        unstable/            one version directory per published version
      python/
        ...
"""

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import registry  # noqa: E402

DEFAULT_SITE = HERE / "_site"
ARTIFACTS = registry.ARTIFACTS
# The virtual environment the product build scripts create and share.
VENV = HERE / "env"


def run(command, cwd=HERE):
    """Run a command. A list runs directly, so its paths need no quoting; a
    string, such as a registry build command, goes through the shell."""
    shown = command if isinstance(command, str) else shlex.join(map(str, command))
    print(f"  $ {shown}")
    result = subprocess.run(
        command, shell=isinstance(command, str), cwd=cwd, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"error: command failed: {shown}")


def python_bin():
    """The interpreter that has Sphinx, which is the one in docs/env."""
    candidate = VENV / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def ensure_dependencies():
    """Create the shared virtual environment if nothing has yet.

    The workflows create it before running this script, and each product's build
    script creates it if missing, so in practice this only matters for
    --skip-build on a checkout that has none.
    """
    if (VENV / "bin" / "python").exists():
        return
    print("creating the documentation virtual environment")
    run([sys.executable, "-m", "venv", VENV])
    pip = [VENV / "bin" / "python", "-m", "pip", "install", "--quiet"]
    run([*pip, "-r", HERE / "requirements.txt"])


def seed_from_branch(branch, site_dir, published_prefix="docs", repo=None):
    """Start the assembly from what is already published.

    The deploy recreates the published branch from the artifact (force_orphan),
    which keeps it at one commit. Seeding every published version in first makes
    the artifact the whole site, so recreating the branch loses nothing.
    """
    # From the repository root, not from docs/: in `git archive <rev>:<path>`
    # the path is resolved against the current directory's prefix, so the same
    # command run from docs/ would look for docs/docs and silently find nothing.
    repo = HERE.parent if repo is None else pathlib.Path(repo)

    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{branch}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print(f"  {branch} does not exist yet -- first publication, nothing to seed")
        return 0

    listing = subprocess.run(
        ["git", "archive", f"{branch}:{published_prefix}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if listing.returncode != 0:
        raise SystemExit(
            f"error: cannot read {published_prefix}/ on {branch}.\n"
            "       Carrying on would publish a site missing every version\n"
            "       already on it, and the deploy recreates the branch."
        )

    tar = pathlib.Path(site_dir) / ".seed.tar"
    tar.write_bytes(listing.stdout)
    shutil.unpack_archive(str(tar), str(site_dir), format="tar")
    tar.unlink()

    seeded = sorted(
        p.name
        for p in pathlib.Path(site_dir).iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )
    print(f"  seeded from {branch}: {', '.join(seeded) or '(nothing)'}")
    return len(seeded)


def build_root(site_dir):
    """The unversioned project at the site root.

    Built with sphinx-build like any other project; it just has no version, so
    nothing stamps it and no switcher is drawn on it.
    """
    root_source = HERE / registry.load()["root"]["source_root"]
    print(f"building the root project from {root_source.relative_to(HERE)}/")
    # -d keeps Sphinx's doctree cache out of the output. Without it the cache
    # lands in the published tree and gets deployed with the site.
    sphinx = [python_bin(), "-m", "sphinx.cmd.build", "-b", "html", "-q"]
    run(
        [*sphinx, "-W", "--keep-going", "-d", HERE / "_build" / "root-doctrees"]
        + [root_source, site_dir]
    )


def build_product(entry, label):
    """One product, via the build command the registry names for it."""
    print(f"building {entry['key']} at {label}")
    run(entry["build"].format(label=label))


def place_product(entry, label, site_dir):
    """Copy a product's build output into the assembled site.

    Everything the product produced moves as it is -- the version directory, the
    manifest the switcher reads, the landing redirect. The assembly does not
    rewrite any of it; it decides where it goes.
    """
    source = ARTIFACTS / entry["key"]
    if not source.is_dir():
        raise SystemExit(
            f"error: {entry['key']} built nothing at {source}.\n"
            f"       Expected the build command to leave output there."
        )

    target = pathlib.Path(site_dir) / entry["key"]
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        destination = target / item.name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
        else:
            shutil.copy2(item, destination)

    pages = (
        len(list((target / label).rglob("*.html"))) if (target / label).is_dir() else 0
    )
    print(f"  placed {entry['key']}/ -- {label}: {pages} pages")
    return pages


def check_manifests(site_dir):
    """Every version directory in the assembled site must be in its manifest.

    The site and the switcher are produced by different things -- the version
    directories by the assembly, the manifest by each product's build reading
    the published branch -- so they can disagree, and the disagreement is
    invisible. Every page renders; the dropdown just does not offer a version
    whose pages are sitting right there.
    """
    for entry in registry.products():
        root = pathlib.Path(site_dir) / entry["key"]
        manifest = root / "nv-versions.json"
        if not root.is_dir() or not manifest.exists():
            continue

        listed = {
            e["version"] for e in json.loads(manifest.read_text(encoding="utf-8"))
        }
        present = {
            p.name
            for p in root.iterdir()
            if p.is_dir() and registry.VERSION.match(p.name)
        }
        missing = present - listed
        if missing:
            raise SystemExit(
                f"error: {entry['key']}/nv-versions.json does not list "
                f"{', '.join(sorted(missing))}.\n"
                f"       Those versions are published at {root}/ and a reader\n"
                "       cannot reach them: the switcher is the only thing that\n"
                "       links versions together."
            )
        print(
            f"  {entry['key']}: manifest lists all {len(present)} published version(s)"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--label", default=registry.DEVELOPMENT, help="version to build the products at"
    )
    parser.add_argument(
        "--product",
        action="append",
        dest="products",
        help="build only this product (repeatable)",
    )
    parser.add_argument(
        "--site", default=str(DEFAULT_SITE), help="directory to assemble into"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="assemble from existing build output"
    )
    parser.add_argument(
        "--seed-from", default="", help="branch whose published site to start from"
    )
    args = parser.parse_args(argv)

    if not registry.VERSION.match(args.label):
        raise SystemExit(
            f"error: {args.label!r} is not a publishable version.\n"
            "       Expected 'unstable' or an exact MAJOR.MINOR.PATCH release."
        )

    wanted = args.products or [p["key"] for p in registry.products()]
    entries = [registry.product(key) for key in wanted]

    site_dir = pathlib.Path(args.site)
    site_dir.mkdir(parents=True, exist_ok=True)

    # Each product builds in a subprocess, and its conf.py resolves
    # cross-references against the tree being assembled here rather than against
    # the live site, which is always a publish behind. The directory travels by
    # environment because conf.py imports registry, not this script.
    os.environ["CCCL_DOCS_SITE_DIR"] = str(site_dir.resolve())

    ensure_dependencies()

    if args.seed_from:
        seed_from_branch(args.seed_from, site_dir)

    total = 0
    for entry in entries:
        if not args.skip_build:
            build_product(entry, args.label)
        total += place_product(entry, args.label, site_dir)

    # Last, not first. The root project resolves cross-references against the
    # products' inventories, and place_product is what puts them in _site/. Built
    # first, it would see only what the seed carried -- and on a first
    # publication, with nothing seeded and nothing published, nothing at all.
    build_root(site_dir)

    # GitHub Pages runs Jekyll unless this exists, and Jekyll drops directories
    # beginning with an underscore -- which is every Sphinx asset directory.
    (site_dir / ".nojekyll").touch()

    check_manifests(site_dir)

    print()
    print(f"assembled {site_dir}")
    print(f"  root project + {len(entries)} product(s) at {args.label}, {total} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
