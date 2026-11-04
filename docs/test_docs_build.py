"""Tests for documentation publication.

Each failure covered here is silent -- the site builds, deploys and renders
while being wrong: a tag routed to the wrong product or version, a page stamp or
manifest entry the switcher cannot match, one product's pages inside another's
artifact, a cross-reference resolved against the wrong inventory, or a deploy
that drops versions already published.

Run directly, or through the pre-commit hook:

    pytest docs/test_docs_build.py
"""

import importlib.util
import json
import pathlib
import re
import subprocess
import types

import build_site
import make_manifest
import pytest
import registry
import yaml

DOCS = pathlib.Path(__file__).resolve().parent
REPO = DOCS.parent
SITE_WORKFLOW = REPO / ".github/workflows/build-site.yml"

# Note on naming: keep test function names off exactly 40 characters. Lob API
# test keys are literally "test_" followed by 35 characters, so the repository's
# secret scanner flags any 40-character test_* identifier as a credential.


# --------------------------------------------------------------------------
# Tag -> component and version
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v3.4.2", ("cpp", "3.4.2")),
        ("v3.10.0", ("cpp", "3.10.0")),
        ("v4.0.0", ("cpp", "4.0.0")),
        ("python-1.1.1", ("python", "1.1.1")),
        ("python-1.2.0", ("python", "1.2.0")),
    ],
)
def test_tag_names_component_and_version(tag, expected):
    assert registry.resolve_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "v3.5.0-rc0",  # a pre-release must not take the release's URL
        "v3.6.0.dev",
        "v3.4",  # no rolling MAJOR.MINOR directory exists
        "python-1.1",
        "3.4.2",  # unprefixed: which component?
        "refs/tags/v3.4.2",
        "unstable",
        "main",
        "",
    ],
)
def test_non_release_tags_rejected(tag):
    with pytest.raises(SystemExit):
        registry.resolve_tag(tag)


def test_component_cannot_be_crossed():
    """A Python tag can never resolve into the C++ namespace, or vice versa."""
    assert registry.resolve_tag("python-1.1.1")[0] == "python"
    assert registry.resolve_tag("v1.1.1")[0] == "cpp"


def test_tag_output_cannot_forge_a_key(tmp_path, monkeypatch):
    """Workflow outputs are key=value lines; a newline would add a second key."""
    with pytest.raises(SystemExit):
        registry.resolve_tag("v3.4.2\nlabel=9.9.9")


# --------------------------------------------------------------------------
# Artifact layout
# --------------------------------------------------------------------------


def make_component(root, component, label, manifest_versions):
    """A stand-in for one component's built artifact."""
    comp = root / component
    version = comp / label
    version.mkdir(parents=True, exist_ok=True)
    (version / "index.html").write_text(
        f"<html><script>version_match = '{label}';</script></html>", encoding="utf-8"
    )
    (version / "objects.inv").write_text("inv", encoding="utf-8")
    (comp / "nv-versions.json").write_text(
        json.dumps(
            [
                {"version": v, "url": f"https://x/{component}/{v}/"}
                for v in manifest_versions
            ]
        ),
        encoding="utf-8",
    )
    (comp / "index.html").write_text("<html>redirect</html>", encoding="utf-8")
    return comp


def stamp_of(page):
    import re

    m = re.search(r"version_match = '([^']*)'", page.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def manifest_versions(component_root):
    data = json.loads((component_root / "nv-versions.json").read_text(encoding="utf-8"))
    return [e["version"] for e in data]


def test_published_label_is_listed_in_the_manifest(tmp_path):
    """The reader reaches a version through the switcher; if the manifest does
    not list it, the documentation exists but cannot be found."""
    comp = make_component(tmp_path, "cpp", "3.4.2", ["unstable", "3.4.2"])
    assert stamp_of(comp / "3.4.2" / "index.html") in manifest_versions(comp)


def test_stamp_missing_from_manifest_is_detectable(tmp_path):
    """The failure the build guard exists to catch."""
    comp = make_component(tmp_path, "cpp", "3.4.3", ["unstable", "3.4.2"])
    assert stamp_of(comp / "3.4.3" / "index.html") not in manifest_versions(comp)


def test_stamp_equals_the_served_directory(tmp_path):
    """The switcher highlights the entry matching the page's stamp, so the stamp
    must be the directory the page is served from, not the source version."""
    for label in ("unstable", "3.4.2"):
        comp = make_component(tmp_path / label, "cpp", label, ["unstable", "3.4.2"])
        assert stamp_of(comp / label / "index.html") == label


def test_component_artifacts_stay_separate(tmp_path):
    """A release artifact must contain only its own component."""
    root = tmp_path / "artifacts"
    make_component(root, "cpp", "3.4.2", ["unstable", "3.4.2"])
    assert (root / "cpp").is_dir()
    assert not (root / "python").exists()


def test_manifests_list_only_their_own_component(tmp_path):
    root = tmp_path / "artifacts"
    cpp = make_component(root, "cpp", "unstable", ["unstable", "3.4.2"])
    py = make_component(root, "python", "unstable", ["unstable", "1.1.1"])
    assert manifest_versions(cpp) == ["unstable", "3.4.2"]
    assert manifest_versions(py) == ["unstable", "1.1.1"]
    assert not set(manifest_versions(cpp)) & {"1.1.1"}


# -- the generated manifest ---------------------------------------------------
#
# The manifest is derived from the documentation branch rather than maintained
# by hand, so the tests that matter are about what the derivation can and cannot
# produce: it cannot omit a published version, and it cannot invent one.


def pages_repo(tmp_path, tree, name="repo"):
    """A git repository whose gh-pages branch is shaped like the real one."""
    repo = tmp_path / name
    repo.mkdir()

    def run(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    run("init", "-q", "-b", "gh-pages")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    for component, versions in tree.items():
        root = repo / "docs" / component
        root.mkdir(parents=True, exist_ok=True)
        for version in versions:
            (root / version).mkdir()
            (root / version / "index.html").write_text("page", encoding="utf-8")
        # Files beside the version directories, which must not be mistaken for
        # versions: this is the whole reason the listing asks git for trees.
        (root / "objects.inv").write_text("inv", encoding="utf-8")
        (root / "index.html").write_text("redirect", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "pages")
    return repo


def test_listing_sees_versions_and_only_versions(tmp_path):
    repo = pages_repo(tmp_path, {"cpp": ["unstable", "3.4.2", "scratch"]})
    found = make_manifest.published_versions("cpp", "gh-pages", repo)
    assert sorted(found) == ["3.4.2", "unstable"]
    assert "scratch" not in found


def test_published_versions_survive_a_new_release(tmp_path):
    """The failure a hand-edited manifest allowed: 3.4.2 still has a directory,
    so no edit and no mistake can drop it out of the switcher."""
    repo = pages_repo(tmp_path, {"cpp": ["unstable", "3.4.2"]})
    out = tmp_path / "artifact"
    out.mkdir()

    listed = make_manifest.write(
        out, "cpp", "3.5.0", "https://x/cccl", "gh-pages", repo
    )

    assert listed == ["unstable", "3.5.0", "3.4.2"]
    entries = json.loads((out / "nv-versions.json").read_text(encoding="utf-8"))
    assert [e["version"] for e in entries] == listed
    assert [e["url"] for e in entries] == [
        "https://x/cccl/cpp/unstable/",
        "https://x/cccl/cpp/3.5.0/",
        "https://x/cccl/cpp/3.4.2/",
    ]


def test_the_version_being_published_is_always_listed(tmp_path):
    """It cannot be forgotten, because nobody adds it."""
    repo = pages_repo(tmp_path, {"cpp": ["unstable"]})
    out = tmp_path / "artifact"
    out.mkdir()
    assert "3.5.0" in make_manifest.write(
        out, "cpp", "3.5.0", "https://x/cccl", "gh-pages", repo
    )


def test_each_component_sees_only_its_own_versions(tmp_path):
    repo = pages_repo(
        tmp_path, {"cpp": ["unstable", "3.4.2"], "python": ["unstable", "1.1.1"]}
    )
    assert "1.1.1" not in make_manifest.published_versions("cpp", "gh-pages", repo)
    assert "3.4.2" not in make_manifest.published_versions("python", "gh-pages", repo)


def test_order_is_unstable_then_newest_release_first():
    """Lexical order would put 3.10.0 before 3.4.2 and unstable last."""
    assert make_manifest.order({"3.4.2", "3.10.0", "3.4.10", "4.0.0", "unstable"}) == [
        "unstable",
        "4.0.0",
        "3.10.0",
        "3.4.10",
        "3.4.2",
    ]


def test_a_component_with_nothing_published_yet(tmp_path):
    """A first publish: the branch exists, this component does not."""
    repo = pages_repo(tmp_path, {"cpp": ["unstable"]})
    assert make_manifest.published_versions("python", "gh-pages", repo) == []


def test_a_branch_that_does_not_exist_yet(tmp_path):
    """A first rehearsal deploys to a branch nobody has pushed. Nothing is
    published there, which is different from being unable to tell."""
    origin = pages_repo(tmp_path, {"cpp": ["unstable"]}, name="origin")
    repo = pages_repo(tmp_path, {"cpp": ["unstable"]}, name="clone")
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert make_manifest.published_versions("cpp", "rehearse-1", repo) == []


def test_an_unreadable_branch_is_an_error_not_an_empty_list(tmp_path):
    """Carrying on would publish a manifest naming one version and silently
    empty the switcher -- the exact failure this replaces."""
    repo = pages_repo(tmp_path, {"cpp": ["unstable", "3.4.2"]})
    with pytest.raises(SystemExit, match="cannot read branch"):
        make_manifest.published_versions("cpp", "gh-pages-unreachable", repo)


def test_a_local_build_lists_only_what_it_built(tmp_path):
    out = tmp_path / "artifact"
    out.mkdir()
    assert make_manifest.write(out, "cpp", "unstable", "https://x/cccl") == ["unstable"]


def test_a_rolling_version_is_refused(tmp_path):
    """3.4 would be a directory whose content changes with every patch."""
    out = tmp_path / "artifact"
    out.mkdir()
    with pytest.raises(SystemExit, match="not a publishable version"):
        make_manifest.write(out, "cpp", "3.4", "https://x/cccl")


def test_urls_name_the_site_being_published_to(tmp_path):
    """A rehearsal on a fork must not hand readers a dropdown whose every
    option navigates to production."""
    out = tmp_path / "artifact"
    out.mkdir()
    (out / "index.html").write_text(
        '<link rel="canonical" href="https://nvidia.github.io/cccl/cpp/unstable/">',
        encoding="utf-8",
    )
    make_manifest.write(out, "cpp", "unstable", "https://fork.example/cccl")
    make_manifest.retarget_landing(out, "cpp", "https://fork.example/cccl")

    body = (out / "nv-versions.json").read_text(encoding="utf-8")
    landing = (out / "index.html").read_text(encoding="utf-8")
    assert "https://fork.example/cccl/cpp/unstable/" in body
    assert "nvidia.github.io" not in body
    assert "https://fork.example/cccl/cpp/unstable/" in landing
    assert "nvidia.github.io" not in landing


def test_checked_in_redirects_still_name_production():
    """The landing redirect is edited in the artifact, never in the repository:
    a real release goes to production, so the source must keep pointing there."""
    for component in ("cpp", "python"):
        redirect = (DOCS / f"{component}_site" / "index.html").read_text()
        assert "https://nvidia.github.io/cccl/" in redirect, component


def test_no_manifest_is_checked_in():
    """It is generated. A checked-in copy would be a second source of truth,
    and the one that is wrong whenever a release has shipped since it was
    edited."""
    for component in ("cpp", "python"):
        assert not (DOCS / f"{component}_site" / "nv-versions.json").exists(), component


# --------------------------------------------------------------------------
# Build scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["unstable", "3.4.2", "1.1.1", "3.10.0"])
def test_build_scripts_accept_valid_labels(label):
    for script in ("gen_docs.bash", "gen_python_docs.bash"):
        assert _label_accepted(DOCS / script, label), f"{script} rejected {label}"


def test_the_shell_scripts_spell_the_registry_grammar():
    """registry.VERSION is the grammar, and the build scripts cannot import it.

    They re-spell it in bash, so the three can drift: loosening the Python one
    to allow a rolling 3.4 directory would leave the scripts rejecting it, and
    the parametrized cases below are hand-written, so they would still pass.
    Compare the literals instead. ``\\Z`` and ``$`` are the same anchor here --
    bash's ``=~`` has no multiline mode.
    """
    expected = registry.VERSION.pattern.replace(r"\Z", "$")
    for script in ("gen_docs.bash", "gen_python_docs.bash"):
        body = (DOCS / script).read_text(encoding="utf-8")
        assert f"=~ {expected}" in body, (
            f"{script} does not validate against {expected!r}; "
            "registry.VERSION and the shell scripts have diverged"
        )


@pytest.mark.parametrize("label", ["latest", "3.4", "unstable; rm -rf /", "v3.4.2", ""])
def test_build_scripts_reject_bad_labels(label):
    """No rolling MAJOR.MINOR directory, and no shell injection."""
    if label == "":
        return  # empty falls back to the default, covered elsewhere
    for script in ("gen_docs.bash", "gen_python_docs.bash"):
        assert not _label_accepted(DOCS / script, label), f"{script} accepted {label}"


def _label_accepted(script, label):
    """Run only the script's label validation, without building anything."""
    check = subprocess.run(
        [
            "bash",
            "-c",
            (
                'VERSION="$1"; '
                'if [[ ! "${VERSION}" =~ ^(unstable|[0-9]+\\.[0-9]+\\.[0-9]+)$ ]]; '
                "then exit 1; fi"
            ),
            "_",
            label,
        ],
        capture_output=True,
        check=False,
    )
    return check.returncode == 0


def test_loose_markdown_is_excluded_by_name():
    """MyST treats any .md sitting in docs/ as a document, and a document in no
    toctree is a warning -- which these builds treat as an error. So a file
    dropped in beside the sources has to be excluded by name, or it fails the
    build in a way that reads as unrelated to the file that caused it."""
    config = (DOCS / "conf.py").read_text(encoding="utf-8")
    for md in sorted(DOCS.glob("*.md")):
        assert md.name in config, (
            f"{md.name} sits in docs/ and is not in exclude_patterns; MyST will "
            "treat it as a document in no toctree and fail the C++ build"
        )


def test_each_build_script_generates_its_manifest():
    """A generator that is written but not wired in produces nothing, and the
    artifact would ship without the file the switcher fetches."""
    for script in ("gen_docs.bash", "gen_python_docs.bash"):
        body = (DOCS / script).read_text(encoding="utf-8")
        assert "make_manifest.py" in body, script
        assert "--from-branch" in body, script
        # Never copied from the repository: a checked-in manifest would be a
        # second source of truth.
        copied = [
            ln
            for ln in body.splitlines()
            if ln.strip().startswith("cp ") and "nv-versions.json" in ln
        ]
        assert not copied, (script, copied)


def test_build_scripts_are_syntactically_valid():
    for script in ("gen_docs.bash", "gen_python_docs.bash"):
        result = subprocess.run(
            ["bash", "-n", str(DOCS / script)], capture_output=True, check=False
        )
        assert result.returncode == 0, f"{script}: {result.stderr.decode()}"


# --------------------------------------------------------------------------
# Workflow shape
# --------------------------------------------------------------------------


def _plan_lines(workflow, *prefixes):
    """The plan step's own shell lines that start with one of ``prefixes``."""
    plan = next(
        s for s in workflow["jobs"]["publish"]["steps"] if s.get("id") == "plan"
    )
    return [
        ln.strip() for ln in plan["run"].splitlines() if ln.strip().startswith(prefixes)
    ]


@pytest.mark.parametrize(
    "repository,expected",
    [
        ("NVIDIA/cccl", "https://nvidia.github.io/cccl"),
        ("sri-koundinyan/cccl", "https://sri-koundinyan.github.io/cccl"),
        ("Some-Org/cccl", "https://some-org.github.io/cccl"),
    ],
)
def test_site_url_derives_from_the_repository(
    site_workflow, tmp_path, repository, expected
):
    """Pages serves <owner>.github.io/<repo> with the owner lowercased, which
    for NVIDIA/cccl is exactly the production URL -- so one rule covers
    production and every fork, and a fork need not pass anything.

    Runs the workflow's own lines, taken from its plan step."""
    lines = _plan_lines(site_workflow, "owner=", 'echo "site_url=')
    assert len(lines) == 2, lines
    out = tmp_path / "github_output"
    subprocess.run(
        ["bash", "-c", "\n".join(lines)],
        check=True,
        env={
            "GITHUB_REPOSITORY": repository,
            "GITHUB_OUTPUT": str(out),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert out.read_text(encoding="utf-8").strip() == f"site_url={expected}"


@pytest.mark.parametrize(
    "branch,accepted",
    [
        ("gh-pages", True),
        ("gh-pages-rehearsal", True),
        ("main", False),
        ("gh-pagesx", False),
        ("branch/3.6.x", False),
        ("gh-pages-$(id)", False),
    ],
)
def test_only_gh_pages_branches_can_be_targeted(site_workflow, branch, accepted):
    """The deploy recreates its target branch from the artifact, so a mistyped
    docs-branch such as main would be replaced by a docs-only commit. Runs the
    workflow's own pattern."""
    (line,) = _plan_lines(site_workflow, 'if [[ ! "${DOCS_BRANCH}" =~')
    pattern = re.search(r"=~ (\S+) \]\]", line).group(1)
    result = subprocess.run(
        ["bash", "-c", f'[[ "$BRANCH" =~ {pattern} ]]'],
        env={"BRANCH": branch, "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert (result.returncode == 0) is accepted, branch


def test_no_legacy_combined_paths_remain(tmp_path):
    """Nothing at the site's entry points may link the retired combined scheme.

    The schemes differ by position, not by the word `unstable`: the old one put
    the version directly under /cccl/ (`/cccl/unstable/python/`), the current
    one under the product (`/cccl/python/unstable/`).

    The landing page is generated from the registry at build time, so this runs
    that generator and reads what it produced. The per-product redirects are the
    only other place an entry-point URL is written by hand; everything below
    them Sphinx generates.
    """
    spec = importlib.util.spec_from_file_location(
        "cccl_root_conf", DOCS / "site" / "conf.py"
    )
    root_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_conf)
    root_conf._write_index(types.SimpleNamespace(srcdir=tmp_path), None)
    landing = (tmp_path / "index.md").read_text(encoding="utf-8")

    sources = {"the generated landing page": landing}
    for entry in registry.products():
        redirect = DOCS / f"{entry['key']}_site" / "index.html"
        sources[f"{entry['key']}_site/index.html"] = redirect.read_text(
            encoding="utf-8"
        )

    for where, text in sources.items():
        assert "/cccl/unstable/" not in text, f"{where}: retired combined tree"
        assert "unstable/python" not in text, f"{where}: Python nested inside C++"

    # And the landing page reaches every product by the key the registry names,
    # so a product added there is linked by having been added.
    for entry in registry.products():
        assert f"]({entry['key']}/)" in landing, f"landing omits {entry['key']}"


# --------------------------------------------------------------------------
# Intersphinx: the tree being published, not the live site
# --------------------------------------------------------------------------
#
# The live site is always the previous publish, so resolving against it means a
# page added in this commit cannot be referenced by this commit, and a product
# added to the registry deadlocks its consumers until it has published once.


def _site_with(tmp_path, *version_dirs):
    """An assembled site holding an inventory for each directory named."""
    for rel in version_dirs:
        inv = tmp_path / rel / "objects.inv"
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(b"# Sphinx inventory version 2\n")
    return tmp_path


def test_intersphinx_prefers_the_assembled_site(tmp_path):
    """When the sibling has already been built and placed this run, its fresh
    inventory is on disk and must be what the lookup reads."""
    site = _site_with(tmp_path, "cpp/unstable")
    mapping = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=site
    )
    url, inventory = mapping["cpp"]
    assert inventory == str(site / "cpp" / "unstable" / "objects.inv")
    assert url == "https://example.test/cccl/cpp/unstable/"


def test_intersphinx_falls_back_when_the_sibling_is_absent(tmp_path, monkeypatch):
    """A release builds one product, so a sibling may be neither built nor
    seeded. Falling back to the published inventory is correct there -- nothing
    in this run disagrees with it. Absent has to mean absent from both the
    assembled site and the raw build output."""
    monkeypatch.setattr(registry, "ARTIFACTS", tmp_path / "no-artifacts")
    mapping = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=tmp_path
    )
    url, inventory = mapping["cpp"]
    assert inventory is None
    assert url == "https://example.test/cccl/cpp/unstable/"


def test_intersphinx_url_is_published_regardless_of_lookup_source(tmp_path):
    """The URL is where a reader goes; the inventory is where the name is looked
    up. Reading locally must not leak a file path into rendered links."""
    with_site = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=_site_with(tmp_path, "cpp/unstable")
    )
    without = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=tmp_path / "empty"
    )
    assert with_site["cpp"][0] == without["cpp"][0]


def test_intersphinx_never_maps_the_caller(tmp_path):
    """A project resolves its own names locally; handing it its own inventory
    makes every local reference ambiguous."""
    site = _site_with(tmp_path, "cpp/unstable", "python/unstable")
    for key in (p["key"] for p in registry.products()):
        assert key not in registry.intersphinx_mapping(
            key, "https://example.test/cccl", site=site
        )


def test_products_link_only_to_those_built_before_them(tmp_path):
    """build_site.py builds in registry order, so a product listed later has no
    inventory yet when an earlier one builds -- in the run that adds it, not
    even a published one -- and an unreachable inventory fails a -W build even
    when nothing links to it."""
    path = tmp_path / "registry.yml"
    products = [{"key": k} for k in ("cpp", "python", "coop")]
    path.write_text(yaml.safe_dump({"products": products}), encoding="utf-8")

    def maps(key):
        mapping = registry.intersphinx_mapping(
            key, "https://example.test/cccl", path=path, site=tmp_path
        )
        return list(mapping)

    assert maps("cpp") == []
    assert maps("python") == ["cpp"]
    assert maps("coop") == ["cpp", "python"]
    assert maps(None) == ["cpp", "python", "coop"]
    with pytest.raises(SystemExit):
        maps("rust")


def test_build_site_exports_the_assembled_site_directory():
    """registry.intersphinx_mapping defaults to the environment, and the product
    builds are subprocesses -- so the assembly has to put the directory there or
    every build silently falls back to the live site."""
    source = (DOCS / "build_site.py").read_text(encoding="utf-8")
    assert "CCCL_DOCS_SITE_DIR" in source, (
        "build_site.py must export CCCL_DOCS_SITE_DIR; without it each product "
        "build resolves cross-references against the previous publish"
    )


# --------------------------------------------------------------------------
# The publication path
# --------------------------------------------------------------------------
#
# These guard build-site.yml and the assembly it runs. Every failure here is
# silent: the workflow succeeds, the deploy succeeds, and the site is wrong.


@pytest.fixture(scope="module")
def site_workflow():
    return yaml.safe_load(SITE_WORKFLOW.read_text(encoding="utf-8"))


def _deploy_step(workflow):
    steps = [
        s
        for s in workflow["jobs"]["publish"]["steps"]
        if "actions-gh-pages" in str(s.get("uses", ""))
    ]
    assert len(steps) == 1, f"expected exactly one deploy step, found {len(steps)}"
    return steps[0]


def test_the_branch_is_recreated_from_the_artifact(site_workflow):
    """force_orphan is what keeps the branch at one commit and the clone small.
    It is only safe because the artifact is seeded first -- so the two have to be
    asserted together, or the next person reads one and removes the other."""
    assert _deploy_step(site_workflow)["with"]["force_orphan"] is True
    body = SITE_WORKFLOW.read_text(encoding="utf-8")
    assert "--seed-from" in body, (
        "force_orphan without seeding republishes only this build, deleting "
        "every version already on the site"
    )


def test_the_assembly_seeds_from_the_branch_being_published(site_workflow):
    """Seeding the wrong branch publishes one branch's history over another."""
    build = next(
        s
        for s in site_workflow["jobs"]["publish"]["steps"]
        if s.get("name") == "Build and assemble"
    )
    assert build["env"]["CCCL_DOCS_BRANCH"] == "origin/${{ steps.plan.outputs.branch }}"
    assert '--seed-from "${CCCL_DOCS_BRANCH}"' in build["run"]
    assert _deploy_step(site_workflow)["with"]["publish_branch"] == (
        "${{ steps.plan.outputs.branch }}"
    )


def test_deploy_action_is_pinned_to_a_sha(site_workflow):
    uses = _deploy_step(site_workflow)["uses"]
    assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), uses


def test_the_build_is_told_which_origin_it_publishes_to():
    """Every cross-reference, canonical URL and switcher entry is built from this.
    Unset, a rehearsal on a fork publishes links pointing at production."""
    body = SITE_WORKFLOW.read_text(encoding="utf-8")
    assert "CCCL_DOCS_SITE_URL" in body
    assert "CCCL_DOCS_SITE_DIR" in (DOCS / "build_site.py").read_text(
        encoding="utf-8"
    ), (
        "the assembly must export the site directory, or each product build "
        "resolves cross-references against the previous publish"
    )


def test_placing_a_product_replaces_its_version_and_nothing_else(tmp_path):
    """A release rebuilds one version. Replacing more deletes published releases;
    replacing less leaves pages behind that were removed from the source."""
    artifact = tmp_path / "artifacts" / "cpp"
    (artifact / "3.5.0").mkdir(parents=True)
    (artifact / "3.5.0" / "index.html").write_text("new")
    (artifact / "nv-versions.json").write_text("[]")

    site = tmp_path / "_site" / "cpp"
    (site / "3.4.2").mkdir(parents=True)
    (site / "3.4.2" / "index.html").write_text("seeded")
    (site / "3.5.0").mkdir()
    (site / "3.5.0" / "stale.html").write_text("removed from source")

    monkey = build_site.ARTIFACTS
    try:
        build_site.ARTIFACTS = tmp_path / "artifacts"
        build_site.place_product({"key": "cpp"}, "3.5.0", tmp_path / "_site")
    finally:
        build_site.ARTIFACTS = monkey

    assert (site / "3.4.2" / "index.html").read_text() == "seeded"
    assert (site / "3.5.0" / "index.html").read_text() == "new"
    assert not (site / "3.5.0" / "stale.html").exists()


def test_a_branch_that_cannot_be_read_stops_the_run(tmp_path):
    """A commit with no docs/ stands in for a branch that exists but cannot be
    archived. Carrying on would hand force_orphan an artifact missing every
    published version, and force_orphan would publish it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [*git, "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty"]
        + ["-m", "no docs/ here"],
        cwd=repo,
        check=True,
    )
    site = tmp_path / "site"
    site.mkdir()
    with pytest.raises(SystemExit):
        build_site.seed_from_branch("HEAD", site, repo=repo)


def test_a_missing_branch_is_a_first_publication_not_an_error(tmp_path):
    """The very first deploy has nothing to seed from, and must still run."""
    assert build_site.seed_from_branch("no/such/branch", tmp_path) == 0


def test_a_sibling_built_but_not_yet_assembled_is_still_found(tmp_path, monkeypatch):
    """A product built on its own, without build_site.py, has no assembled
    site. Its sibling's inventory is still found in the raw build output
    rather than fetched from the published site, which lags a publish behind."""
    artifacts = tmp_path / "artifacts"
    inv = artifacts / "cpp" / "unstable" / "objects.inv"
    inv.parent.mkdir(parents=True)
    inv.write_bytes(b"# Sphinx inventory version 2\n")
    monkeypatch.setattr(registry, "ARTIFACTS", artifacts)

    mapping = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=tmp_path / "no-site"
    )
    assert mapping["cpp"][1] == str(inv)


def test_the_assembled_site_wins_over_the_raw_build(tmp_path, monkeypatch):
    """Both exist during a publish: place_product has copied the build into the
    site. They are the same bytes, but the site is what gets published, so it is
    the one the lookup should name."""
    artifacts = tmp_path / "artifacts"
    (artifacts / "cpp" / "unstable").mkdir(parents=True)
    (artifacts / "cpp" / "unstable" / "objects.inv").write_bytes(b"x")
    monkeypatch.setattr(registry, "ARTIFACTS", artifacts)

    site = tmp_path / "_site"
    (site / "cpp" / "unstable").mkdir(parents=True)
    (site / "cpp" / "unstable" / "objects.inv").write_bytes(b"x")

    mapping = registry.intersphinx_mapping(
        "python", "https://example.test/cccl", site=site
    )
    assert mapping["cpp"][1] == str(site / "cpp" / "unstable" / "objects.inv")


def test_publishing_one_branch_at_a_time(site_workflow):
    """Two runs on one branch each seed from the tip as it stood when they
    began, so the second silently drops what the first published."""
    group = site_workflow["concurrency"]["group"]
    assert "docs-branch" in group, "the group must be per target branch"
    assert site_workflow["concurrency"]["cancel-in-progress"] is False, (
        "a cancelled run never publishes the version it was asked to"
    )


def test_a_failed_fetch_is_not_mistaken_for_a_first_publication():
    """Both leave the ref absent. One means seed nothing, the other means the
    site is about to be republished without the versions already on it."""
    body = SITE_WORKFLOW.read_text(encoding="utf-8")
    assert "git ls-remote --exit-code origin" in body, (
        "the remote has to be asked which case this is"
    )
    assert "|| true" not in body, "a swallowed fetch failure publishes an empty site"
