"""
Sphinx extension to automatically generate API reference pages from Doxygen XML.
Replicates repo-docs' automatic API generation functionality.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def extract_param_summary(params):
    """Extract a simplified parameter summary for section headers."""
    if not params:
        return ""

    # Remove template details and namespaces for brevity
    params = params.strip()
    if params.startswith("(") and params.endswith(")"):
        params = params[1:-1]

    # If empty after removing parentheses
    if not params.strip():
        return ""

    # Split by comma (handling nested templates/parentheses)
    param_parts = []
    depth = 0
    current = []

    for char in params:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            param_parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if current:
        param_parts.append("".join(current).strip())

    # Extract parameter names
    param_names = []

    for param in param_parts:
        param = param.strip()

        # Special case for execution policy - shorten for readability
        if "execution_policy_base" in param:
            param_names.append("exec")
        else:
            # Split by spaces and find the parameter name (typically the last word)
            words = param.split()
            if words:
                # The parameter name is the last word
                param_name = words[-1]
                # Clean up reference/pointer markers
                param_name = param_name.strip("&*,")

                # Just use the parameter name as-is
                if param_name:
                    param_names.append(param_name)

    return ", ".join(param_names)


def extract_function_signatures(func_name, refids, xml_dir, namespace=""):
    """Extract exact function signatures from Doxygen XML for overloaded functions."""
    signatures = []
    xml_path = Path(xml_dir)

    if not xml_path.exists():
        return signatures

    # Extract the simple function name (without namespace) for comparison
    simple_func_name = func_name.split("::")[-1] if "::" in func_name else func_name

    # Parse both namespace and group XML files to get function signatures
    # Functions can be defined in either location (often in group_*.xml for thrust/cub)
    xml_files = list(xml_path.glob("namespace*.xml")) + list(
        xml_path.glob("group*.xml")
    )

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find all function members with matching refids
            for memberdef in root.findall('.//memberdef[@kind="function"]'):
                member_refid = memberdef.get("id")
                if member_refid in refids:
                    # Get the function name
                    name_elem = memberdef.find("name")
                    if name_elem is not None and name_elem.text == simple_func_name:
                        # Get the exact args string and definition
                        argsstring_elem = memberdef.find("argsstring")
                        definition_elem = memberdef.find("definition")

                        if argsstring_elem is not None and argsstring_elem.text:
                            # Get the full qualified name from definition if available
                            if definition_elem is not None and definition_elem.text:
                                # Definition includes namespace and return type
                                # Extract just the qualified function name
                                definition = definition_elem.text
                                # Look for the function name in the definition
                                if (
                                    "::" in definition
                                    and simple_func_name in definition
                                ):
                                    # Extract namespace::function from definition
                                    parts = definition.split()
                                    for part in parts:
                                        if simple_func_name in part and "::" in part:
                                            qualified_name = part
                                            break
                                    else:
                                        # Use the original func_name if it has namespace, otherwise add namespace
                                        qualified_name = (
                                            func_name
                                            if "::" in func_name
                                            else (
                                                f"{namespace}::{func_name}"
                                                if namespace
                                                else func_name
                                            )
                                        )
                                else:
                                    # Use the original func_name if it has namespace, otherwise add namespace
                                    qualified_name = (
                                        func_name
                                        if "::" in func_name
                                        else (
                                            f"{namespace}::{func_name}"
                                            if namespace
                                            else func_name
                                        )
                                    )
                            else:
                                # Use the original func_name if it has namespace, otherwise add namespace
                                qualified_name = (
                                    func_name
                                    if "::" in func_name
                                    else (
                                        f"{namespace}::{func_name}"
                                        if namespace
                                        else func_name
                                    )
                                )

                            # Build the complete signature for breathe
                            full_signature = (
                                qualified_name + argsstring_elem.text.strip()
                            )
                            # Check if this signature is already in the list (avoid duplicates)
                            if not any(sig[1] == full_signature for sig in signatures):
                                signatures.append((member_refid, full_signature))
        except Exception as e:
            logger.debug(f"Failed to extract signatures from {xml_file}: {e}")

    return signatures


def extract_doxygen_items(xml_dir):
    """Extract all items (classes, structs, functions, etc.) from Doxygen XML."""
    items = {
        "classes": [],
        "structs": [],
        "functions": [],
        "typedefs": [],
        "enums": [],
        "variables": [],
        "function_groups": {},  # Group functions by name for overloads
        "groups": [],  # Doxygen groups
    }

    xml_path = Path(xml_dir)
    if not xml_path.exists():
        return items

    # Parse index.xml to get all compounds and members
    index_file = xml_path / "index.xml"
    if not index_file.exists():
        return items

    try:
        tree = ET.parse(index_file)
        root = tree.getroot()

        # Get the namespace compound (e.g., cub, thrust, cuda::experimental)
        namespace_compounds = []
        for compound in root.findall('.//compound[@kind="namespace"]'):
            name = compound.find("name").text
            # Match primary namespaces and nested namespaces for cudax
            if name in ["cub", "thrust", "cuda"] or name.startswith(
                "cuda::experimental"
            ):
                namespace_compounds.append(compound)

        # Extract classes and structs
        all_classes = []
        all_structs = []

        for compound in root.findall('.//compound[@kind="class"]'):
            name = compound.find("name").text
            refid = compound.get("refid")

            # Skip internal/detail classes
            if "detail" in name.lower() or "__" in name:
                continue

            all_classes.append((name, refid))

        for compound in root.findall('.//compound[@kind="struct"]'):
            name = compound.find("name").text
            refid = compound.get("refid")

            # Skip internal/detail structs
            if "detail" in name.lower() or "__" in name:
                continue

            all_structs.append((name, refid))

        # Filter out nested classes/structs when their parent is also documented
        # This prevents duplicate declarations in Sphinx
        def is_nested_and_parent_exists(name, all_classes, all_structs):
            """Check if this is a nested class/struct and its parent is also documented."""
            if "::" not in name:
                return False

            # Get the parent name by removing the last component
            parent_name = "::".join(name.split("::")[:-1])

            # Check if parent exists in either classes or structs list
            all_items = all_classes + all_structs
            for item_name, _ in all_items:
                if item_name == parent_name:
                    return True
            return False

        # Filter classes (check against both classes and structs for parents)
        for name, refid in all_classes:
            if not is_nested_and_parent_exists(name, all_classes, all_structs):
                items["classes"].append((name, refid))

        # Filter structs (check against both classes and structs for parents)
        for name, refid in all_structs:
            if not is_nested_and_parent_exists(name, all_classes, all_structs):
                items["structs"].append((name, refid))

        # Extract groups and their members
        for compound in root.findall('.//compound[@kind="group"]'):
            name = compound.find("name").text
            refid = compound.get("refid")
            items["groups"].append((name, refid))

            # Also extract typedefs that are members of groups
            # We need to get the qualified name from the actual group XML file
            for member in compound.findall('member[@kind="typedef"]'):
                name_elem = member.find("name")
                if name_elem is None:
                    continue
                simple_name = name_elem.text
                typedef_refid = member.get("refid")

                # Try to get the qualified name from the group XML file
                qualified_name = simple_name  # Default to simple name
                group_xml_file = xml_path / f"{refid}.xml"
                if group_xml_file.exists():
                    try:
                        group_tree = ET.parse(group_xml_file)
                        group_root = group_tree.getroot()
                        # Find the typedef with matching refid
                        typedef_elem = group_root.find(
                            f'.//memberdef[@id="{typedef_refid}"]'
                        )
                        if typedef_elem is not None:
                            qualifiedname_elem = typedef_elem.find("qualifiedname")
                            if (
                                qualifiedname_elem is not None
                                and qualifiedname_elem.text
                            ):
                                qualified_name = qualifiedname_elem.text
                    except Exception:
                        pass

                items["typedefs"].append((qualified_name, typedef_refid))

        # Extract functions, typedefs, enums, and variables from namespaces
        for namespace_compound in namespace_compounds:
            namespace_name = namespace_compound.find("name").text

            for member in namespace_compound.findall('member[@kind="function"]'):
                name = member.find("name").text
                refid = member.get("refid")
                # Only include full namespace for nested namespaces (cudax)
                # For simple namespaces like 'thrust', 'cub', just use the function name
                if namespace_name and "::" in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items["functions"].append((full_name, refid))

                # Also track function groups for overloads
                if full_name not in items["function_groups"]:
                    items["function_groups"][full_name] = []
                items["function_groups"][full_name].append(refid)

            for member in namespace_compound.findall('member[@kind="typedef"]'):
                name = member.find("name").text
                refid = member.get("refid")
                # Only include full namespace for nested namespaces
                if namespace_name and "::" in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items["typedefs"].append((full_name, refid))

            for member in namespace_compound.findall('member[@kind="enum"]'):
                name = member.find("name").text
                refid = member.get("refid")
                # Only include full namespace for nested namespaces
                if namespace_name and "::" in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items["enums"].append((full_name, refid))

            for member in namespace_compound.findall('member[@kind="variable"]'):
                name = member.find("name").text
                refid = member.get("refid")
                # Only include full namespace for nested namespaces
                if namespace_name and "::" in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items["variables"].append((full_name, refid))

    except Exception as e:
        logger.warning(f"Failed to parse Doxygen XML: {e}")

    return items


def extract_doxygen_classes(xml_dir, project_name=None):
    """Extract classes categorized by type for category pages."""
    # Only CUB uses these specific categories
    if project_name == "cub":
        classes = {
            "device": [],
            "block": [],
            "warp": [],
            "grid": [],
            "iterator": [],
            "thread": [],
            "utility": [],
        }
    else:
        # Other projects don't need category pages
        classes = {}

    # Only categorize for CUB
    if project_name == "cub":
        items = extract_doxygen_items(xml_dir)

        # Filter out internal implementation details
        internal_patterns = [
            "StoreInternal",
            "LoadInternal",
            "_TempStorage",
            "TileDescriptor",
        ]

        # Categorize classes and structs
        for name, refid in items["classes"] + items["structs"]:
            # Skip internal implementation details
            if any(pattern in name for pattern in internal_patterns):
                continue

            # Remove namespace prefixes for categorization
            simple_name = name.split("::")[-1] if "::" in name else name

            # Categorize based on name
            if "Device" in simple_name:
                classes["device"].append((name, refid))
            elif "Block" in simple_name:
                classes["block"].append((name, refid))
            elif "Warp" in simple_name:
                classes["warp"].append((name, refid))
            elif "Grid" in simple_name:
                classes["grid"].append((name, refid))
            elif "Iterator" in simple_name.lower() or "iterator" in simple_name.lower():
                classes["iterator"].append((name, refid))
            elif any(
                x in simple_name
                for x in ["Traits", "Type", "Allocator", "Debug", "Caching"]
            ):
                classes["utility"].append((name, refid))

    return classes


def generate_api_page(category, classes, project_name):
    """Generate RST content for an API category page."""

    category_titles = {
        "device": "Device-wide Primitives",
        "block": "Block-wide Primitives",
        "warp": "Warp-wide Primitives",
        "grid": "Grid-level Primitives",
        "iterator": "Iterator Utilities",
        "thread": "Thread-level Primitives",
        "utility": "Utility Components",
    }

    content = []
    content.append(category_titles.get(category, f"{category.title()} API"))
    content.append("=" * len(content[0]))
    content.append("")
    content.append(".. contents:: Table of Contents")
    content.append("   :local:")
    content.append("   :depth: 2")
    content.append("")

    # Sort classes by name
    classes.sort(key=lambda x: x[0])

    for class_name, refid in classes:
        # Use the full name including namespace for display
        display_name = class_name

        content.append(display_name)
        content.append("-" * len(display_name))
        content.append("")

        # Check if this is a struct by looking at the refid
        # Doxygen uses 'struct' prefix in the refid for structs
        directive = "doxygenstruct" if refid.startswith("struct") else "doxygenclass"

        # Use the full qualified name including namespace
        content.append(f".. {directive}:: {class_name}")

        content.append(f"   :project: {project_name}")
        content.append("   :members:")
        content.append("   :undoc-members:")
        content.append("")

    return "\n".join(content)


def generate_group_index_page(group_name, group_refid, project_name, xml_dir):
    """Generate RST content for a Doxygen group index page."""
    content = []

    # Add marker comment for auto-generated files
    content.append(".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT")
    content.append("")

    # Parse the group XML to get more details
    group_xml_file = Path(xml_dir) / f"{group_refid}.xml"
    title = group_name
    members = []
    brief_description = ""

    if group_xml_file.exists():
        try:
            tree = ET.parse(group_xml_file)
            root = tree.getroot()
            compounddef = root.find('.//compounddef[@kind="group"]')

            if compounddef is not None:
                # Get the title
                title_elem = compounddef.find("title")
                if title_elem is not None and title_elem.text:
                    title = title_elem.text

                # Get brief description if available
                brief_elem = compounddef.find("briefdescription")
                if brief_elem is not None:
                    # Extract text from brief description
                    brief_text = "".join(brief_elem.itertext()).strip()
                    if brief_text:
                        brief_description = brief_text

                # Get all inner classes/structs
                for innerclass in compounddef.findall("innerclass"):
                    class_refid = innerclass.get("refid")
                    class_name = innerclass.text
                    members.append(("class", class_name, class_refid))

                # Get all member functions, typedefs, variables, etc.
                for sectiondef in compounddef.findall("sectiondef"):
                    for memberdef in sectiondef.findall("memberdef"):
                        member_kind = memberdef.get("kind")
                        member_name = memberdef.find("name")
                        if member_name is not None:
                            member_refid = memberdef.get("id")
                            members.append(
                                (member_kind, member_name.text, member_refid)
                            )
        except Exception as e:
            logger.warning(f"Failed to parse group XML {group_xml_file}: {e}")

    # Add title
    content.append(title)
    content.append("=" * len(title))
    content.append("")

    # Add brief description if available
    if brief_description:
        content.append(brief_description)
        content.append("")

    # Do NOT use doxygengroup directive to avoid duplicate declarations
    # Instead, just provide a simple page with links to members

    # Add toctree for inner classes/structs
    inner_classes = [m for m in members if m[0] == "class"]
    if inner_classes:
        content.append(".. toctree::")
        content.append("   :maxdepth: 1")
        content.append("")

        # Add inner classes to toctree (they have their own pages)
        for member_kind, member_name, member_refid in inner_classes:
            content.append(f"   {member_refid}")
        content.append("")

    return "\n".join(content)


def generate_individual_api_page(class_name, refid, project_name):
    """Generate RST content for a single class/struct API page."""
    content = []

    # Add marker comment for auto-generated files
    content.append(".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT")
    content.append("")

    # Add title
    content.append(class_name)
    content.append("=" * len(class_name))
    content.append("")

    # Check if this is a struct by looking at the refid
    directive = "doxygenstruct" if refid.startswith("struct") else "doxygenclass"

    # Add the doxygen directive
    content.append(f".. {directive}:: {class_name}")
    content.append(f"   :project: {project_name}")
    content.append("   :members:")
    content.append("   :undoc-members:")
    content.append("")

    return "\n".join(content)


def check_function_in_namespace(member_name, xml_dir, namespace):
    """Check if a function is defined in the namespace XML (not just referenced)."""
    namespace_xml = os.path.join(xml_dir, f"namespace{namespace}.xml")
    if not os.path.exists(namespace_xml):
        return False

    try:
        tree = ET.parse(namespace_xml)
        root = tree.getroot()

        # Look for actual function definitions, not just references
        for memberdef in root.findall('.//memberdef[@kind="function"]'):
            name_elem = memberdef.find("name")
            if name_elem is not None and name_elem.text == member_name:
                # Check if it has a definition (not just a reference)
                definition = memberdef.find("definition")
                if definition is not None and definition.text:
                    return True
        return False
    except Exception:
        return False


def generate_member_api_page(
    member_name,
    member_type,
    project_name,
    refid=None,
    overload_refids=None,
    xml_dir=None,
):
    """Generate RST content for a single function/typedef/enum/variable API page."""
    content = []

    # Add marker comment for auto-generated files
    content.append(".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT")
    content.append("")

    # Map member types to Doxygen directives
    directive_map = {
        "function": "doxygenfunction",
        "typedef": "doxygentypedef",
        "enum": "doxygenenum",
        "variable": "doxygenvariable",
    }

    directive = directive_map.get(member_type, "doxygenfunction")

    # For thrust and cub, we need to use the namespace-qualified name
    # For cudax, the member_name already includes the namespace
    if project_name in ["thrust", "cub"]:
        # If the member_name doesn't already include the namespace, add it
        if "::" not in member_name:
            qualified_name = f"{project_name}::{member_name}"
        else:
            qualified_name = member_name
    else:
        qualified_name = member_name

    # Add title
    content.append(f"{qualified_name}")
    content.append("=" * (len(qualified_name) + 4))
    content.append("")

    if member_type == "function" and overload_refids:
        # Check if functions are in a group
        is_group_function = False
        group_name = None

        if overload_refids and "_1" in overload_refids[0]:
            parts = overload_refids[0].split("_1")
            if parts[0].startswith("group__"):
                is_group_function = True
                # Get the group refid (e.g., 'group__stream__compaction')
                group_refid = parts[0]

                # Look up the actual group name from index.xml
                group_name = None
                if xml_dir:
                    index_xml = os.path.join(xml_dir, "index.xml")
                    if os.path.exists(index_xml):
                        try:
                            tree = ET.parse(index_xml)
                            root = tree.getroot()
                            # Find the compound with this refid
                            compound = root.find(f'.//compound[@refid="{group_refid}"]')
                            if compound is not None:
                                name_elem = compound.find("name")
                                if name_elem is not None:
                                    group_name = name_elem.text
                        except Exception:
                            pass

                # Fallback: remove 'group__' prefix if lookup fails
                if not group_name:
                    group_name = group_refid[7:]

        if is_group_function and group_name:
            # For group functions with overloads, we need to handle them specially
            if len(overload_refids) > 1 and xml_dir:
                # Extract signatures for all overloads
                signatures = extract_function_signatures(
                    member_name, overload_refids, xml_dir, namespace=project_name
                )

                if signatures:
                    content.append("Overloads")
                    content.append("---------")
                    content.append("")

                    for idx, (refid, full_sig) in enumerate(signatures, 1):
                        # Extract just the parameter list from the full signature
                        simple_name = (
                            member_name.split("::")[-1]
                            if "::" in member_name
                            else member_name
                        )
                        if simple_name in full_sig:
                            # Use find instead of rfind to get the first occurrence
                            sig_idx = full_sig.find(simple_name)
                            if sig_idx != -1:
                                params = full_sig[sig_idx + len(simple_name) :].strip()
                                # Handle trailing noexcept specifier
                                # Look for the closing parenthesis of the parameter list
                                paren_count = 0
                                param_end = -1
                                for i, char in enumerate(params):
                                    if char == "(":
                                        paren_count += 1
                                    elif char == ")":
                                        paren_count -= 1
                                        if paren_count == 0:
                                            param_end = i + 1
                                            break
                                if param_end > 0:
                                    # Keep only the parameter list (up to and including the closing parenthesis)
                                    params = params[:param_end]

                                # Create a simplified signature for the section header
                                param_summary = extract_param_summary(params)

                                # Add a section header for this overload
                                content.append(f"``{simple_name}({param_summary})``")
                                content.append(
                                    "^" * (len(simple_name) + len(param_summary) + 6)
                                )
                                content.append("")

                                # Use doxygenfunction with the specific parameter signature
                                content.append(
                                    f".. doxygenfunction:: {qualified_name}{params}"
                                )
                                content.append(f"   :project: {project_name}")
                                content.append("")
                else:
                    # Fallback to using doxygengroup if we can't extract signatures
                    content.append(f".. doxygengroup:: {group_name}")
                    content.append(f"   :project: {project_name}")
                    content.append("   :content-only:")
                    content.append("")
            else:
                # Single overload from group - but there might be other overloads in the namespace
                # Extract the signature to be specific
                if xml_dir:
                    signatures = extract_function_signatures(
                        member_name, overload_refids, xml_dir, namespace=project_name
                    )
                    if signatures and len(signatures) == 1:
                        refid, full_sig = signatures[0]
                        simple_name = (
                            member_name.split("::")[-1]
                            if "::" in member_name
                            else member_name
                        )
                        if simple_name in full_sig:
                            # Use find instead of rfind to get the first occurrence
                            sig_idx = full_sig.find(simple_name)
                            if sig_idx != -1:
                                params = full_sig[sig_idx + len(simple_name) :].strip()
                                # Handle trailing noexcept specifier
                                # Look for the closing parenthesis of the parameter list
                                paren_count = 0
                                param_end = -1
                                for i, char in enumerate(params):
                                    if char == "(":
                                        paren_count += 1
                                    elif char == ")":
                                        paren_count -= 1
                                        if paren_count == 0:
                                            param_end = i + 1
                                            break
                                if param_end > 0:
                                    # Keep only the parameter list (up to and including the closing parenthesis)
                                    params = params[:param_end]
                                content.append(
                                    f".. doxygenfunction:: {qualified_name}{params}"
                                )
                                content.append(f"   :project: {project_name}")
                                content.append("")
                            else:
                                # Fallback to simple
                                content.append(f".. doxygenfunction:: {qualified_name}")
                                content.append(f"   :project: {project_name}")
                                content.append("")
                        else:
                            # Fallback to simple
                            content.append(f".. doxygenfunction:: {qualified_name}")
                            content.append(f"   :project: {project_name}")
                            content.append("")
                    else:
                        # Fallback to simple
                        content.append(f".. doxygenfunction:: {qualified_name}")
                        content.append(f"   :project: {project_name}")
                        content.append("")
                else:
                    # No xml_dir, use simple
                    content.append(f".. doxygenfunction:: {qualified_name}")
                    content.append(f"   :project: {project_name}")
                    content.append("")
        elif len(overload_refids) > 1 and xml_dir:
            # For functions with multiple overloads in namespace, extract signatures
            # For cudax, member_name already includes namespace, for others use project_name
            if "::" in member_name:
                # Extract namespace from the qualified name for cudax
                namespace_parts = member_name.split("::")[:-1]
                namespace_name = (
                    "::".join(namespace_parts) if namespace_parts else project_name
                )
            else:
                namespace_name = project_name
            signatures = extract_function_signatures(
                member_name, overload_refids, xml_dir, namespace=namespace_name
            )

            if signatures:
                content.append("Overloads")
                content.append("---------")
                content.append("")

                for idx, (refid, full_sig) in enumerate(signatures, 1):
                    # Extract just the parameter list from the full signature
                    # Look for the simple function name (without namespace) in the signature
                    simple_name = (
                        member_name.split("::")[-1]
                        if "::" in member_name
                        else member_name
                    )
                    if simple_name in full_sig:
                        sig_idx = full_sig.rfind(simple_name)
                        if sig_idx != -1:
                            params = full_sig[sig_idx + len(simple_name) :].strip()

                            # Create a simplified signature for the section header
                            # Extract key parameter types for identification
                            param_summary = extract_param_summary(params)

                            # Add a section header for this overload
                            # Use simple name for readability in headers
                            content.append(f"``{simple_name}({param_summary})``")
                            content.append(
                                "^" * (len(simple_name) + len(param_summary) + 6)
                            )
                            content.append("")

                            # Use doxygenfunction with the specific parameter signature and qualified name
                            content.append(
                                f".. doxygenfunction:: {qualified_name}{params}"
                            )
                            content.append(f"   :project: {project_name}")
                            content.append("   :no-link:")
                            content.append("")
            else:
                # Fallback to simple directive with qualified name
                content.append(f".. {directive}:: {qualified_name}")
                content.append(f"   :project: {project_name}")
                content.append("")
        else:
            # Single function with qualified name
            content.append(f".. {directive}:: {qualified_name}")
            content.append(f"   :project: {project_name}")
            content.append("")
    elif member_type == "function":
        # For single functions or when we don't have xml_dir, use qualified name
        content.append(f".. {directive}:: {qualified_name}")
        content.append(f"   :project: {project_name}")
        content.append("")
    else:
        # For other types, use the qualified name
        content.append(f".. {directive}:: {qualified_name}")
        content.append(f"   :project: {project_name}")
        content.append("")

    return "\n".join(content)


def generate_category_index(category, class_list, project_name):
    """Generate an index page for a category with links to individual class pages."""
    category_titles = {
        "device": "Device-wide Primitives",
        "block": "Block-wide Primitives",
        "warp": "Warp-wide Primitives",
        "grid": "Grid-level Primitives",
        "iterator": "Iterator Utilities",
        "thread": "Thread-level Primitives",
        "utility": "Utility Components",
    }

    content = []

    # Add marker comment for auto-generated files
    content.append(".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT")
    content.append("")

    title = category_titles.get(category, f"{category.title()} API")
    content.append(title)
    content.append("=" * len(title))
    content.append("")

    # Add toctree for all classes in this category
    content.append(".. toctree::")
    content.append("   :maxdepth: 1")
    content.append("   :hidden:")
    content.append("")

    # Sort classes by name
    class_list.sort(key=lambda x: x[0])

    for class_name, refid in class_list:
        # Generate filename from refid (e.g., structcub_1_1DeviceAdjacentDifference)
        filename = refid
        content.append(f"   {filename}")

    content.append("")
    content.append(".. list-table::")
    content.append("   :widths: 50 50")
    content.append("   :header-rows: 1")
    content.append("")
    content.append("   * - Class/Struct")
    content.append("     - Description")

    for class_name, refid in class_list:
        filename = refid
        # Use format_doc_reference but without the list item marker
        doc_ref = format_doc_reference(class_name, filename, "", as_list_item=False)
        content.append(f"   * - {doc_ref}")
        content.append("     - ")  # Description would go here if available

    return "\n".join(content)


def clean_template_name(name):
    """Remove spaces around template parameters to avoid Sphinx parsing issues."""
    # Remove spaces after '<' and before '>'
    cleaned = name.replace("< ", "<").replace(" >", ">")
    # Handle pointer types - remove space before *
    cleaned = cleaned.replace(" *", "*")
    # Handle spaces before commas in template parameters
    while " ," in cleaned:
        cleaned = cleaned.replace(" ,", ",")
    # Handle spaces after commas in template parameters
    while ", " in cleaned:
        cleaned = cleaned.replace(", ", ",")
    # Handle spaces after :: in nested namespaces
    cleaned = cleaned.replace(":: ", "::")
    return cleaned


def format_doc_reference(name, refid, doc_prefix="", as_list_item=True):
    """Format a documentation reference, handling template specializations."""
    clean_name = clean_template_name(name)

    # Build the reference path
    ref_path = f"{doc_prefix}{refid}" if doc_prefix else refid

    # Check if the name contains any angle brackets (template)
    if "<" in clean_name:
        # For any template specialization, just use the file reference without display name
        # This avoids RST parsing issues with angle brackets in the display name
        doc_ref = f":doc:`{ref_path}`"
    else:
        # For non-templates, use the full name as display
        doc_ref = f":doc:`{clean_name} <{ref_path}>`"

    # Return with or without list item marker
    return f"* {doc_ref}" if as_list_item else doc_ref


def generate_namespace_api_page(project_name, items, title=None, doc_prefix=""):
    """Generate a comprehensive namespace API reference page."""
    content = []

    # Add marker comment for auto-generated files
    content.append(".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT")
    content.append("")

    # Determine namespace name
    namespace_name = project_name  # e.g., 'cub', 'thrust', etc.

    # Title - use provided title or default
    if not title:
        title = f"{project_name.upper()} API Reference"

    content.append(title)
    content.append("=" * len(title))
    content.append("")

    # Add namespace description
    namespace_title = f"Namespace ``{namespace_name}``"
    content.append(namespace_title)
    # Make underline slightly longer to avoid "too short" warnings
    content.append("-" * (len(namespace_title) + 2))
    content.append("")

    # Filter out internal implementation details
    internal_patterns = [
        "StoreInternal",
        "LoadInternal",
        "_TempStorage",
        "TileDescriptor",
    ]

    # Classes section
    filtered_classes = [
        (name, refid)
        for name, refid in items["classes"]
        if not any(pattern in name for pattern in internal_patterns)
    ]
    if filtered_classes:
        content.append("Classes")
        content.append("~~~~~~~")
        content.append("")

        # Sort classes alphabetically
        filtered_classes.sort(key=lambda x: x[0].lower())
        for name, refid in filtered_classes:
            content.append(format_doc_reference(name, refid, doc_prefix))
        content.append("")

    # Structs section
    filtered_structs = [
        (name, refid)
        for name, refid in items["structs"]
        if not any(pattern in name for pattern in internal_patterns)
    ]
    if filtered_structs:
        content.append("Structs")
        content.append("~~~~~~~")
        content.append("")

        # Sort structs alphabetically
        filtered_structs.sort(key=lambda x: x[0].lower())
        for name, refid in filtered_structs:
            content.append(format_doc_reference(name, refid, doc_prefix))
        content.append("")

    # Functions section
    if items.get("function_groups"):
        content.append("Functions")
        content.append("~~~~~~~~~")
        content.append("")

        # Sort functions alphabetically by name
        sorted_functions = sorted(
            items["function_groups"].keys(), key=lambda x: x.lower()
        )
        for func_name in sorted_functions:
            # Use the first refid for the link
            first_refid = items["function_groups"][func_name][0]
            content.append(format_doc_reference(func_name, first_refid, doc_prefix))
        content.append("")

    # Typedefs section
    if items["typedefs"]:
        content.append("Type Definitions")
        content.append("~~~~~~~~~~~~~~~~")
        content.append("")

        # Sort typedefs alphabetically
        items["typedefs"].sort(key=lambda x: x[0].lower())
        for name, refid in items["typedefs"]:
            content.append(format_doc_reference(name, refid, doc_prefix))
        content.append("")

    # Enums section
    if items["enums"]:
        content.append("Enumerations")
        content.append("~~~~~~~~~~~~")
        content.append("")

        # Sort enums alphabetically
        items["enums"].sort(key=lambda x: x[0].lower())
        for name, refid in items["enums"]:
            content.append(format_doc_reference(name, refid, doc_prefix))
        content.append("")

    # Variables section
    if items["variables"]:
        content.append("Variables")
        content.append("~~~~~~~~~")
        content.append("")

        # Sort variables alphabetically
        items["variables"].sort(key=lambda x: x[0].lower())
        for name, refid in items["variables"]:
            content.append(format_doc_reference(name, refid, doc_prefix))
        content.append("")

    # Add hidden toctree for all generated pages to avoid orphan warnings
    # This is only for the api/index.rst page (when doc_prefix is empty)
    if not doc_prefix:
        content.append(".. toctree::")
        content.append("   :hidden:")
        content.append("   :maxdepth: 1")
        content.append("")

        # Add category pages only for CUB
        if project_name == "cub":
            for category in [
                "device",
                "block",
                "warp",
                "grid",
                "iterator",
                "thread",
                "utility",
            ]:
                content.append(f"   {category}")

        # Add all class/struct pages
        for name, refid in items["classes"] + items["structs"]:
            content.append(f"   {refid}")

        # Add all function pages
        if items.get("function_groups"):
            for func_name in items["function_groups"]:
                first_refid = items["function_groups"][func_name][0]
                content.append(f"   {first_refid}")

        # Add all typedef pages (deduplicate by refid)
        typedef_refids_seen = set()
        for name, refid in items["typedefs"]:
            if refid not in typedef_refids_seen:
                content.append(f"   {refid}")
                typedef_refids_seen.add(refid)

        # Add all enum pages (deduplicate by refid)
        enum_refids_seen = set()
        for name, refid in items["enums"]:
            if refid not in enum_refids_seen:
                content.append(f"   {refid}")
                enum_refids_seen.add(refid)

        # Add all variable pages (deduplicate by refid)
        variable_refids_seen = set()
        for name, refid in items["variables"]:
            if refid not in variable_refids_seen:
                content.append(f"   {refid}")
                variable_refids_seen.add(refid)

        # Add all group pages
        for name, refid in items.get("groups", []):
            content.append(f"   {refid}")

        content.append("")

    return "\n".join(content)


def generate_api_docs(app, config):
    """Generate API documentation pages during Sphinx build."""

    # Only generate for projects with breathe configuration
    if not hasattr(config, "breathe_projects"):
        return

    for project_name, xml_dir in config.breathe_projects.items():
        # Skip if XML directory doesn't exist
        if not os.path.exists(xml_dir):
            continue

        # Extract all items from Doxygen XML
        items = extract_doxygen_items(xml_dir)

        # Also extract categorized classes for category pages
        classes = extract_doxygen_classes(xml_dir, project_name)

        # Determine output directory based on project
        api_dir = None
        if project_name == "cub":
            api_dir = Path(app.srcdir) / "cub" / "api"
        elif project_name == "thrust":
            api_dir = Path(app.srcdir) / "thrust" / "api"
        elif project_name == "libcudacxx":
            api_dir = Path(app.srcdir) / "libcudacxx" / "api"
        elif project_name == "cudax":
            api_dir = Path(app.srcdir) / "cudax" / "api"

        if not api_dir:
            continue

        # Clean up auto-generated files first
        # Remove all auto-generated .rst files (those matching certain patterns)
        if api_dir.exists():
            logger.info(f"Cleaning up auto-generated files in {api_dir}")
            # Remove files matching Doxygen reference patterns
            for pattern in [
                "class*.rst",
                "struct*.rst",
                "group*.rst",
                "namespace*.rst",
            ]:
                for file in api_dir.glob(pattern):
                    file.unlink()
                    logger.debug(f"Removed {file}")

            # Also remove category files that are auto-generated
            for category in [
                "device",
                "block",
                "warp",
                "grid",
                "iterator",
                "thread",
                "utility",
            ]:
                category_file = api_dir / f"{category}.rst"
                if category_file.exists():
                    # Check if it's auto-generated by looking for our marker
                    try:
                        with open(category_file, "r") as f:
                            first_line = f.readline()
                            if "AUTO-GENERATED" in first_line:
                                category_file.unlink()
                                logger.debug(f"Removed auto-generated {category_file}")
                    except Exception:
                        pass

            # Remove index.rst if it's auto-generated
            index_file = api_dir / "index.rst"
            if index_file.exists():
                try:
                    with open(index_file, "r") as f:
                        first_line = f.readline()
                        if "AUTO-GENERATED" in first_line:
                            index_file.unlink()
                            logger.debug(f"Removed auto-generated {index_file}")
                except Exception:
                    pass

        # Note: We no longer generate or clean up auto_api.rst files

        # Create API directory if it doesn't exist
        api_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual pages for each class/struct
        # Skip internal implementation details that cause template warnings
        internal_patterns = [
            "StoreInternal",
            "LoadInternal",
            "_TempStorage",
            "TileDescriptor",
        ]

        for name, refid in items["classes"] + items["structs"]:
            # Skip internal implementation details
            if any(pattern in name for pattern in internal_patterns):
                logger.debug(f"Skipping internal implementation detail: {name}")
                continue

            # Note: We generate individual pages for all classes/structs,
            # even if they're in groups, so that references work correctly

            # Generate individual page
            content = generate_individual_api_page(name, refid, project_name)
            output_file = api_dir / f"{refid}.rst"

            # Write the individual class page
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated API page: {output_file}")

        # Generate individual pages for functions (one per unique function name)
        # Use function_groups to handle overloads
        function_groups = items.get("function_groups", {})
        for func_name in function_groups:
            # Use the first refid as the filename for consistency
            first_refid = function_groups[func_name][0]
            content = generate_member_api_page(
                func_name,
                "function",
                project_name,
                refid=first_refid,
                overload_refids=function_groups[func_name],
                xml_dir=xml_dir,
            )
            output_file = api_dir / f"{first_refid}.rst"
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated function API page: {output_file}")

        # Generate individual pages for typedefs
        for name, refid in items["typedefs"]:
            content = generate_member_api_page(name, "typedef", project_name, refid)
            output_file = api_dir / f"{refid}.rst"
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated typedef API page: {output_file}")

        # Generate individual pages for enums
        for name, refid in items["enums"]:
            content = generate_member_api_page(name, "enum", project_name, refid)
            output_file = api_dir / f"{refid}.rst"
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated enum API page: {output_file}")

        # Generate individual pages for variables
        for name, refid in items["variables"]:
            content = generate_member_api_page(name, "variable", project_name, refid)
            output_file = api_dir / f"{refid}.rst"
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated variable API page: {output_file}")

        # Generate group index pages
        for group_name, group_refid in items["groups"]:
            content = generate_group_index_page(
                group_name, group_refid, project_name, xml_dir
            )
            output_file = api_dir / f"{group_refid}.rst"
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated group index page: {output_file}")

        # Generate the main namespace API reference page for api/index.rst
        namespace_content = generate_namespace_api_page(project_name, items)
        namespace_file = api_dir / "index.rst"
        with open(namespace_file, "w") as f:
            f.write(namespace_content)
        logger.info(f"Generated namespace API reference: {namespace_file}")

        # Note: We no longer generate auto_api.rst as api/index.rst serves the same purpose

        # Generate category index pages (for backward compatibility)
        # Create stub files for all categories to avoid toctree warnings
        for category, class_list in classes.items():
            output_file = api_dir / f"{category}.rst"

            if class_list:
                content = generate_category_index(category, class_list, project_name)
            else:
                # Create a minimal stub file for empty categories
                category_titles = {
                    "device": "Device-wide Primitives",
                    "block": "Block-wide Primitives",
                    "warp": "Warp-wide Primitives",
                    "grid": "Grid-wide Primitives",
                    "iterator": "Iterator Utilities",
                    "thread": "Thread-level Primitives",
                    "utility": "Utility Components",
                }
                title = category_titles.get(category, category.title())
                content = f""".. AUTO-GENERATED by auto_api_generator.py - DO NOT EDIT

{title}
{"=" * len(title)}

.. note::
   No items in this category.
"""

            # Write the category index
            with open(output_file, "w") as f:
                f.write(content)
            logger.info(f"Generated category index: {output_file}")


def setup(app: Sphinx):
    """Setup the extension."""
    app.connect("config-inited", generate_api_docs)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
