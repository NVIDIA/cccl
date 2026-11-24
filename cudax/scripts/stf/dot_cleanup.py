import re
import subprocess
import sys


def extract_function_name(template_content):
    """
    Extract the function name from complex template parameters.
    Examples:
    - 'compute_residual(cuda::experimental::stf::stackable_ctx &, ...' -> 'compute_residual'
    - 'main::[lambda(unsigned long, T1, T2) (instance 1)]' -> 'main'
    """
    # First, try to find a function-like pattern: name(args...)
    func_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", template_content)
    if func_match:
        return func_match.group(1)

    # Handle patterns like "main::[lambda...]" - take part before "::"
    if "::" in template_content:
        first_part = template_content.split("::", 1)[0].strip()
        # Extract the identifier
        identifier_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)$", first_part)
        if identifier_match:
            return identifier_match.group(1)
        return first_part

    # Fallback: take first identifier before comma
    first_part = template_content.split(",", 1)[0].strip()

    # Extract just the identifier part (remove any type qualifiers)
    identifier_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)$", first_part)
    if identifier_match:
        return identifier_match.group(1)

    return first_part


def find_matching_bracket(text, start_pos, open_char="<", close_char=">"):
    """Find the position of the matching closing bracket for nested brackets"""
    bracket_count = 1
    i = start_pos
    while i < len(text) and bracket_count > 0:
        if text[i] == open_char:
            bracket_count += 1
        elif text[i] == close_char:
            bracket_count -= 1
        i += 1
    return i - 1 if bracket_count == 0 else -1


def simplify(line: str) -> str:
    # First, remove the long "void cuda::experimental::stf::reserved::" prefix and variations
    line = line.replace("void cuda::experimental::stf::reserved::", "")
    line = line.replace("cuda::experimental::stf::reserved::", "")

    # Handle special cases first
    if "loop_redux_finalize" in line:
        line = re.sub(r".*loop_redux_finalize.*", "loop_redux_finalize", line)
        return line

    # Handle condition_reset - remove angle brackets entirely
    line = re.sub(r"condition_reset<[^>]*>", "condition_reset", line)

    # Handle loop_redux<...> patterns
    result = line
    pos = 0
    while True:
        match = re.search(r"loop_redux<", result[pos:])
        if not match:
            break
        start = pos + match.start()
        bracket_start = pos + match.end() - 1  # position of '<'
        bracket_end = find_matching_bracket(result, bracket_start + 1)
        if bracket_end != -1:
            inside = result[bracket_start + 1 : bracket_end]
            func_name = extract_function_name(inside)
            # Replace the entire loop_redux<...> with simplified version
            result = (
                result[:start] + f"loop_redux<{func_name}>" + result[bracket_end + 1 :]
            )
            pos = start + len(f"loop_redux<{func_name}>")
        else:
            pos = start + 1

    # Handle regular loop<...> patterns
    pos = 0
    while True:
        match = re.search(r"loop<", result[pos:])
        if not match:
            break
        start = pos + match.start()
        bracket_start = pos + match.end() - 1  # position of '<'
        bracket_end = find_matching_bracket(result, bracket_start + 1)
        if bracket_end != -1:
            inside = result[bracket_start + 1 : bracket_end]
            func_name = extract_function_name(inside)
            # Replace the entire loop<...> with simplified version
            result = result[:start] + f"loop<{func_name}>" + result[bracket_end + 1 :]
            pos = start + len(f"loop<{func_name}>")
        else:
            pos = start + 1

    # Handle condition_update_kernel<...> patterns (like loop patterns)
    pos = 0
    while True:
        match = re.search(r"condition_update_kernel<", result[pos:])
        if not match:
            break
        start = pos + match.start()
        bracket_start = pos + match.end() - 1  # position of '<'
        bracket_end = find_matching_bracket(result, bracket_start + 1)
        if bracket_end != -1:
            inside = result[bracket_start + 1 : bracket_end]
            func_name = extract_function_name(inside)
            # Replace the entire condition_update_kernel<...> with simplified version
            result = (
                result[:start]
                + f"condition_update_kernel<{func_name}>"
                + result[bracket_end + 1 :]
            )
            pos = start + len(f"condition_update_kernel<{func_name}>")
        else:
            pos = start + 1

    # Remove function arguments - keep only allowed characters, stop at :: or (
    def clean_after_loop(match):
        loop_part = match.group(1)  # the loop<...> or loop_redux<...> part
        after_loop = match.group(2)  # everything after it

        # Keep only letters, underscore, whitespace, and stop at :: or (
        cleaned = ""
        i = 0
        while i < len(after_loop):
            char = after_loop[i]
            if char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_\t ":
                cleaned += char
                i += 1
            elif i < len(after_loop) - 1 and after_loop[i : i + 2] == "::":
                break  # Stop at ::
            elif char == "(":
                break  # Stop at (
            elif char == "\n":
                cleaned += char
                break  # Stop at newline but include it
            else:
                i += 1  # Skip other characters but keep going

        return loop_part + cleaned

    # Apply argument cleanup to loop patterns and condition_update_kernel
    result = re.sub(
        r"((?:loop(?:_redux)?|condition_update_kernel)<[^>]+>)(.*)",
        clean_after_loop,
        result,
    )

    # Also clean up condition_reset (which has no angle brackets after simplification)
    result = re.sub(r"(condition_reset)(.*)", clean_after_loop, result)

    return result


def apply_cu_filt(input_text):
    """Apply cu++filt to demangle C++ symbols"""
    try:
        # Run cu++filt to demangle C++ symbols
        process = subprocess.Popen(
            ["cu++filt"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input_text)

        if process.returncode == 0:
            return stdout
        else:
            # If cu++filt fails, fall back to c++filt
            print(f"cu++filt failed: {stderr}, trying c++filt", file=sys.stderr)
            process = subprocess.Popen(
                ["c++filt"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input_text)
            if process.returncode == 0:
                return stdout
            else:
                print(
                    f"c++filt also failed: {stderr}, using original text",
                    file=sys.stderr,
                )
                return input_text
    except FileNotFoundError:
        # If cu++filt is not available, try c++filt
        try:
            process = subprocess.Popen(
                ["c++filt"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input_text)
            if process.returncode == 0:
                return stdout
            else:
                print(f"c++filt failed: {stderr}, using original text", file=sys.stderr)
                return input_text
        except FileNotFoundError:
            print(
                "Neither cu++filt nor c++filt found, using original text",
                file=sys.stderr,
            )
            return input_text


if __name__ == "__main__":
    # Read all input
    input_text = sys.stdin.read()

    # First apply cu++filt to demangle symbols
    demangled_text = apply_cu_filt(input_text)

    # Then apply our custom simplification to each line
    for line in demangled_text.splitlines(keepends=True):
        print(simplify(line), end="")
