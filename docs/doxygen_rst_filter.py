#!/usr/bin/env python3
"""
Doxygen filter to convert @rst...@endrst blocks to proper doxygen format.
This allows us to embed reStructuredText directly in doxygen comments.
"""

import sys
import re

def process_rst_blocks(content):
    """Convert @rst...@endrst blocks to doxygen's \\verbatim embed:rst format."""
    
    # Pattern to match @rst...@endrst blocks (including multiline)
    pattern = r'@rst\s*(.*?)@endrst'
    
    def replace_rst_block(match):
        rst_content = match.group(1)
        # Use doxygen's verbatim embed:rst syntax
        return '\\verbatim embed:rst:leading-asterisk\n' + rst_content + '\\endverbatim'
    
    # Process the content with DOTALL flag to match across lines
    result = re.sub(pattern, replace_rst_block, content, flags=re.DOTALL)
    
    # Also handle simpler @rst commands that might not have @endrst
    result = re.sub(r'@rst\s+', '\\verbatim embed:rst:leading-asterisk\n', result)
    
    # Handle orphaned @endrst
    result = re.sub(r'@endrst', '\\endverbatim', result)
    
    return result

def main():
    """Read from stdin, process, and write to stdout."""
    content = sys.stdin.read()
    processed = process_rst_blocks(content)
    sys.stdout.write(processed)

if __name__ == '__main__':
    main()