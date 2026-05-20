#!/usr/bin/env python3
"""
Utility to fix formatting issues in tdost_llm_lang_descriptions.txt
without requiring API calls. Parses malformed entries and reconstructs them
with proper tab separation.
"""

import re
import os
import json


def parse_problematic_line(line: str) -> dict | None:
    """
    Try to parse a line that may have tabs in the JSON array column.
    Expected format (with tabs as \t):
        <sensor_key> <state> <time>\t<structured>\t<json_descriptions>\t<activity>
    """
    # Split by tabs to extract columns
    parts = line.split('\t')
    
    if len(parts) >= 4:
        raw_event = parts[0].strip()
        structured_event = parts[1].strip()

        # Everything between the second and last element (excluding last) is descriptions
        descriptions = '\t'.join(parts[2:-1]).strip()
        activity = parts[-1].strip()
    elif len(parts) == 3:
        # Recovery path: activity might be appended after JSON with spaces, not a tab.
        raw_event = parts[0].strip()
        structured_event = parts[1].strip()
        third = parts[2].strip()

        match = re.match(r'^(?P<desc>\[.*\])\s+(?P<label>\S+)$', third)
        if not match:
            return None
        descriptions = match.group('desc').strip()
        activity = match.group('label').strip()
    else:
        return None
    
    # Normalize descriptions: remove internal newlines and extra whitespace
    descriptions = ' '.join(descriptions.split())
    
    return {
        'raw_event': raw_event,
        'structured_event': structured_event,
        'descriptions': descriptions,
        'activity': activity,
    }


def fix_file(input_path: str, output_path: str) -> None:
    """
    Fix the malformed TSV file and write a properly formatted version.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    fixed_count = 0
    skipped_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                
                parsed = parse_problematic_line(line)
                
                if parsed:
                    # Reconstruct with proper tab separation
                    fixed_line = '\t'.join([
                        parsed['raw_event'],
                        parsed['structured_event'],
                        parsed['descriptions'],
                        parsed['activity'],
                    ])
                    f_out.write(fixed_line + '\n')
                    fixed_count += 1
                else:
                    print(f"Warning: Could not parse line {line_num}: {line[:80]}...")
                    skipped_count += 1
    
    print(f"\nFixing complete!")
    print(f"  Fixed lines: {fixed_count}")
    print(f"  Skipped lines: {skipped_count}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "tdost_llm_lang_descriptions.txt")
    output_file = os.path.join(BASE_DIR, "tdost_llm_lang_descriptions_fixed.txt")
    
    print(f"Fixing file: {input_file}")
    fix_file(input_file, output_file)
    
    # Optionally replace the original
    print(f"\nTo use the fixed file, run:")
    print(f"  mv {output_file} {input_file}")
