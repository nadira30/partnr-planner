#!/usr/bin/env python3
"""
Fix malformed rows in the TDOST descriptions file.
Malformed rows have only 1 tab instead of 3, with format: error_message\tlabel
Reconstructs them as: \t\terror_message\tlabel (empty raw_event, empty structured_event)
"""

import os


def fix_malformed_file(input_path: str, output_path: str) -> None:
    """Fix rows with incorrect tab count by adding empty columns."""
    
    fixed_count = 0
    good_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                
                tab_count = line.count('\t')
                
                if tab_count == 3:
                    # Already correct format
                    f_out.write(line + '\n')
                    good_count += 1
                elif tab_count == 1:
                    # Malformed: error_message\tlabel
                    # Reconstruct as: \t\terror_message\tlabel
                    parts = line.split('\t')
                    error_msg = parts[0].strip()
                    label = parts[1].strip() if len(parts) > 1 else ''
                    
                    # Create properly formatted row with empty raw_event and structured_event
                    fixed_line = '\t'.join(['', '', error_msg, label])
                    f_out.write(fixed_line + '\n')
                    fixed_count += 1
                else:
                    # Other malformed cases - log and skip
                    print(f"Skipping line {line_num}: unexpected tab count {tab_count}")
    
    print(f"\nFile fixed!")
    print(f"  Good rows (3 tabs): {good_count}")
    print(f"  Fixed rows (1 tab): {fixed_count}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "tdost_file/tdost_llm_lang_descriptions.txt")
    output_file = os.path.join(BASE_DIR, "tdost_file/tdost_llm_lang_descriptions_fixed.txt")
    
    print(f"Fixing malformed rows in: {input_file}")
    fix_malformed_file(input_file, output_file)
    
    print(f"\nTo replace the original file:")
    print(f"  mv {input_file} {input_file}.bak")
    print(f"  mv {output_file} {input_file}")
