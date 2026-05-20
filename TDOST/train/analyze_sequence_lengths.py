"""
Analyze sequence length distribution and timing characteristics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

TDOST_INPUT_PATH = Path("/home/nadira/partnr-planner/TDOST/tdost_file")

def load_tdost_raw(path):
    """Load raw TDOST data to inspect structure."""
    rows = []
    
    input_path = Path(path)
    if input_path.is_dir():
        file_paths = sorted(p for p in input_path.glob("*.txt") if p.is_file())
    else:
        file_paths = [input_path]
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                
                tab_count = line.count('\t')
                if tab_count != 3:
                    continue
                
                parts = line.split('\t')
                rows.append({
                    'raw_event': parts[0].strip(),
                    'structured_event': parts[1].strip(),
                    'descriptions': parts[2].strip(),
                    'label': parts[3].strip(),
                    'source_file': file_path.name,
                    'raw_line': line,
                })
    
    return pd.DataFrame(rows)

# Load raw data
df = load_tdost_raw(TDOST_INPUT_PATH)

print(f"Total rows: {len(df)}")
print(f"\nFirst few rows:")
print(df[['raw_event', 'structured_event', 'label']].head(10))

print(f"\n=== Column Info ===")
print(f"Columns: {df.columns.tolist()}")
print(f"\nUnique files: {df['source_file'].unique()}")

print(f"\n=== Raw Event Examples ===")
print(df['raw_event'].head(20))

print(f"\n=== Structured Event Examples ===")
print(df['structured_event'].head(20))

# Check if there's any timing info in raw_event or structured_event
print(f"\n=== Checking for timing keywords ===")
time_keywords = ['time', 'duration', 'start', 'end', 'hour', 'minute', 'second', ':']
for kw in time_keywords:
    count = df['raw_event'].str.contains(kw, case=False, na=False).sum()
    if count > 0:
        print(f"  '{kw}' found in {count} raw_event rows")

print(f"\n=== Sequence-level Statistics ===")
# Group by activity to get sequence lengths
df['activity'] = df['label'].str.lower().str.strip()

# Create sequence IDs (similar to main.py)
df['seq_id'] = (df['activity'] != df['activity'].shift()).cumsum() - 1
seq_lengths = df.groupby('seq_id').size()

print(f"Total sequences: {len(seq_lengths)}")
print(f"Sequence length statistics:")
print(f"  Mean: {seq_lengths.mean():.2f} events")
print(f"  Median: {seq_lengths.median():.0f} events")
print(f"  Min: {seq_lengths.min()} events")
print(f"  Max: {seq_lengths.max()} events")
print(f"  Std: {seq_lengths.std():.2f}")

print(f"\nSequence length distribution:")
print(seq_lengths.describe())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(seq_lengths, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Sequence Length (# events)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Sequence Lengths')
axes[0].axvline(seq_lengths.mean(), color='r', linestyle='--', label=f'Mean: {seq_lengths.mean():.1f}')
axes[0].axvline(100, color='g', linestyle='--', label='MAX_LEN: 100')
axes[0].legend()

axes[1].boxplot(seq_lengths)
axes[1].set_ylabel('Sequence Length (# events)')
axes[1].set_title('Sequence Length Box Plot')
axes[1].axhline(100, color='g', linestyle='--', label='MAX_LEN: 100')

plt.tight_layout()
plt.savefig('sequence_length_analysis.png', dpi=100, bbox_inches='tight')
print(f"\nPlot saved to sequence_length_analysis.png")

# Check for percentage of sequences that exceed MAX_LEN
exceed_max = (seq_lengths > 100).sum()
print(f"\nSequences exceeding MAX_LEN=100: {exceed_max} ({100*exceed_max/len(seq_lengths):.1f}%)")
