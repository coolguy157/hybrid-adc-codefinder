# preprocessor.py
import json
import networkx as nx
import numpy as np
from dataclasses import dataclass, asdict
import os
from collections import defaultdict

# --- Data Structure and Converters ---
# This dataclass matches the required JSON output schema from your memory file.
@dataclass
class CodeRecord:
    n: int
    d: int
    is_indecomposable: bool
    type: int
    graph6_string: str
    adjacency_matrix: np.ndarray

    def to_dict(self):
        """Converts the dataclass to a dictionary, serializing the NumPy array."""
        d = asdict(self)
        d['adjacency_matrix'] = self.adjacency_matrix.tolist()
        return d

def graph6_to_adj_matrix(graph6_str: str) -> np.ndarray:
    """Converts a graph6 string to a NumPy adjacency matrix."""
    graph6_bytes = graph6_str.encode('ascii')
    graph_obj = nx.from_graph6_bytes(graph6_bytes)
    adj_matrix = nx.to_numpy_array(graph_obj, nodelist=sorted(graph_obj.nodes()), dtype=int)
    return adj_matrix

def parse_record_line(line: str) -> CodeRecord:
    """Parses a single tab-separated line from the raw file into a CodeRecord."""
    parts = line.strip().split('\t')
    
    # Extract data based on the tab-separated format
    n = int(parts[0])
    d = int(parts[1])
    is_indecomposable = parts[2].strip() == 'I'
    type_val = int(parts[3])
    # The other fields (weight distribution, etc.) are in parts[4] through [7]
    # but are ignored to match the specified JSON output schema.
    graph6_string = parts[8].strip()
    
    adjacency_matrix = graph6_to_adj_matrix(graph6_string)

    return CodeRecord(
        n=n,
        d=d,
        is_indecomposable=is_indecomposable,
        type=type_val,
        graph6_string=graph6_string,
        adjacency_matrix=adjacency_matrix,
    )

# --- Main Execution Logic ---
def main(raw_db_path: str, output_dir: str):
    """
    Reads the raw, line-by-line database file, processes it, and writes out
    structured JSON Lines files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(raw_db_path, 'r') as f:
            lines = f.readlines()
        print(f"Successfully opened and read {len(lines)} lines from {raw_db_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{raw_db_path}'")
        print("Please make sure the input file is in the same directory as the script, or provide the full path.")
        return

    codes_by_n = defaultdict(list)
    # Iterate through each line instead of chunks of 9.
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            continue
        try:
            record = parse_record_line(line)
            codes_by_n[record.n].append(record)
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping malformed line #{i+1}: '{line.strip()}' | Error: {e}")

    if not codes_by_n:
        print("No valid records were parsed. Exiting.")
        return

    for n, codes_list in codes_by_n.items():
        output_path = os.path.join(output_dir, f"candidates_n{n}.jsonl")
        with open(output_path, 'w') as out_f:
            for code in codes_list:
                out_f.write(json.dumps(code.to_dict()) + '\n')
        print(f"Created {output_path} with {len(codes_list)} records.")
    
    print("\nProcessing complete.")

if __name__ == '__main__':
    # Define the name of your input file.
    # Make sure this file is in the same directory as this script.
    input_filename = 'thebigselfdualfile.txt'
    
    # Define the directory where the output files will be saved.
    output_directory = './preprocessed_data'
    
    main(input_filename, output_directory)