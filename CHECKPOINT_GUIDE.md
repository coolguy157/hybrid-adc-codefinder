# Checkpoint & Resume System

## Overview

The checkpoint system saves progress during long-running searches for hybrid quantum codes. If a search is interrupted (by Ctrl+C, system crash, or timeout), it resumes from where it left off without recomputing already-processed combinations.

## How It Works

### 1. **Automatic Checkpointing During Runs**

Every `checkpoint_interval` results (default: 50), the system saves:
- **processed_combinations**: Set of `(graph_key, orientation_str)` tuples already computed
- **results**: List of SearchResult objects found so far

```
Checkpoint saved: 50 processed, 50 results
Checkpoint saved: 100 processed, 100 results
Checkpoint saved: 150 processed, 150 results
...
```

### 2. **Resuming from Checkpoints**

On rerun, the system automatically:
1. Loads the checkpoint file (if it exists)
2. Skips all already-processed combinations
3. Continues from the first unprocessed combination
4. Merges new results with previous results

### 3. **Checkpoint File Structure**

Location: `checkpoints/search_n{n}_{error_set_type}.json`

Example: `checkpoints/search_n4_E1.json`

Contents:
```json
{
  "processed_combinations": [
    ["4q_star_vncorbits_0", "idididid"],
    ["4q_star_vncorbits_0", "ididid(13)"],
    ...
  ],
  "results": [
    {
      "graph_key": "4q_star_vncorbits_0",
      "orientation": "idididid",
      "max_clique_size": 2,
      "max_clique": ["0011", "0101"],
      "admissible_vertex_count": 6,
      "hybrid_kl_valid": true,
      ...
    },
    ...
  ]
}
```

## Usage Examples

### Run with Checkpointing (Default)

```bash
python ad_hybrid_code_search.py
```

Saves checkpoint every 50 results. On interrupt and rerun, resumes automatically.

### Disable Checkpointing

In `ad_hybrid_code_search.py` main():
```python
results = run_exhaustive_search(n=4, error_set_type="E1", verbose=True, checkpoint_interval=0)
```

### Clear Checkpoint and Start Fresh

```bash
python checkpoint_demo.py
# Answer 'y' to clear checkpoint
```

Or manually:
```bash
rm checkpoints/search_n4_E1.json
```

### Adjust Checkpoint Interval

In `ad_hybrid_code_search.py` main():
```python
# Save every 25 results (more frequent)
results = run_exhaustive_search(n=4, error_set_type="E1", verbose=True, checkpoint_interval=25)

# Save every 200 results (less frequent, faster)
results = run_exhaustive_search(n=4, error_set_type="E1", verbose=True, checkpoint_interval=200)
```

## Key Features

| Feature | Benefit |
|---------|---------|
| **Automatic Loading** | No manual intervention required on resume |
| **Combination Tracking** | Prevents re-computing (graph, orientation) pairs |
| **Progressive Saving** | Data saved every N results, not just at the end |
| **Zero Data Loss** | All processed combinations remembered even after interruption |
| **Result Merging** | New results combined with previous results automatically |
| **Space-Efficient** | Only stores (graph_key, orientation_str) pairs, not full computation |

## Performance Impact

### Checkpoint Save Overhead
- **Per save**: ~50-200ms for n=4 (324 results, 204 KB file)
- **Interval**: Default 50 results = ~1% overhead
- **For n=6**: Checkpoint interval becomes more valuable (larger files, longer runs)

### When Checkpointing Pays Off

| Scenario | Checkpoint Value |
|----------|------------------|
| n=4, single run | Minimal (runs in ~5 seconds anyway) |
| n=6, full exhaustive | **High** (~10-30 minutes, may be interrupted) |
| n=8, partial search | **Very High** (many hours of computation) |
| Server/cluster runs | **Critical** (tolerance for unexpected interrupts) |

## Troubleshooting

### Checkpoint Not Loading

If you see messages like "Warning: Could not load checkpoint", the file may be corrupted:
```bash
rm checkpoints/search_n4_E1.json
python ad_hybrid_code_search.py  # Start fresh
```

### What Counts as "Processed"?

A combination is marked processed when:
1. Admissible vertices found + clique search completed ✓
2. NO admissible vertices (skipped) ✓
3. NO clique found (skipped) ✓

This means **even failed combinations** count toward progress, so no wasted recomputation.

### Partial Results in Checkpoint

The checkpoint saves results only for combinations that produced valid cliques (max_clique_size > 0). This is correct because:
- Skipped combinations don't need to be recomputed
- We only care about the final merged result set

## Advanced: Manual Resume Control

To implement custom resume logic:
```python
from ad_hybrid_code_search import load_checkpoint, run_exhaustive_search

# Load existing checkpoint
processed, results = load_checkpoint(n=6, error_set_type="E1")

if processed:
    print(f"Resuming: {len(processed)} combinations already done")
    print(f"Previous best clique: {max(r.max_clique_size for r in results)}")

# Run search (will automatically skip processed combinations)
new_results = run_exhaustive_search(n=6, error_set_type="E1")
```

## Next Steps

1. **Test on n=6**: Run exhaustive search with checkpoint
   ```bash
   python ad_hybrid_code_search.py  # (modify main to use n=6)
   ```
   Will save checkpoints every 50 results over ~10,000 combinations

2. **Monitor Checkpoint Size**: Use `inspect_checkpoint.py`
   ```bash
   python inspect_checkpoint.py
   ```

3. **Production Runs**: For n≥8, checkpoint becomes essential for fault tolerance
