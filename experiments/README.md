# Experiment Records

This directory stores structured experiment records for thesis tables, plots, and result verification.

## Auto-generated files

These files are created automatically after you run the current scripts:

- `runtime_round_metrics.csv`
  - written by `baseline_centralized.py` and `federated_main.py`
  - stores epoch-level or round-level metrics
- `runtime_experiment_summary.csv`
  - written by `baseline_centralized.py`, `federated_main.py`, and `evaluation.py`
  - stores final summaries for each reproducible run

## Manual records

If you also keep historical notes from Obsidian or handwritten experiment logs, they can be converted into separate CSV files in this directory. Those manual records should be treated as historical references until they are rechecked with the current unified scripts.

## Recommended workflow

1. Run training or evaluation with the current scripts.
2. Check `runtime_round_metrics.csv` for curves and intermediate metrics.
3. Check `runtime_experiment_summary.csv` for final results.
4. Use the runtime CSV files as the primary source for thesis figures and tables.
