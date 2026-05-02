import csv
import os
from datetime import datetime


EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_ROUND_METRICS_CSV = os.path.join(EXPERIMENTS_DIR, "runtime_round_metrics.csv")
RUNTIME_EXPERIMENT_SUMMARY_CSV = os.path.join(EXPERIMENTS_DIR, "runtime_experiment_summary.csv")


def make_run_id(experiment_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_id}_{timestamp}"


def append_csv_row(csv_path, fieldnames, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_round_metric(row):
    fieldnames = [
        "run_id",
        "experiment_id",
        "experiment_type",
        "generated_at",
        "round",
        "loss",
        "auc",
        "mrr",
        "sigma",
        "use_attention",
        "num_clients",
        "num_rounds",
        "batch_size",
        "checkpoint_path",
        "notes",
    ]
    append_csv_row(RUNTIME_ROUND_METRICS_CSV, fieldnames, row)


def log_experiment_summary(row):
    fieldnames = [
        "run_id",
        "experiment_id",
        "experiment_type",
        "generated_at",
        "model_name",
        "model_path",
        "best_round",
        "final_round",
        "final_loss",
        "final_auc",
        "final_mrr",
        "use_attention",
        "sigma",
        "num_clients",
        "num_rounds",
        "batch_size",
        "lr",
        "epochs",
        "notes",
    ]
    append_csv_row(RUNTIME_EXPERIMENT_SUMMARY_CSV, fieldnames, row)
