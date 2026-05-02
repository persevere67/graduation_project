import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from experiments.experiment_logger import log_experiment_summary, make_run_id
from models import NewsRecommender


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained news recommendation model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "centralized_model.pth"),
        help="Path to the checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Enable multi-head attention in the evaluation model.",
    )
    parser.add_argument(
        "--disable-attention",
        dest="use_attention",
        action="store_false",
        help="Disable attention for evaluating ablation checkpoints.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Model",
        help="Display name used in logs.",
    )
    return parser.parse_args()


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    ranks = np.where(y_true_sorted == 1)[0]
    if len(ranks) > 0:
        return 1.0 / (ranks[0] + 1)
    return 0.0


def evaluate_model(model_path, model_name="Model", use_attention=True):
    print(f"\nEvaluating model: {model_name}")
    print(f"Checkpoint: {model_path}")
    print(f"use_attention={use_attention}")

    news_embeddings = torch.FloatTensor(np.load(os.path.join(PROCESSED_DIR, "news_embeddings.npy"))).to(device)
    with open(os.path.join(PROCESSED_DIR, "dev_data_all.pkl"), "rb") as f:
        dev_data = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "news_id_dict.pkl"), "rb") as f:
        news_id_dict = pickle.load(f)

    model = NewsRecommender(embedding_dim=384, use_attention=use_attention).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_auc = []
    all_mrr = []

    print(f"Loaded {len(dev_data)} evaluation rows.")
    with torch.no_grad():
        for idx, row in dev_data.iterrows():
            raw_hist = row["History"].split() if isinstance(row["History"], str) else row["History"]
            hist_ids = [news_id_dict[nid] for nid in raw_hist if nid in news_id_dict]

            if len(hist_ids) > 50:
                hist_ids = hist_ids[-50:]
            else:
                hist_ids = [0] * (50 - len(hist_ids)) + hist_ids

            hist_tensor = torch.tensor([hist_ids], dtype=torch.long).to(device)
            hist_vecs = news_embeddings[hist_tensor]

            impressions = row["parsed_impressions"]
            cand_ids = [imp[0] for imp in impressions]
            labels = [imp[1] for imp in impressions]

            if sum(labels) == 0 or sum(labels) == len(labels):
                continue

            cand_tensor = torch.tensor([cand_ids], dtype=torch.long).to(device)
            cand_vecs = news_embeddings[cand_tensor]

            scores = model(hist_vecs, cand_vecs).squeeze(0).cpu().numpy()
            all_auc.append(roc_auc_score(labels, scores))
            all_mrr.append(mrr_score(labels, scores))

            if (idx + 1) % 5000 == 0:
                print(f"Evaluated {idx + 1} rows | AUC: {np.mean(all_auc):.4f} | MRR: {np.mean(all_mrr):.4f}")

    final_auc = np.mean(all_auc)
    final_mrr = np.mean(all_mrr)
    print(f"\n{model_name} evaluation finished.")
    print(f"Final AUC: {final_auc:.4f}")
    print(f"Final MRR: {final_mrr:.4f}")
    print("-" * 40)
    experiment_id = f"eval_{os.path.splitext(os.path.basename(model_path))[0]}"
    run_id = make_run_id(experiment_id)
    log_experiment_summary(
        {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_type": "evaluation",
            "generated_at": pd.Timestamp.now().isoformat(),
            "model_name": model_name,
            "model_path": model_path,
            "best_round": "",
            "final_round": "",
            "final_loss": "",
            "final_auc": final_auc,
            "final_mrr": final_mrr,
            "use_attention": use_attention,
            "sigma": "",
            "num_clients": "",
            "num_rounds": "",
            "batch_size": "",
            "lr": "",
            "epochs": "",
            "notes": "offline evaluation finished",
        }
    )


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.model_path):
        evaluate_model(args.model_path, model_name=args.model_name, use_attention=args.use_attention)
    else:
        print(f"Checkpoint not found: {args.model_path}")
