import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MINDDataset
from experiments.experiment_logger import log_round_metric, log_experiment_summary, make_run_id
from models import NewsRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="Train the centralized baseline model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=True,
        help="Enable multi-head attention in the user encoder.",
    )
    parser.add_argument(
        "--disable-attention",
        dest="use_attention",
        action="store_false",
        help="Disable multi-head attention for ablation experiments.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional checkpoint filename. Defaults to a name derived from the experiment config.",
    )
    return parser.parse_args()


def build_output_name(args):
    if args.output_name:
        return args.output_name
    suffix = "attn" if args.use_attention else "mean"
    return f"centralized_{suffix}_e{args.epochs}.pth"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading precomputed news embeddings...")
    news_embeddings = np.load(os.path.join(processed_dir, "news_embeddings.npy"))
    news_embeddings = torch.FloatTensor(news_embeddings).to(device)

    print("Loading training data...")
    with open(os.path.join(processed_dir, "federated_data.pkl"), "rb") as f:
        federated_data = pickle.load(f)
    with open(os.path.join(processed_dir, "news_id_dict.pkl"), "rb") as f:
        news_id_dict = pickle.load(f)

    all_train_df = pd.concat(federated_data.values(), ignore_index=True)
    train_dataset = MINDDataset(all_train_df, news_id_dict)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NewsRecommender(embedding_dim=384, use_attention=args.use_attention).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    experiment_id = f"centralized_{'attn' if args.use_attention else 'mean'}_e{args.epochs}"
    run_id = make_run_id(experiment_id)

    model.train()
    print(
        f"Starting centralized training | use_attention={args.use_attention} "
        f"| epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}"
    )
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (hist_idx, cand_idx, label) in enumerate(train_loader):
            hist_idx = hist_idx.to(device)
            cand_idx = cand_idx.to(device)
            label = label.to(device)

            hist_vecs = news_embeddings[hist_idx]
            cand_vecs = news_embeddings[cand_idx]

            optimizer.zero_grad()
            scores = model(hist_vecs, cand_vecs)
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} finished | Avg Loss: {epoch_loss:.4f}")
        log_round_metric(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "experiment_type": "centralized_train",
                "generated_at": pd.Timestamp.now().isoformat(),
                "round": epoch + 1,
                "loss": epoch_loss,
                "auc": "",
                "mrr": "",
                "sigma": 0,
                "use_attention": args.use_attention,
                "num_clients": "",
                "num_rounds": args.epochs,
                "batch_size": args.batch_size,
                "checkpoint_path": "",
                "notes": "epoch average training loss",
            }
        )

    output_name = build_output_name(args)
    save_path = os.path.join(checkpoint_dir, output_name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to: {save_path}")
    log_experiment_summary(
        {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_type": "centralized_train",
            "generated_at": pd.Timestamp.now().isoformat(),
            "model_name": experiment_id,
            "model_path": save_path,
            "best_round": "",
            "final_round": args.epochs,
            "final_loss": epoch_loss,
            "final_auc": "",
            "final_mrr": "",
            "use_attention": args.use_attention,
            "sigma": 0,
            "num_clients": "",
            "num_rounds": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "notes": "centralized training finished",
        }
    )


if __name__ == "__main__":
    train(parse_args())
