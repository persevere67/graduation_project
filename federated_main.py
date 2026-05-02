import argparse
import os
import pickle
import shutil
import subprocess
import sys
from collections import OrderedDict

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import MINDDataset
from experiments.experiment_logger import log_round_metric, log_experiment_summary, make_run_id
from models import NewsRecommender


os.environ["RAY_worker_register_timeout_seconds"] = "600"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Run federated training with Flower.")
    parser.add_argument("--num-clients", type=int, default=50, help="Total number of simulated clients.")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds.")
    parser.add_argument("--fraction-fit", type=float, default=0.2, help="Client sampling ratio per round.")
    parser.add_argument("--batch-size", type=int, default=32, help="Local training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Local learning rate.")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise scale for DP-style experiments.")
    parser.add_argument(
        "--data-file",
        type=str,
        default=os.path.join(PROCESSED_DIR, "federated_data.pkl"),
        help="Path to the federated client data pickle file.",
    )
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
        help="Disable attention for ablation experiments.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a checkpoint every N rounds. Set to 0 to disable periodic saving.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment tag used in checkpoint names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for the federated experiment. Use cpu for Windows stability.",
    )
    parser.add_argument("--client-gpu", type=float, default=0.0, help="GPU resources per simulated client.")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client.")
    return parser.parse_args()


def build_experiment_name(args):
    if args.experiment_name:
        return args.experiment_name
    encoder = "attn" if args.use_attention else "mean"
    dp = f"dp{args.sigma:g}" if args.sigma > 0 else "nodp"
    data_tag = "noniid" if "noniid" in os.path.basename(args.data_file).lower() else "iid"
    return f"fed_{encoder}_{data_tag}_{dp}_r{args.num_rounds}"


def stop_stale_ray():
    ray_exe = shutil.which("ray")
    if ray_exe is None and os.name == "nt":
        candidate = os.path.join(os.path.dirname(sys.executable), "Scripts", "ray.exe")
        if os.path.exists(candidate):
            ray_exe = candidate

    if ray_exe is None:
        print("Ray CLI not found, skipping stale Ray cleanup.")
        return

    try:
        print("Stopping stale Ray processes before simulation...")
        subprocess.run(
            [ray_exe, "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Ray cleanup skipped due to OS error: {exc}")


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    ranks = np.where(y_true_sorted == 1)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, lr, sigma, device, news_embeddings):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        self.sigma = sigma
        self.device = device
        self.news_embeddings = news_embeddings

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for hist_idx, cand_idx, label in self.train_loader:
            hist_idx = hist_idx.to(self.device)
            cand_idx = cand_idx.to(self.device)
            label = label.to(self.device)
            hist_vecs = self.news_embeddings[hist_idx]
            cand_vecs = self.news_embeddings[cand_idx]

            optimizer.zero_grad()
            output = self.model(hist_vecs, cand_vecs)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        if self.sigma > 0:
            dp_params = []
            for param in new_params:
                noise = np.random.normal(0, self.sigma, param.shape)
                dp_params.append(param + noise)
            return dp_params, len(self.train_loader.dataset), {}

        return new_params, len(self.train_loader.dataset), {}


def get_evaluate_fn(args, experiment_name, device, news_embeddings, dev_data, news_id_dict):
    local_val_dataset = MINDDataset(dev_data, news_id_dict)
    run_id = make_run_id(experiment_name)
    best = {"round": None, "auc": None, "mrr": None}

    def evaluate(server_round, parameters, config):
        eval_rounds = {0, 1, args.num_rounds}
        if args.save_every > 0:
            eval_rounds.update(range(args.save_every, args.num_rounds + 1, args.save_every))
        if server_round not in eval_rounds:
            return 0.0, {}

        model = NewsRecommender(embedding_dim=384, use_attention=args.use_attention).to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        all_auc = []
        all_mrr = []
        test_loader = DataLoader(local_val_dataset, batch_size=256, shuffle=False)

        print(f"Evaluating round {server_round}...")
        with torch.no_grad():
            for hist_idx, cand_idx, label in test_loader:
                hist_idx = hist_idx.to(device)
                cand_idx = cand_idx.to(device)
                label = label.to(device)
                hist_vecs = news_embeddings[hist_idx]
                cand_vecs = news_embeddings[cand_idx]
                output = model(hist_vecs, cand_vecs)

                scores_batch = output.cpu().numpy()
                labels_batch = label.cpu().numpy()
                for i in range(len(labels_batch)):
                    y_true = np.zeros(scores_batch[i].shape)
                    y_true[labels_batch[i]] = 1
                    try:
                        all_auc.append(roc_auc_score(y_true, scores_batch[i]))
                        all_mrr.append(mrr_score(y_true, scores_batch[i]))
                    except ValueError:
                        continue

        mean_auc = np.mean(all_auc)
        mean_mrr = np.mean(all_mrr)
        print(f"[Round {server_round}] AUC: {mean_auc:.4f} | MRR: {mean_mrr:.4f}")

        if server_round > 0 and (args.save_every > 0 and server_round % args.save_every == 0 or server_round == args.num_rounds):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}_round_{server_round}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to: {save_path}")
        else:
            save_path = ""

        log_round_metric(
            {
                "run_id": run_id,
                "experiment_id": experiment_name,
                "experiment_type": "federated",
                "generated_at": pd.Timestamp.now().isoformat(),
                "round": server_round,
                "loss": "",
                "auc": mean_auc,
                "mrr": mean_mrr,
                "sigma": args.sigma,
                "use_attention": args.use_attention,
                "num_clients": args.num_clients,
                "num_rounds": args.num_rounds,
                "batch_size": args.batch_size,
                "checkpoint_path": save_path,
                "notes": "server-side evaluation round",
            }
        )

        if best["auc"] is None or mean_auc > best["auc"]:
            best["round"] = server_round
            best["auc"] = mean_auc
            best["mrr"] = mean_mrr

        if server_round == args.num_rounds:
            log_experiment_summary(
                {
                    "run_id": run_id,
                    "experiment_id": experiment_name,
                    "experiment_type": "federated",
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "model_name": experiment_name,
                    "model_path": save_path,
                    "best_round": best["round"],
                    "final_round": server_round,
                    "final_loss": "",
                    "final_auc": mean_auc,
                    "final_mrr": mean_mrr,
                    "use_attention": args.use_attention,
                    "sigma": args.sigma,
                    "num_clients": args.num_clients,
                    "num_rounds": args.num_rounds,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs": "",
                    "notes": "federated training finished",
                }
            )

        return 0.0, {"AUC": mean_auc, "MRR": mean_mrr}

    return evaluate


def make_client_fn(args, device, news_embeddings, federated_data, news_id_dict):
    def client_fn(cid: str):
        model = NewsRecommender(embedding_dim=384, use_attention=args.use_attention).to(device)
        client_df = federated_data[f"client_{cid}"]
        dataset = MINDDataset(client_df, news_id_dict)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return FlowerClient(
            cid,
            model,
            train_loader,
            lr=args.lr,
            sigma=args.sigma,
            device=device,
            news_embeddings=news_embeddings,
        )

    return client_fn


def main():
    args = parse_args()
    stop_stale_ray()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Loading processed artifacts...")
    news_embeddings = torch.FloatTensor(np.load(os.path.join(PROCESSED_DIR, "news_embeddings.npy"))).to(device)
    with open(os.path.join(PROCESSED_DIR, "news_id_dict.pkl"), "rb") as f:
        news_id_dict = pickle.load(f)
    with open(args.data_file, "rb") as f:
        federated_data = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "dev_data_all.pkl"), "rb") as f:
        dev_data = pickle.load(f)

    experiment_name = build_experiment_name(args)
    min_available_clients = max(1, int(np.ceil(args.num_clients * args.fraction_fit)))

    print(
        f"Starting federated simulation | experiment={experiment_name} | use_attention={args.use_attention} "
        f"| sigma={args.sigma} | clients={args.num_clients} | rounds={args.num_rounds} | device={args.device}"
    )
    print(f"Using federated data file: {args.data_file}")

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(args, experiment_name, device, news_embeddings, dev_data, news_id_dict),
        fraction_fit=args.fraction_fit,
        min_available_clients=min_available_clients,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
    )

    fl.simulation.start_simulation(
        client_fn=make_client_fn(args, device, news_embeddings, federated_data, news_id_dict),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_gpus": args.client_gpu, "num_cpus": args.client_cpu},
    )


if __name__ == "__main__":
    main()
