import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

behavior_path = os.path.join(DATASET_DIR, "MINDsmall_train", "behaviors.tsv")
train_news_path = os.path.join(DATASET_DIR, "MINDsmall_train", "news.tsv")
output_path = os.path.join(PROCESSED_DIR, "federated_data_noniid.pkl")


def process_impressions(imp_str, news_id_dict):
    pairs = [pair.split("-") for pair in imp_str.split()]
    return [(news_id_dict[p[0]], int(p[1])) for p in pairs if p[0] in news_id_dict]


def pad_groups(category_to_users, num_clients):
    ordered = sorted(category_to_users.items(), key=lambda item: len(item[1]), reverse=True)
    client_user_groups = [[] for _ in range(num_clients)]
    client_sizes = [0 for _ in range(num_clients)]

    for _, users in ordered:
        target_idx = client_sizes.index(min(client_sizes))
        client_user_groups[target_idx].extend(users)
        client_sizes[target_idx] += len(users)

    return client_user_groups


def main():
    print("Loading processed dictionaries and raw train files...")
    with open(os.path.join(PROCESSED_DIR, "news_id_dict.pkl"), "rb") as f:
        news_id_dict = pickle.load(f)

    behavior_columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
    news_columns = ["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"]

    df = pd.read_csv(behavior_path, sep="\t", names=behavior_columns)
    news_df = pd.read_csv(train_news_path, sep="\t", names=news_columns, encoding="utf-8", keep_default_na=False)

    df = df.dropna(subset=["History"])
    print("Parsing user impressions...")
    df["parsed_impressions"] = df["Impressions"].apply(lambda x: process_impressions(x, news_id_dict))
    df = df[df["parsed_impressions"].map(len) > 0]

    news_category = dict(zip(news_df["NewsID"], news_df["Category"]))
    user_category_counter = defaultdict(Counter)

    print("Building dominant category for each user...")
    for _, row in df.iterrows():
        history_ids = row["History"].split()
        for news_id in history_ids:
            category = news_category.get(news_id)
            if category:
                user_category_counter[row["UserID"]][category] += 1

    user_main_category = {}
    for user_id, counter in user_category_counter.items():
        if counter:
            user_main_category[user_id] = counter.most_common(1)[0][0]
        else:
            user_main_category[user_id] = "unknown"

    category_to_users = defaultdict(list)
    unique_users = sorted(df["UserID"].unique())
    for user_id in unique_users:
        category = user_main_category.get(user_id, "unknown")
        category_to_users[category].append(user_id)

    num_clients = 50
    client_user_groups = pad_groups(category_to_users, num_clients)

    print("Constructing Non-IID client datasets...")
    client_datasets = {}
    for i, users in enumerate(client_user_groups):
        user_set = set(users)
        client_df = df[df["UserID"].isin(user_set)]
        client_datasets[f"client_{i}"] = client_df[["UserID", "History", "parsed_impressions"]]

    with open(output_path, "wb") as f:
        pickle.dump(client_datasets, f)

    print(f"Saved Non-IID federated data to: {output_path}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of dominant categories: {len(category_to_users)}")


if __name__ == "__main__":
    main()
