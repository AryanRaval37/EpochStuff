
# ! Partly AI generated file alert

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models import MF, MLP, NeuMF

# === Configuration Variables ===
MODEL_TYPE = "neumf"  # Options: "mf", "mlp", "neumf"
MODEL_PATH = f"models/{MODEL_TYPE}_model.pt"
DATA_PATH = "interactions.csv"

# Evaluation setup
K = 10
NUM_NEG = 100
TEST_SIZE = 0.1
EVAL_SEED = 37
# ===============================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ! Load model function : AI Generated
def load_model_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint format not recognized. Expected dict with key 'model_state_dict'."
        )

    model_type = str(checkpoint.get("model_type", MODEL_TYPE)).lower()
    
    if model_type == "neumf":
        model, checkpoint = NeuMF.load_checkpoint(model_path, device=device)
        return model, checkpoint

    state_dict = checkpoint["model_state_dict"]
    num_users = int(checkpoint["num_users"])
    num_items = int(checkpoint["num_items"])
    emb_dim = int(checkpoint["emb_dim"])

    if model_type == "mf":
        model = MF(num_users=num_users, num_items=num_items, emb_dim=emb_dim).to(device)
    elif model_type == "mlp":
        hidden_dims = tuple(int(x) for x in checkpoint["hidden_dims"])
        dropout = float(checkpoint["dropout"])
        model = MLP(
            num_users=num_users,
            num_items=num_items,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


# Same as hitk given
def hit_at_k(model, test_df, full_df, k=10, num_neg=100, device=None):
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    hits = 0
    total = len(test_df)

    interacted_items = full_df.groupby("user_id")["item_id"].apply(set).to_dict()
    
    # NOTE: Important: here i made this unique but I think it was just to numpy as array.
    all_items = full_df["item_id"].unique()

    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            user_id = int(row.user_id)
            pos_item = int(row.item_id)

            user_seen = interacted_items.get(user_id, set())
            max_neg = min(num_neg, len(all_items) - len(user_seen))

            negatives = set()
            while len(negatives) < max_neg:
                neg_item = int(np.random.choice(all_items))
                if neg_item not in user_seen:
                    negatives.add(neg_item)

            item_list = [pos_item] + list(negatives)
            user_tensor = torch.full((len(item_list),), user_id, dtype=torch.long, device=device)
            item_tensor = torch.tensor(item_list, dtype=torch.long, device=device)

            scores = model(user_tensor, item_tensor)
            top_k = min(k, len(item_list))
            _, top_indices = torch.topk(scores, top_k)

            if 0 in top_indices.cpu().tolist():
                hits += 1

    return hits / max(total, 1)


def main():
    device = get_device()
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    data_path = Path(DATA_PATH)
    model_path = Path(MODEL_PATH)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df = pd.read_csv(data_path)
    model, checkpoint = load_model_checkpoint(model_path, device)

    # get the dataset splitting seed to match train test split of training
    split_seed = int(checkpoint.get("split_seed", 37))
    model_type = str(checkpoint.get("model_type", MODEL_TYPE)).lower()

    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=split_seed)

    score = hit_at_k(
        model=model,
        test_df=test_df,
        full_df=df,
        k=K,
        num_neg=NUM_NEG,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Hit@{K}: {score:.4f} (num_neg={NUM_NEG}, test_size={TEST_SIZE})")


if __name__ == "__main__":
    main()
