import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# import model file
from mf_model import MF, MLP

import pandas as pd
import numpy as np


# select device added here for compatibility, i'll just be using mps
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SPLIT_SEED = 37

df = pd.read_csv("interactions.csv")  # columns: user_id, movie_id

print("Num users:", df['user_id'].nunique())
print("Num items:", df['item_id'].nunique())
print("Num interactions:", len(df))
print("Device:", DEVICE)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=SPLIT_SEED)

# ! AI modified for faster negative sampling, vectorization used.
def negative_sampling(df, num_negatives=4):
    user_interactions = df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_items = df["item_id"].unique()  # unique here too
    samples = []

    for row in df.itertuples(index=False):
        user = int(row.user_id)
        item = int(row.item_id)
        samples.append((user, item, 1))

        interacted = np.array(list(user_interactions.get(user, set())))
        available = np.setdiff1d(all_items, interacted)  # vectorized
        
        n = min(num_negatives, len(available))
        if n == 0:
            continue
            
        negs = np.random.choice(available, size=n, replace=False)  # no loop needed
        for neg in negs:
            samples.append((user, int(neg), 0))

    return pd.DataFrame(samples, columns=["user_id", "item_id", "label"])

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long)
        self.items = torch.as_tensor(df["item_id"].to_numpy(), dtype=torch.long)
        self.labels = torch.as_tensor(df["label"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return self.users.size(0)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def move_batch_to_device(batch, device=DEVICE):
    user, item, label = batch
    return user.to(device), item.to(device), label.to(device)

# Interactions per user
user_counts = df['user_id'].value_counts()
print(user_counts.describe())

# Interactions per item
item_counts = df['item_id'].value_counts()
print(item_counts.describe())

# So there are about 4 percent interactions observed and rest unobserved.
# a median user interacts with 40 interactions so a a small set of heavy users contributes a lot to the data
# a median item has 13 interactions max 501 and min 1 so atleast each item as some interaction. very spread out interactions...

def train(
    model,
    train_df,
    *,
    num_negatives=4,
    batch_size=1024,
    epochs=15,
    lr=1e-3,
    weight_decay=0.0,
    device=DEVICE,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # weighted loss function : consider switching to BPR Loss - better for ranking tasks - will fit Hit K metric better
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([num_negatives], device=device))
    losses = []

    for epoch in range(epochs):
        # Resample negatives every epoch so the model sees more unobserved space.
        epoch_train_data = negative_sampling(train_df, num_negatives=num_negatives)
        epoch_train_ds = InteractionDataset(epoch_train_data)
        epoch_loader = DataLoader(
            epoch_train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in epoch_loader:
            user, item, label = move_batch_to_device(batch, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(user, item)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            current_batch_size = user.size(0)
            total_loss += loss.detach() * current_batch_size
            total_examples += current_batch_size

        avg_loss = (total_loss / total_examples).item()
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        if device.type == "mps":
            torch.mps.empty_cache()

    return losses

# ! AI generated function alert
def save_model_checkpoint(
    model,
    model_path,
    *,
    model_type,
    num_users,
    num_items,
    emb_dim,
    hidden_dims,
    dropout,
    split_seed,
    num_negatives,
    epochs,
    lr,
):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": str(model_type),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "emb_dim": int(emb_dim),
        "hidden_dims": None if hidden_dims is None else list(hidden_dims),
        "dropout": None if dropout is None else float(dropout),
        "split_seed": int(split_seed),
        "num_negatives": int(num_negatives),
        "epochs": int(epochs),
        "lr": float(lr),
    }
    torch.save(checkpoint, model_path)
    print(f"Saved checkpoint: {model_path}")


if __name__ == "__main__":
    split_seed = SPLIT_SEED

    model_type = "mf"  # "mf" or "mlp"
    emb_dim = 48
    hidden_dims = (512, 256, 32)
    dropout = 0.2

    num_negatives = 100
    batch_size = 2048
    epochs = 30
    lr = 1e-3
    weight_decay = 1e-5

    num_users = int(df["user_id"].max()) + 1
    num_items = int(df["item_id"].max()) + 1

    if model_type == "mf":
        model = MF(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
        checkpoint_hidden_dims = None
        checkpoint_dropout = None
    elif model_type == "mlp":
        model = MLP(
            num_users=num_users,
            num_items=num_items,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        checkpoint_hidden_dims = hidden_dims
        checkpoint_dropout = dropout
    
    model_path = f"models/{model_type}_model.pt"

    train_losses = train(
        model,
        train_df,
        num_negatives=num_negatives,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=DEVICE,
    )

    save_model_checkpoint(
        model,
        model_path=model_path,
        model_type=model_type,
        num_users=num_users,
        num_items=num_items,
        emb_dim=emb_dim,
        hidden_dims=checkpoint_hidden_dims,
        dropout=checkpoint_dropout,
        split_seed=split_seed,
        num_negatives=num_negatives,
        epochs=epochs,
        lr=lr,
    )
