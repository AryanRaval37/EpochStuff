import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

import pandas as pd
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class InteractionDataset(Dataset):
    def __init__(self, df, num_negatives=4):
        self.users = torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long)
        self.items = torch.as_tensor(df["item_id"].to_numpy(), dtype=torch.long)
        
        self.num_negatives = num_negatives
        
        # Fast lookup set for each user
        self.user_interacted = df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.all_items = df["item_id"].unique()

    def __len__(self):
        return len(self.users)


    # NOTE: negative sampling moved in here because it was quite slow, now it will work natively with pytorch with multiple workers
    def __getitem__(self, idx):
        user = self.users[idx].item()
        pos_item = self.items[idx].item()

        user_seen = self.user_interacted.get(user, set())
        
        users_out = torch.full((1 + self.num_negatives,), user, dtype=torch.long)
        items_out = torch.empty(1 + self.num_negatives, dtype=torch.long)
        labels_out = torch.zeros(1 + self.num_negatives, dtype=torch.float32)

        items_out[0] = pos_item
        labels_out[0] = 1.0

        n_sampled = 0
        num_all_items = len(self.all_items)
        while n_sampled < self.num_negatives:
            neg_idx = np.random.randint(0, num_all_items)
            neg = self.all_items[neg_idx]
            if neg not in user_seen:
                n_sampled += 1
                items_out[n_sampled] = neg

        return users_out, items_out, labels_out

def move_batch_to_device(batch, device):
    users_batch, items_batch, labels_batch = batch
    user = users_batch.view(-1).to(device)
    item = items_batch.view(-1).to(device)
    label = labels_batch.view(-1).to(device)
    return user, item, label

def load_data(data_path="interactions.csv", split_seed=37, test_size=0.1):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=split_seed)
    num_users = int(df["user_id"].max()) + 1
    num_items = int(df["item_id"].max()) + 1
    return df, train_df, test_df, num_users, num_items

def train(
    model,
    train_df,
    *,
    num_negatives=4,
    batch_size=1024,
    epochs=15,
    lr=1e-3,
    weight_decay=0.0,
    device=None,
):
    if device is None:
        device = get_device()
        
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # weighted loss function : consider switching to BPR Loss - better for ranking tasks - will fit Hit K metric better
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([num_negatives], device=device))
    losses = []

    # NOTE: Moved the original function of negative sampling inside the data loader
    # Initialize Dataset once!
    train_ds = InteractionDataset(train_df, num_negatives=num_negatives)
    
    # Each __getitem__ returns (1 + num_negatives) examples, so we need to scale
    # down the loader batch_size to keep the effective number of gradient steps
    # per epoch the same as when each __getitem__ returned a single example.
    loader_batch_size = max(1, batch_size // (1 + num_negatives))
    
    epoch_loader = DataLoader(
        train_ds,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=False,
    )

    for epoch in range(epochs):
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
    df = pd.read_csv("interactions.csv")  # columns: user_id, movie_id
    
    print("Num users:", df['user_id'].nunique())
    print("Num items:", df['item_id'].nunique())
    print("Num interactions:", len(df))
    print("Device:", get_device())

    # Interactions per user
    user_counts = df['user_id'].value_counts()
    print("\n--- Interactions per user ---")
    print(user_counts.describe())

    # Interactions per item
    item_counts = df['item_id'].value_counts()
    print("\n--- Interactions per item ---")
    print(item_counts.describe())

    # So there are about 4 percent interactions observed and rest unobserved.
    # a median user interacts with 40 interactions so a a small set of heavy users contributes a lot to the data
    # a median item has 13 interactions max 501 and min 1 so atleast each item as some interaction. very spread out interactions...
