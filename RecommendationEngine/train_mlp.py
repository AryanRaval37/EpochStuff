from train import load_data, train, save_model_checkpoint, get_device
from models import MLP

# Configuration Variables
DATA_PATH = "interactions.csv"
SPLIT_SEED = 37
EMB_DIM = 48
HIDDEN_DIMS = (64, 32, 16)
DROPOUT = 0.2

NUM_NEGATIVES = 100
BATCH_SIZE = 4096
EPOCHS = 20
LR = 1e-2
WEIGHT_DECAY = 1e-5

if __name__ == "__main__":
    device = get_device()
    print("Device:", device)

    # Load data
    df, train_df, test_df, num_users, num_items = load_data(DATA_PATH, split_seed=SPLIT_SEED)

    # Instantiate model
    model = MLP(
        num_users=num_users,
        num_items=num_items,
        emb_dim=EMB_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )
    model_type = "mlp"
    model_path = f"models/{model_type}_model.pt"

    # Train model
    train_losses = train(
        model,
        train_df,
        num_negatives=NUM_NEGATIVES,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    # Save checkpoint
    save_model_checkpoint(
        model,
        model_path=model_path,
        model_type=model_type,
        num_users=num_users,
        num_items=num_items,
        emb_dim=EMB_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        split_seed=SPLIT_SEED,
        num_negatives=NUM_NEGATIVES,
        epochs=EPOCHS,
        lr=LR,
    )
