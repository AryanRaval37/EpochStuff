import torch
from train import load_data, train, get_device
from models import NeuMF, MF, MLP

# Configuration Variables
DATA_PATH = "interactions.csv"
SPLIT_SEED = 37

# Model architecture config
GMF_EMB_DIM = 48
MLP_EMB_DIM = 48
HIDDEN_DIMS = (64, 32, 16)
DROPOUT = 0.2

# Pretraining options
PRETRAINED_MF_PATH = "models/mf_model.pt"  # Set to None if no pretraining
PRETRAINED_MLP_PATH = "models/mlp_model.pt" # Set to None if no pretraining
ALPHA_PRETRAIN = 0.5
FREEZE_EMBEDDINGS = False

# Training config
NUM_NEGATIVES = 100
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

if __name__ == "__main__":
    device = get_device()
    print("Device:", device)

    # Load data
    df, train_df, test_df, num_users, num_items = load_data(DATA_PATH, split_seed=SPLIT_SEED)

    # Instantiate model
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        gmf_emb_dim=GMF_EMB_DIM,
        mlp_emb_dim=MLP_EMB_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )
    
    # Load Pretrained Models if specified
    if PRETRAINED_MF_PATH and PRETRAINED_MLP_PATH:
        print("Loading pretrained models...")
        # Load MF
        mf_ckpt = torch.load(PRETRAINED_MF_PATH, map_location=device, weights_only=True)
        gmf_model = MF(
            num_users=mf_ckpt["num_users"],
            num_items=mf_ckpt["num_items"],
            emb_dim=mf_ckpt["emb_dim"],
        )
        gmf_model.load_state_dict(mf_ckpt["model_state_dict"])
        
        # Load MLP
        mlp_ckpt = torch.load(PRETRAINED_MLP_PATH, map_location=device, weights_only=True)
        mlp_model = MLP(
            num_users=mlp_ckpt["num_users"],
            num_items=mlp_ckpt["num_items"],
            emb_dim=mlp_ckpt["emb_dim"],
            hidden_dims=tuple(mlp_ckpt["hidden_dims"]),
            dropout=mlp_ckpt["dropout"],
        )
        mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
        
        model.init_from_pretrained(
            gmf_model=gmf_model,
            mlp_model=mlp_model,
            alpha=ALPHA_PRETRAIN,
            freeze_embeddings=FREEZE_EMBEDDINGS,
        )

    model_path = "models/neumf_model.pt"

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
    model.save_checkpoint(
        model_path=model_path,
        split_seed=SPLIT_SEED,
        num_negatives=NUM_NEGATIVES,
        epochs=EPOCHS,
        lr=LR,
    )
