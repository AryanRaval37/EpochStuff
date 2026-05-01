import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=1.0 / emb_dim**0.5)
        nn.init.normal_(self.item_emb.weight, std=1.0 / emb_dim**0.5)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        user = user.long()
        item = item.long()
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)

        # Here an important detail: this is an MF model, therefore just weight all the elements equally
        # howver in a generalized MF, there is a learnable weight for each element of the product.
        dot = (user_vec * item_vec).sum(dim=1)
        user_bias = self.user_bias(user).squeeze(1)
        item_bias = self.item_bias(item).squeeze(1)
        logits = dot + user_bias + item_bias + self.global_bias
        return logits


class MLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        emb_dim=32,
        hidden_dims=(256, 128, 64, 32),
        dropout=0.2,
    ):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=1.0 / emb_dim**0.5)
        nn.init.normal_(self.item_emb.weight, std=1.0 / emb_dim**0.5)

        layers = []
        input_dim = emb_dim * 3
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, user, item):
        user = user.long()
        item = item.long()

        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)

        x = torch.cat([user_vec, item_vec, user_vec * item_vec], dim=1)
        x = self.mlp(x)
        logits = self.output(x).squeeze(1)
        return logits
    

# ! Some helper functions and documentation is AI generated.

"""
Neural Matrix Factorization (NeuMF)
Implementation based on:
  "Neural Collaborative Filtering"
  https://arxiv.org/abs/1708.05031
"""

from pathlib import Path
from typing import Optional, Sequence

def _build_mlp_tower(
    in_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> tuple[nn.Sequential, int]:
    """
    Build the MLP tower and return (module, output_dim).

    Architecture per the NCF paper: Linear → ReLU → Dropout, repeated.
    The paper uses a tower pattern where each hidden layer is half the
    previous — e.g. [256, 128, 64] — but we keep it configurable.
    """
    layers: list[nn.Module] = []
    current_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        current_dim = h_dim
    tower = nn.Sequential(*layers) if layers else nn.Identity()
    return tower, current_dim


def _init_embedding(emb: nn.Embedding, std: float = 0.01) -> None:
    nn.init.normal_(emb.weight, mean=0.0, std=std)


def _init_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization combining GMF and MLP pathways.

    Parameters
    ----------
    num_users : int
        Total number of users (max user_id + 1).
    num_items : int
        Total number of items (max item_id + 1).
    gmf_emb_dim : int
        Embedding dimension for the GMF pathway.
        The paper typically uses 8–32. Smaller than MLP is fine.
    mlp_emb_dim : int
        Embedding dimension for the MLP pathway.
        Input to the first MLP layer is mlp_emb_dim * 3 (user ∥ item ∥ user⊙item).
    hidden_dims : sequence of int
        Hidden layer sizes for the MLP tower (excluding input/output).
        Paper default: (128, 64, 32). Tower should be monotonically
        decreasing (each layer half the previous) per the paper.
    dropout : float
        Dropout probability applied after each ReLU in the MLP tower.
        0.0 to disable. Paper uses 0.0–0.5; 0.2 is a sensible default.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_emb_dim: int = 32,
        mlp_emb_dim: int = 32,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gmf_emb_dim = gmf_emb_dim
        self.mlp_emb_dim = mlp_emb_dim
        self._hidden_dims = tuple(hidden_dims)
        self._dropout = dropout

        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_dim)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)

        self.mlp_tower, mlp_out_dim = _build_mlp_tower(
            in_dim=mlp_emb_dim * 3,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self._mlp_out_dim = mlp_out_dim  # needed for checkpoint and pretraining

        # ── Prediction layer ───────────────────────────────────────────────
        # The paper uses a single linear layer with NO bias and NO sigmoid
        # Though - why im not sure (also why no bias?) Maybe I could play around with adding more mlp layers here
        # NOTE: Idea of adding a transformer model somewhere around here or atlesat after the pretraining and having atleast the rough embeddings being leart
        # (sigmoid happens implicitly in BCEWithLogitsLoss).
        # Input size = GMF output (gmf_emb_dim) + MLP output (mlp_out_dim).
        self.predict = nn.Linear(gmf_emb_dim + mlp_out_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Default random initialisation used when NOT pretraining.

        Embeddings: small normal (prevents exploding logits at init).
        Linear layers: Xavier uniform (maintains variance through layers).
        Prediction layer: uniform 0.01
        """
        for emb in (
            self.gmf_user_emb, self.gmf_item_emb,
            self.mlp_user_emb, self.mlp_item_emb,
        ):
            _init_embedding(emb, std=0.01)

        for module in self.mlp_tower.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)

        # Neutral init: equal weight to every output dimension
        nn.init.constant_(self.predict.weight, 1.0 / (self.gmf_emb_dim + self._mlp_out_dim))

    def init_from_pretrained(
        self,
        gmf_model: nn.Module,
        mlp_model: nn.Module,
        alpha: float = 0.5,
        freeze_embeddings: bool = False,
    ) -> None:
        """
        Initialise NeuMF from separately pretrained GMF (MF) and MLP models.

        Strategy (paper Section 4.2):
          1. Copy GMF embedding weights → NeuMF GMF embeddings.
          2. Copy MLP embedding weights → NeuMF MLP embeddings.
          3. Copy MLP tower weights directly.
          4. Blend the two prediction heads into NeuMF's single predict layer:
               h = [ alpha * ones(gmf_emb_dim)  ‖  (1-alpha) * h_mlp ]
             - The GMF part is initialised to uniform weights (alpha * 1)
               because classic MF uses an equal sum, so ones is the natural
               prior for the GMF pathway.
             - The MLP part copies the pretrained MLP output head scaled by
               (1-alpha).
             - alpha=0.5 gives equal contribution; tune if one model is
               significantly better than the other.

        Parameters
        ----------
        gmf_model : nn.Module
            Pretrained MF model (expects attributes user_emb, item_emb).
        mlp_model : nn.Module
            Pretrained MLP model (expects attributes user_emb, item_emb,
            mlp, output).
        alpha : float
            Trade-off weight. 0 = MLP only, 1 = GMF only. Default 0.5.
        freeze_embeddings : bool
            If True, freeze all embedding layers during fine-tuning.
            Useful when the pretrained embeddings are high quality and
            you only want to fine-tune the upper layers.
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

        # 1. GMF embeddings
        self.gmf_user_emb.weight.data.copy_(gmf_model.user_emb.weight.data)
        self.gmf_item_emb.weight.data.copy_(gmf_model.item_emb.weight.data)

        # 2. MLP embeddings
        self.mlp_user_emb.weight.data.copy_(mlp_model.user_emb.weight.data)
        self.mlp_item_emb.weight.data.copy_(mlp_model.item_emb.weight.data)

        # 3. MLP tower
        self.mlp_tower.load_state_dict(mlp_model.mlp.state_dict())

        # NOTE: InTeReStInG way to blend the weights - convex combination of the weights?? why?
        # Look more into this: This is a whole big thing about doing this (DARE, SLERP, WIDEN, Git Re-basin or somethings....)
        # 4. Prediction layer blend
        #    gmf part: alpha * ones  →  shape [1, gmf_emb_dim]
        gmf_h = alpha * torch.ones(1, self.gmf_emb_dim)
        #    mlp part: (1-alpha) * pretrained output weights  →  shape [1, mlp_out_dim]
        mlp_h = (1.0 - alpha) * mlp_model.output.weight.data.clone()
        combined_h = torch.cat([gmf_h, mlp_h], dim=1)  # [1, gmf_emb_dim + mlp_out_dim]
        self.predict.weight.data.copy_(combined_h)

        if freeze_embeddings:
            for emb in (
                self.gmf_user_emb, self.gmf_item_emb,
                self.mlp_user_emb, self.mlp_item_emb,
            ):
                emb.weight.requires_grad_(False)

        print(
            f"[NeuMF] Initialised from pretrained GMF + MLP "
            f"(alpha={alpha}, freeze_embeddings={freeze_embeddings})"
        )

    def unfreeze_all(self) -> None:
        """Re-enable gradients on all parameters (useful after pretraining warmup)."""
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, user: torch.Tensor, item: torch.Tensor):
        # ! Documentation is AI generated.
        """
        Parameters
        ----------
        user : LongTensor [B]
        item : LongTensor [B]

        Returns
        -------
        logits : FloatTensor [B]
            Raw logits (no sigmoid). Pass to BCEWithLogitsLoss directly.

        Forward computation:
          GMF branch:
            p_u = gmf_user_emb(user)          # [B, gmf_emb_dim]
            q_i = gmf_item_emb(item)          # [B, gmf_emb_dim]
            φ_gmf = p_u ⊙ q_i                 # element-wise product [B, gmf_emb_dim]

          MLP branch:
            u_vec = mlp_user_emb(user)        # [B, mlp_emb_dim]
            i_vec = mlp_item_emb(item)        # [B, mlp_emb_dim]
            z = [u_vec ∥ i_vec]              # concat [B, mlp_emb_dim*2]
            φ_mlp = MLP_tower(z)             # [B, hidden_dims[-1]]

          Prediction:
            ŷ = predict( [φ_gmf ∥ φ_mlp] )   # linear [B, 1] → squeeze → [B]
        """
        user = user.long()
        item = item.long()

        # GMF pathway
        gmf_u = self.gmf_user_emb(user)   # [B, gmf_emb_dim]
        gmf_i = self.gmf_item_emb(item)   # [B, gmf_emb_dim]
        phi_gmf = gmf_u * gmf_i           # element-wise product [B, gmf_emb_dim]

        # MLP pathway
        mlp_u = self.mlp_user_emb(user)   # [B, mlp_emb_dim]
        mlp_i = self.mlp_item_emb(item)   # [B, mlp_emb_dim]
        mlp_in = torch.cat([mlp_u, mlp_i, mlp_u * mlp_i], dim=1)  # [B, mlp_emb_dim*3]
        phi_mlp = self.mlp_tower(mlp_in)  # [B, hidden_dims[-1]]

        # Concatenate both pathways
        combined = torch.cat([phi_gmf, phi_mlp], dim=1)  # [B, gmf_emb_dim + mlp_out_dim]

        logits = self.predict(combined).squeeze(1)  # [B]
        return logits

    # ! AI generated function alert
    def save_checkpoint(self, model_path: str | Path, **train_meta) -> None:
        """
        Save NeuMF checkpoint with all config needed to reconstruct the model.

        Parameters
        ----------
        model_path : str or Path
        **train_meta : arbitrary training metadata (epochs, lr, etc.)
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_type": "neumf",
            "num_users": self.gmf_user_emb.num_embeddings,
            "num_items": self.gmf_item_emb.num_embeddings,
            "gmf_emb_dim": self.gmf_emb_dim,
            "mlp_emb_dim": self.mlp_emb_dim,
            "mlp_out_dim": self._mlp_out_dim,
            "hidden_dims": list(self._hidden_dims),
            "dropout": self._dropout,
            **train_meta,
        }
        torch.save(checkpoint, model_path)
        print(f"[NeuMF] Saved checkpoint → {model_path}")

    # ! AI generated function alert
    @classmethod
    def load_checkpoint(
        cls,
        model_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> tuple["NeuMF", dict]:
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        checkpoint = torch.load(
            model_path, map_location=device, weights_only=True
        )

        hidden_dims = tuple(checkpoint.get("hidden_dims", (128, 64, 32)))
        dropout = float(checkpoint.get("dropout", 0.2))

        model = cls(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            gmf_emb_dim=checkpoint["gmf_emb_dim"],
            mlp_emb_dim=checkpoint["mlp_emb_dim"],
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[NeuMF] Loaded checkpoint ← {model_path}  (device={device})")
        return model, checkpoint

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"NeuMF(\n"
            f"  gmf_emb_dim={self.gmf_emb_dim}, mlp_emb_dim={self.mlp_emb_dim},\n"
            f"  mlp_out_dim={self._mlp_out_dim},\n"
            f"  total_params={total:,}, trainable={trainable:,}\n"
            f")"
        )
