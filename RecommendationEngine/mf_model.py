import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        # Has no bias term: add bias term here.
        nn.init.normal_(self.user_emb.weight, std=1.0 / emb_dim**0.5)
        nn.init.normal_(self.item_emb.weight, std=1.0 / emb_dim**0.5)

    def forward(self, user, item):
        user = user.long()
        item = item.long()
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        logits = (user_vec * item_vec).sum(dim=1)
        return logits


class MLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        emb_dim=32,
        hidden_dims=(512, 256, 32),
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
