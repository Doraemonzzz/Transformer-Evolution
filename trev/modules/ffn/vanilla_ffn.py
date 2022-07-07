import torch.nn as nn
from ..utils import get_activation_fn

class VanillaFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        act_dropout=0.0,
        final_dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act_fun = get_activation_fn(activation)
        self.act_dropout = nn.Dropout(act_dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.final_dropout = nn.Dropout(final_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fun(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.final_dropout(x)

        return x