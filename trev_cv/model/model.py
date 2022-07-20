# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .utils import get_activation_fn
from .utils import get_attn
from .utils import get_ffn
from .utils import get_norm

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type=""):
        super().__init__()
        self.norm = get_norm(norm_type, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        dropout=0,
        attn_type="vanilla",
        ffn_type="vanilla",
        norm_type="layernorm",
        activation="gelu",
        # linear
        act_fun="relu",
        # performer, rfa
        proj_dim=64,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        Attention = self.get_attention(attn_type, dim, heads, dropout, act_fun, proj_dim)
        FeedForward = get_ffn(ffn_type)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention, norm_type=norm_type),
                PreNorm(dim, FeedForward(embed_dim=dim, hidden_dim=mlp_dim, act_dropout=dropout, final_dropout=dropout, activation=activation), norm_type=norm_type)
            ]))

    def get_attention(
        self, 
        attn_type, 
        dim,
        heads,
        dropout,
        act_fun,
        proj_dim,
    ):
        Attention = get_attn(attn_type)
        if attn_type == "vanilla":
            return Attention(embed_dim=dim, num_heads=heads, dropout=dropout)
        elif attn_type == "linear":
            return Attention(embed_dim=dim, num_heads=heads, dropout=dropout, act_fun=act_fun)
        elif attn_type == "performer" or attn_type == "rfa":
            return Attention(embed_dim=dim, num_heads=heads, proj_dim=proj_dim, dropout=dropout)
        else:
            return Attention(embed_dim=dim, num_heads=heads, dropout=dropout)
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

##### no cls
class ViT(nn.Module):
    def __init__(
        self, *, 
        image_size=224, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        channels=3, 
        dim_head=64,  
        emb_dropout=0,
        attn_type="vanilla",
        ffn_type="vanilla",
        norm_type="layernorm",
        activation="gelu",
        # linear
        act_fun="relu",
        # performer, rfa
        proj_dim=64,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            mlp_dim=mlp_dim, 
            dropout=0, 
            attn_type=attn_type,
            ffn_type=ffn_type,
            norm_type=norm_type,
            activation=activation,
            act_fun=act_fun,
            proj_dim=proj_dim,
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        print(f"attn_type {attn_type}")
        print(f"ffn_type {ffn_type}")
        print(f"norm_type {norm_type}")
        print(f"activation {activation}")
        print(f"num_heads {heads}")
        print(f"act_fun {act_fun}")
        print(f"proj_dim {proj_dim}")

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.to_latent(x)

        return self.mlp_head(x)