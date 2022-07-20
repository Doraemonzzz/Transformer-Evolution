from .model import ViT

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

@register_model
def performer_vit_tiny_64(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="performer", proj_dim=64, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_vit_tiny_128(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="performer", proj_dim=128, **kwargs)
    model.default_cfg = _cfg()
    return model