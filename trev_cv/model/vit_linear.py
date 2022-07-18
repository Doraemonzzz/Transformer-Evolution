from .model import ViT

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

@register_model
def linear_vit_relu_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="linear", act_fun="relu", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def linear_vit_relu2_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="linear", act_fun="relu2", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def linear_vit_sigmoid_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="linear", act_fun="sigmoid", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def linear_vit_silu_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="linear", act_fun="silu", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def linear_vit_1_elu_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, attn_type="linear", act_fun="1+elu", **kwargs)
    model.default_cfg = _cfg()
    return model