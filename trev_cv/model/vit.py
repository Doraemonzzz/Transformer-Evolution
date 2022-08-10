from .model import ViT

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

@register_model
def vit_tiny(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model

##### norm test
@register_model
def vit_tiny_simplermsnorm(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, norm_type="simplermsnorm", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_rmsnorm(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, norm_type="rmsnorm", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_gatedrmsnorm(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, norm_type="gatedrmsnorm", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_scalenorm(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, norm_type="scalenorm", **kwargs)
    model.default_cfg = _cfg()
    return model
##### norm test

##### head test
@register_model
def vit_tiny_one_head(pretrained=False, **kwargs):
    model = ViT(patch_size=16, dim=192, depth=12, heads=1, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model
##### head test

########## de rpe test
@register_model
def vit_tiny_rope_default(pretrained=False, **kwargs):
    kwargs["de_rpe_type"] = "rope"
    kwargs["theta_type"] = "default"
    kwargs["learned"] = False
    print(f"de_rpe_type {kwargs['de_rpe_type']}")
    print(f"theta_type {kwargs['theta_type']}")
    print(f"learned {kwargs['learned']}")
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_rope_default_learned(pretrained=False, **kwargs):
    kwargs["de_rpe_type"] = "rope"
    kwargs["theta_type"] = "default"
    kwargs["learned"] = True
    print(f"de_rpe_type {kwargs['de_rpe_type']}")
    print(f"theta_type {kwargs['theta_type']}")
    print(f"learned {kwargs['learned']}")
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_rope_random(pretrained=False, **kwargs):
    kwargs["de_rpe_type"] = "rope"
    kwargs["theta_type"] = "random"
    kwargs["learned"] = False
    print(f"de_rpe_type {kwargs['de_rpe_type']}")
    print(f"theta_type {kwargs['theta_type']}")
    print(f"learned {kwargs['learned']}")
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_rope_random_learned(pretrained=False, **kwargs):
    kwargs["de_rpe_type"] = "rope"
    kwargs["theta_type"] = "random"
    kwargs["learned"] = True
    print(f"de_rpe_type {kwargs['de_rpe_type']}")
    print(f"theta_type {kwargs['theta_type']}")
    print(f"learned {kwargs['learned']}")
    model = ViT(patch_size=16, dim=192, depth=12, heads=3, mlp_dim=192*4, **kwargs)
    model.default_cfg = _cfg()
    return model
########## de rpe test