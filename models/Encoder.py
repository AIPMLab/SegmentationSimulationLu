import torch
import torch.nn as nn
import torchvision
from timm.models.layers import trunc_normal_
from utils import *
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict, namedtuple
from typing import Any, Mapping, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the namedtuple _IncompatibleKeys
_IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

class Attention(nn.Module):
    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()
        # 确保窗口大小来自配置
        self.window_size = window_size

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size=224, in_channels=1):
        super().__init__()

        model_path = config.swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size, in_chans=in_channels, window_size=config.window_size)

        if model_path:
            checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
            unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                          "patch_embed.norm.bias",
                          "head.weight", "head.bias", "layers.0.downsample.norm.weight",
                          "layers.0.downsample.norm.bias",
                          "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                          "layers.1.downsample.norm.bias",
                          "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                          "layers.2.downsample.norm.bias",
                          "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

            for key in list(checkpoint.keys()):
                if key in unexpected or 'layers.3' in key:
                    del checkpoint[key]
            self.load_state_dict(checkpoint, strict=False)

        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]

        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        self.p1_pm = PatchMerging((config.image_size // config.patch_size, config.image_size // config.patch_size),
                                  config.swin_pyramid_fm[0])
        if model_path:
            state_dict = self.p1_pm.state_dict()
            if 'reduction.weight' in state_dict and "layers.0.downsample.reduction.weight" in checkpoint:
                state_dict['reduction.weight'][:] = checkpoint["layers.0.downsample.reduction.weight"]
            if 'norm.weight' in state_dict and "layers.0.downsample.norm.weight" in checkpoint:
                state_dict['norm.weight'][:] = checkpoint["layers.0.downsample.norm.weight"]
            if 'norm.bias' in state_dict and "layers.0.downsample.norm.bias" in checkpoint:
                state_dict['norm.bias'][:] = checkpoint["layers.0.downsample.norm.bias"]

        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)

        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_pm = PatchMerging(
            (config.image_size // config.patch_size // 2, config.image_size // config.patch_size // 2),
            config.swin_pyramid_fm[1])
        if model_path:
            state_dict = self.p2_pm.state_dict()
            if 'reduction.weight' in state_dict and "layers.1.downsample.reduction.weight" in checkpoint:
                state_dict['reduction.weight'][:] = checkpoint["layers.1.downsample.reduction.weight"]
            if 'norm.weight' in state_dict and "layers.1.downsample.norm.weight" in checkpoint:
                state_dict['norm.weight'][:] = checkpoint["layers.1.downsample.norm.weight"]
            if 'norm.bias' in state_dict and "layers.1.downsample.norm.bias" in checkpoint:
                state_dict['norm.bias'][:] = checkpoint["layers.1.downsample.norm.bias"]

        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for i in range(5):
            x = self.resnet_layers[i](x)

        # Level 1
        fm1 = x
        fm1_ch = self.p1_ch(x)
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped + sw1
        norm1 = self.norm_1(sw1_skipped)
        sw1_CLS = self.avgpool_1(norm1.transpose(1, 2))
        sw1_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw1_CLS)
        fm1_sw1 = self.p1_pm(sw1_skipped)

        # Level 2
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        fm2 = self.p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        fm2_sw2_skipped = fm2_reshaped + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)

        # Level 3
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        fm3 = self.p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        fm3_sw3_skipped = fm3_reshaped + fm2_sw3
        norm2 = self.norm_2(fm3_sw3_skipped)
        sw3_CLS = self.avgpool_2(norm2.transpose(1, 2))
        sw3_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw3_CLS)

        return [torch.cat((sw1_CLS_reshaped, sw1_skipped), dim=1),
                torch.cat((sw3_CLS_reshaped, fm3_sw3_skipped), dim=1)]

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata['assign_to_params_buffers'] = assign
            module_state_dict = module.state_dict()

            for name, param in local_state_dict.items():
                if name in module_state_dict:
                    if module_state_dict[name].shape != param.shape:
                        print(f"Skipping '{name}' due to shape mismatch: {param.shape} vs {module_state_dict[name].shape}")
                        local_state_dict[name] = module_state_dict[name]

            module._load_from_state_dict(
                local_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)


class All2Cross(nn.Module):
    def __init__(self, config, img_size=224, in_chans=1, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures(config=config, img_size=img_size, in_channels=in_chans)

        n_p1 = (img_size // 4) ** 2  # Using 4 as the patch size here
        n_p2 = (img_size // 16) ** 2  # Using 16 as the patch size here
        num_patches = (n_p1, n_p2)
        self.num_branches = 2

        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                                  qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate,
                                  attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        xs = self.pyramid(x)

        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs

