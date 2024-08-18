import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from collections import OrderedDict, namedtuple
from typing import Any, Mapping, List
from models.Encoder import All2Cross
from models.Decoder import ConvUpsample, SegmentationHead

_IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=1, n_classes=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]  # Set patch sizes as required
        self.n_classes = n_classes

        # Ensure All2Cross uses the correct config with window_size
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            h, w = (self.img_size // self.patch_size[i], self.img_size // self.patch_size[i])
            embed = Rearrange('b (h w) d -> b d h w', h=h, w=w)(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)

        return out

    def load_from(self, pretrained_path):
        if pretrained_path is None:
            return
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cuda'))['model']
        self.load_state_dict(checkpoint, strict=False)

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
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata['assign_to_params_buffers'] = assign
            module_state_dict = module.state_dict()

            # Modify this part to ignore shape mismatches
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

            # Note that the hook can modify missing_keys and unexpected_keys.
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
