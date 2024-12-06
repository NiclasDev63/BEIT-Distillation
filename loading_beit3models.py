import torch
from timm.models.registry import register_model
from torch import nn

from BEiT3_Code.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config

### Taken from BEit3 codebase
@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(self, args, num_classes, norm_layer=nn.LayerNorm, **kwargs):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask=None):
        outputs = self.beit3(
            textual_tokens=question,
            visual_tokens=image,
            text_padding_position=padding_mask,
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep)
###

def load_beit3_base(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = model_path
    model = beit3_base_patch16_480_vqav2()
    checkpoint = torch.load(model_dir, map_location=device)["model"]
    model.load_state_dict(checkpoint)
    model.to(device)
    return model


def load_beit3_large(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = model_path
    model = beit3_large_patch16_480_vqav2()
    checkpoint = torch.load(model_dir, map_location=device)["model"]
    model.load_state_dict(checkpoint)
    model.to(device)
    return model




def load_beit3_base_finetuned(checkpoint_path: str, is_compiled_model_checkpoint = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = beit3_base_patch16_480_vqav2()

    if is_compiled_model_checkpoint:
        preprocessed__model_checkpoint = {}
        for param_name, value in checkpoint["model_state_dict"].items():
            preprocessed__model_checkpoint[param_name.split("_orig_mod.")[1]] = value
        model.load_state_dict(preprocessed__model_checkpoint)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model
    
def load_beit3_large_finetuned(checkpoint_path: str, is_compiled_model_checkpoint = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = beit3_large_patch16_480_vqav2()

    if is_compiled_model_checkpoint:
        preprocessed__model_checkpoint = {}
        for param_name, value in checkpoint["model_state_dict"].items():
            preprocessed__model_checkpoint[param_name.split("_orig_mod.")[1]] = value
        model.load_state_dict(preprocessed__model_checkpoint)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    return model
