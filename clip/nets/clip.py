import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from transformers import BertModel, BertTokenizer

from .bert import Transformer
from .simple_tokenizer import SimpleTokenizer, tokenize
from .vit import VisionTransformer


# 新加Modifiedresnet50结构
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # print("x"+str(x.shape))
        # print("self.positional_embedding[:, None, :]" + str(self.positional_embedding[:, None, :].shape))
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC (50,2048)
        # print("x"+str(x.shape))
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet50(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        embed_dim = width * 32  # the ResNet feature dimension
        self.embed_dim = embed_dim

        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        # self.bn1 = nn.BatchNorm2d(num_features=512)
        # self.bn2 = nn.BatchNorm2d(num_features=1024)
        # self.bn3 = nn.BatchNorm2d(num_features=2048)
        # self.relu = nn.ReLU(inplace=True)
        #
        # self.conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # 输出形状: (512, 40, 40)
        # self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)  # 输出形状: (1024, 20, 20)
        # self.conv3 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1)  # 输出形状: (2048, 10, 10)
        # self.Adaptivepool = nn.AdaptiveAvgPool2d(output_size=(7, 7))  # 输出形状: (2048, 7, 7)


    def _make_layer(self, planes, blocks, stride=1):
            layers = [Bottleneck(self._inplanes, planes, stride)]

            self._inplanes = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                layers.append(Bottleneck(self._inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x):
        # print("输入的特征为"+str(x.shape))  #[8, 128, 80, 80]
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        # x = self.Adaptivepool(x)
        #需要维度为2048*7*7的特征图的输入
        # print("x的特征"+str(x.shape))
        x = self.attnpool(x) #1024
        return x

class CLIP(nn.Module):
    def __init__(
        self,
        bert_type           = "huggingface",

        embed_dim          = 512,
        # vision
        input_resolution   = 224,
        vision_layers      = 12,
        vision_width       = 768,
        vision_patch_size  = 32,

        # text
        context_length      = 77,
        transformer_layers  = 12,
        transformer_width   = 768,
        transformer_heads   = 12,
        vocab_size          = 49408,
        **kwargs
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads    = vision_width // 64
        self.visual     = VisionTransformer(
            input_resolution    = input_resolution,
            patch_size          = vision_patch_size,
            width               = vision_width,
            layers              = vision_layers,
            heads               = vision_heads,
            output_dim          = embed_dim
        )
        # print("visual"+str(self.visual))
        vision_heads1     = 32
        self.visual1      = ModifiedResNet50(
            layers          = [3,4,6,3],
            output_dim      = embed_dim,
            heads           = vision_heads1,
            input_resolution= input_resolution,
            width           = 64
        )

        self.bert_type = bert_type
        if bert_type == "openai":
            self.tokenizer          = SimpleTokenizer()
            self.transformer        = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.vocab_size             = vocab_size
            self.token_embedding        = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding   = nn.Parameter(torch.empty(self.context_length, transformer_width))
        elif bert_type == "huggingface":
            # kwargs['huggingface_model_name']
            self.tokenizer          = BertTokenizer.from_pretrained("/data/user7/wt/code/bert-base-cn")
            self.transformer        = BertModel.from_pretrained("/data/user7/wt/code/bert-base-cn")
            transformer_width       = self.transformer.config.hidden_size

        self.text_projection        = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        self.ln_final               = nn.LayerNorm(transformer_width)
        self.logit_scale            = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_image(self, image):
        # image.type(self.dtype)                torch.Size([4, 3, 224, 224])
        # self.visual(image.type(self.dtype))   torch.Size([4, 512])
        return self.visual1(image.type(self.dtype))


    def encode_text(self, text):
        if self.bert_type == "openai":
            text = tokenize(self.tokenizer, text).to(self.visual.conv1.weight.device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            # print("text"+str(x.shape)) torch.Size([2, 77, 768])
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        elif self.bert_type == "huggingface":
            x = self.tokenizer(text, return_tensors="pt", padding=True)
            input_ids       = x.input_ids.to(self.visual.conv1.weight.device)
            attention_mask  = x.attention_mask.to(self.visual.conv1.weight.device)
            token_type_ids  = x.token_type_ids.to(self.visual.conv1.weight.device)
            x = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
            x = self.ln_final(x).type(self.dtype)
            x = x @ self.text_projection
        # print("encode_text"+str(x))

        return x

    def forward(self, image, text):
        image_features  = self.encode_image(image) #512
        text_features   = self.encode_text(text) # 512
        # 打印输出看看为什么是相同的？
        # print("image_features" +str(image_features))
        # print("text_features" + str(text_features))

        image_features  = image_features / image_features.norm(dim=-1, keepdim=True)
        # print("image_features" + str(image_features))
        text_features   = text_features / text_features.norm(dim=-1, keepdim=True)
        # print("text_features" + str(text_features))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # print("image_features" + str(image_features))
        # print("text_features" + str(text_features))
        logit_scale         = self.logit_scale.exp()
        logits_per_image    = logit_scale * image_features @ text_features.t()
        # logits_per_image = image_features @ text_features.t()
        # print("logit_scale" + str(logit_scale))
        # print("logits_per_image" + str(logits_per_image))
        logits_per_text     = logits_per_image.t()
        # print("logits_per_image"+str(logits_per_image))
        # print("logits_per_text"+str(logits_per_text))
        return logits_per_image, logits_per_text

    