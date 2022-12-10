import copy
from typing import Callable, Optional

import torch
from pl_bolts.models.self_supervised.resnets import Bottleneck, ResNet
from timm.models.vision_transformer import Block, LayerScale, DropPath
from torch import nn

from utils.pos_embed import get_2d_sincos_pos_embed, interpolate_pos_encoding


class SimCLRResNet(nn.Module):

    def __init__(
            self,
            block: Callable = Bottleneck,
            layers: tuple = (3, 4, 6, 3),
            embed_dim: int = 2048,
            proj_dim: int = 128,
    ):
        super(SimCLRResNet, self).__init__()

        # Encoder
        self.encoder = ResNet(block, layers)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, img, position=True, shuffle=False):
        # img: [batch_size(*2), in_chans, height, weight]
        embed = self.encoder(img)[0]
        # embed: [batch_size*2, embed_dim]
        proj = self.proj_head(embed)
        # proj: [batch_size*2, proj_dim]
        return embed, proj


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    Modified and simplified from timm library for resolution adaptation
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AttentionWithWeightSharing(nn.Module):
    def __init__(self, qkv, proj, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = qkv.weight.size(-1)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = proj
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, weight_output=False):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads,
                          embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        attn_weight = attn_weight.softmax(dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        x = (attn_weight @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return (x, attn_weight) if weight_output else (x, None)


class MlpWithWeightSharing(nn.Module):

    def __init__(
            self,
            fc1: nn.Linear,
            fc2: nn.Linear,
            act_layer: Callable = nn.GELU,
            drop: float = 0.,
    ):
        super().__init__()

        self.fc1 = fc1
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = fc2
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class BlockWithWeightSharing(nn.Module):

    def __init__(
            self,
            qkv: nn.Linear,
            proj: nn.Linear,
            mlp_fc1: nn.Linear,
            mlp_fc2: nn.Linear,
            num_heads: int,
            drop: float = 0.,
            attn_drop: float = 0.,
            attention_weight: bool = False,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        self.attention_weight = attention_weight
        dim = qkv.weight.size(-1)
        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithWeightSharing(qkv, proj, num_heads=num_heads,
                                               attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MlpWithWeightSharing(mlp_fc1, mlp_fc2, act_layer, drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_, attn_weight = self.attn(self.norm1(x), self.attention_weight)
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        # Use dynamic return type here to make sequential module easier
        return (x, attn_weight) if attn_weight is not None else x


class SimCLRViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 384,
            depth: int = 12,
            num_heads: int = 6,
            mlp_ratio: int = 4,
            proj_dim: int = 128,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            weight_sharing: Optional[str] = None,
    ):
        """
        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of in channels
            embed_dim: embedding dimension
            depth: encoder number of Transformer blocks
            num_heads: encoder number of self-attention heads
            mlp_ratio: MLP dimension ratio (mlp_dim = embed_dim * mlp_ratio)
            proj_dim: projection head output dimension
            drop_rate: dropout rate
            attention_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            act_layer: activation layer
            norm_layer: normalization layer
            weight_sharing: ALBERT-like weight sharing,
                            choose from None, attn, ffn, or all
        """
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Following DeiT-3, exclude pos_embed from cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)

        self.blocks = self.build_blocks(
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            drop_rate,
            attention_drop_rate,
            drop_path_rate,
            act_layer,
            norm_layer,
            weight_sharing,
        )
        self.norm = norm_layer(embed_dim)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.init_weights()

    def build_blocks(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            drop_rate: float = 0.,
            attention_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            weight_sharing: Optional[str] = None,
    ) -> nn.Module:
        blocks = []
        qkv = nn.Linear(embed_dim, embed_dim * 3)
        proj = nn.Linear(embed_dim, embed_dim)
        mlp_fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        mlp_fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        for layer, dpr in enumerate(torch.linspace(0, drop_path_rate, depth)):
            block = BlockWithWeightSharing(
                qkv, proj, mlp_fc1, mlp_fc2, num_heads,
                drop=drop_rate, attn_drop=attention_drop_rate, drop_path=dpr.item(),
                act_layer=act_layer, norm_layer=norm_layer,
                attention_weight=(True if layer == depth - 1 else False),
            )
            if weight_sharing is None or weight_sharing == "attn":
                mlp_fc1 = copy.deepcopy(mlp_fc1)
                mlp_fc2 = copy.deepcopy(mlp_fc2)
            if weight_sharing is None or weight_sharing == "ffn":
                qkv = copy.deepcopy(qkv)
                proj = copy.deepcopy(proj)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.size(-1), int(self.patch_embed.num_patches ** .5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Init weights in convolutional layers like in MLPs
        patch_conv_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_conv_weight.view(patch_conv_weight.size(0), -1))

        nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_other_weights)

    def _init_other_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def rand_shuffle(x, pos_embed):
        batch_size, seq_len, embed_dim = x.size()
        # pos_embed: [1, seq_len, embed_dim]
        batch_pos_embed = pos_embed.expand(batch_size, -1, -1)
        # batch_pos_embed: [batch_size, seq_len, embed_dim]
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]
        expand_shuffled_indices = shuffled_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_shuffled_indices: [batch_size, seq_len, embed_dim]
        batch_shuffled_pos_embed = batch_pos_embed.gather(1, expand_shuffled_indices)
        # batch_shuffled_pos_embed: [batch_size, seq_len, embed_dim]
        return x + batch_shuffled_pos_embed

    def pre_encode(self, img, position=True, shuffle=False):
        x = self.patch_embed(img)
        if position:
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = interpolate_pos_encoding(x, self.pos_embed)
            else:
                pos_embed = self.pos_embed

            if shuffle:
                x = self.rand_shuffle(x, pos_embed)
            else:
                x += pos_embed
        # x: [batch_size, seq_len, embed_dim]

        return x

    def forward_encoder(self, x):
        # Concatenate [CLS] tokens w/o pos_embed
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x: [batch_size, 1 + seq_len, embed_dim]

        x, attn_weight = self.blocks(x)
        x = self.norm(x)
        # x: [batch_size, 1 + seq_len, embed_dim]

        return x, attn_weight

    def forward(self, img, position=True, shuffle=False):
        # img: [batch_size, in_chans, height, weight]
        patch_embed = self.pre_encode(img, position, shuffle)
        patch_embed, _ = self.forward_encoder(patch_embed)
        proj = self.proj_head(patch_embed[:, 0, :])
        # proj: [batch_size, proj_dim]

        return patch_embed, proj


class SimCLRMaskedViT(SimCLRViT):

    @staticmethod
    def mask_interval(x, mask_ratio, left_ratio, indices):
        """
        Leave $seq_len * (1 - mask_ratio)$ elements after center masking,
        with $seq_len * left_ratio$ elements tailing in the end.
        indices: [batch_size, seq_len]
        """

        batch_size, seq_len, embed_dim = x.size()
        visible_len = int(seq_len * (1 - mask_ratio))
        invisible_len = seq_len - visible_len
        tail_len = int(seq_len * left_ratio)
        mask_start_index = visible_len - tail_len
        mask_end_index = mask_start_index + invisible_len

        visible_indices_mask = torch.ones(seq_len, dtype=torch.bool)
        visible_indices_mask[mask_start_index:mask_end_index] = False
        # visible_indices_mask: [seq_len]
        visible_indices = indices[:, visible_indices_mask]
        # visible_indices: [batch_size, seq_len * mask_ratio]
        expand_visible_indices = visible_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        # expand_visible_indices: [batch_size, seq_len * mask_ratio, embed_dim]
        x_masked = x.gather(1, expand_visible_indices)
        # x_masked: [batch_size, seq_len * mask_ratio, embed_dim]

        return x_masked, expand_visible_indices

    def first_k_mask(self, x, mask_ratio, indices):
        """
        Leave first $k = seq_len * (1 - mask_ratio)$ elements after masking
        indices: [batch_size, seq_len]
        """

        return self.mask_interval(x, mask_ratio, 0., indices)

    def rand_mask(self, x, mask_ratio):
        batch_size, seq_len, embed_dim = x.size()
        noise = torch.rand(batch_size, seq_len, device=x.device)
        shuffled_indices = noise.argsort()
        # shuffled_indices: [batch_size, seq_len]

        return self.first_k_mask(x, mask_ratio, shuffled_indices)

    def pre_encode(self, img, position=True, shuffle=False, mask_ratio=0.):
        x = super().pre_encode(img, position, shuffle)
        # x: [batch_size, seq_len, embed_dim]

        visible_indices = None
        if mask_ratio > 0:
            x, visible_indices = self.rand_mask(x, mask_ratio)
            # x: [batch_size, seq_len * (1 - mask_ratio), embed_dim]

        return x, visible_indices

    def forward(self, x, position=True, shuffle=False, mask_ratio=0.):
        patch_embed, visible_idx = self.pre_encode(x, position, shuffle, mask_ratio)
        patch_embed, _ = self.forward_encoder(patch_embed)
        proj = self.proj_head(patch_embed[:, 0, :])
        # proj: [batch_size, proj_dim]

        return patch_embed, visible_idx, proj


class PosReconHead(nn.Module):
    def __init__(
            self,
            embed_dim: int = 384,
            depth: int = 3,
            num_heads: int = 6,
            mlp_ratio: int = 4,
            norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        if depth == 0:
            self.pos_decoder = nn.Linear(embed_dim, embed_dim)
        else:
            self.pos_decoder = nn.Sequential(*[Block(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            ) for _ in range(depth)
            ])

    def forward(self, x):
        return self.pos_decoder(x)
