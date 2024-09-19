import torch

from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WISE_BRCA(nn.Module):
    def __init__(self, input_dim=768, target_len=60, depth=12, embed_dim=768, num_heads=12, mlp_ratio=4., drop_rate=0.4, attn_drop_rate=0.4,
                  drop_path_rate=0.4):

        super(WISE_BRCA, self).__init__()

        self.cluster_num = 30
        self.proj_224 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.BatchNorm1d(90), nn.ReLU())
        self.proj_512 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.BatchNorm1d(target_len),  nn.ReLU())
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(2*embed_dim, 1))
        self.cls_224 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, 1, embed_dim), 0., 0.2))
        self.cls_512 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, 1, embed_dim), 0., 0.2))

        self.cluster_embedding = nn.Embedding(self.cluster_num+1, embed_dim, padding_idx=30)
        self.depth = depth
        self.attention = nn.Sequential(nn.Linear(embed_dim, int(embed_dim/2)),
                                       nn.Tanh(),
                                       nn.Linear(int(embed_dim/2), 1))

        def get_attention_block(depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            return nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i])
                for i in range(depth)])

        def forward_cross_attention(x1_f, x2_f, i):

            x1_f = self.l_branch[i](x1_f)
            x2_f = self.h_branch[i](x2_f)
            cls_x1 = x1_f[:, -1, :]
            cls_x2 = x2_f[:, -1, :]
            x1_f_ = torch.cat([x1_f[:, 0:-1, :], cls_x2.unsqueeze(1)], dim=1)
            x2_f_ = torch.cat([x2_f[:, 0:-1, :], cls_x1.unsqueeze(1)], dim=1)

            return x1_f_, x2_f_

        self.get_attention_block = get_attention_block
        self.forward_cross_attention = forward_cross_attention
        self.l_branch = nn.Sequential(*[self.get_attention_block(depth=1) for i in range(0, depth)])
        self.h_branch = nn.Sequential(*[self.get_attention_block(depth=1) for i in range(0, depth)])

        for layer in self.head:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.proj_224:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)
        for layer in self.proj_512:
            if type(layer) == torch.nn.modules.linear.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x1, x2):

        x1_f = self.proj_224(x1)
        x2_f = self.proj_512(x2)
        x1_f = torch.cat([x1_f, self.cls_224.repeat(x1_f.shape[0], 1, 1)], dim=1)
        x2_f = torch.cat([x2_f, self.cls_512.repeat(x2_f.shape[0], 1, 1)], dim=1)
        for i in range(0, int(self.depth)):
            x1_f, x2_f = self.forward_cross_attention(x1_f, x2_f, i)
        cls_dec = torch.cat([x1_f[:, -1, :], x2_f[:, -1, :]], dim=1)


        return cls_dec
















