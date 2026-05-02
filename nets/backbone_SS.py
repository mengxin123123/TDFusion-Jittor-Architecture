import jittor as jt
import jittor.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv(in_features, hidden_features, 1)
        self.dwconv = nn.Conv(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Conv(dim, dim, 1)
        self.kv = nn.Conv(dim, dim * 2, 1)
        self.proj = nn.Conv(dim, dim, 1)
        self.sr_ratio = sr_ratio
        self.sr = nn.Conv(dim, dim, sr_ratio, stride=sr_ratio, groups=dim) if sr_ratio > 1 else None
        self.norm = nn.BatchNorm(dim) if sr_ratio > 1 else None

    def execute(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, N).transpose(2, 3)
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = self.norm(x_)
            kv = self.kv(x_)
            _, _, H_, W_ = x_.shape
            Nk = H_ * W_
        else:
            kv = self.kv(x)
            Nk = N
        kv = kv.reshape(B, 2, self.num_heads, C // self.num_heads, Nk)
        k = kv[:, 0].transpose(2, 3)
        v = kv[:, 1].transpose(2, 3)
        q2 = q.reshape(B * self.num_heads, N, C // self.num_heads)
        k2 = k.reshape(B * self.num_heads, C // self.num_heads, Nk)
        v2 = v.reshape(B * self.num_heads, C // self.num_heads, Nk)
        attn = jt.matmul(q2, k2) * self.scale
        attn = jt.nn.softmax(attn, dim=-1)
        x = jt.matmul(attn, v2.transpose(1, 2)).reshape(B, self.num_heads, N, C // self.num_heads).transpose(2, 3).reshape(B, C, H, W)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.BatchNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.BatchNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, stride=4):
        super().__init__()
        self.proj = nn.Conv(in_chans, embed_dim, 7 if stride == 4 else 3, stride=stride, padding=3 if stride == 4 else 1)
        self.norm = nn.BatchNorm(embed_dim)

    def execute(self, x):
        return self.norm(self.proj(x))


class MixVisionTransformer(nn.Module):
    def __init__(self, embed_dims, num_heads, depths, sr_ratios):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(1, embed_dims[0], 4)
        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], sr_ratios[0]) for _ in range(depths[0])])
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], 2)
        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], sr_ratios[1]) for _ in range(depths[1])])
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], 2)
        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], sr_ratios[2]) for _ in range(depths[2])])
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], 2)
        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], sr_ratios[3]) for _ in range(depths[3])])

    def execute(self, x):
        x1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1)
        x2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2)
        x3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3)
        x4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4)
        return x1, x2, x3, x4


def mit_b0(pretrained=False):
    return MixVisionTransformer([32, 64, 160, 256], [1, 2, 5, 8], [2, 2, 2, 2], [8, 4, 2, 1])


def mit_b1(pretrained=False):
    return MixVisionTransformer([64, 128, 320, 512], [1, 2, 5, 8], [2, 2, 2, 2], [8, 4, 2, 1])


def mit_b2(pretrained=False):
    return mit_b1(pretrained)


def mit_b3(pretrained=False):
    return mit_b1(pretrained)


def mit_b4(pretrained=False):
    return mit_b1(pretrained)


def mit_b5(pretrained=False):
    return mit_b1(pretrained)
