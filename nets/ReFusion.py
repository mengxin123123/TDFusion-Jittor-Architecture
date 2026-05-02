import numbers
import jittor as jt
import jittor.nn as nn


def to_3d(x):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x, h, w):
    b, n, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


def _norm(x, dim=-1, eps=1e-5):
    mean = x.mean(dim=dim, keepdims=True)
    var = ((x - mean) * (x - mean)).mean(dim=dim, keepdims=True)
    return (x - mean) / jt.sqrt(var + eps)


class MetaConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv(*args, **kwargs)
        self.weight_meta = self.conv.weight
        self.bias_meta = self.conv.bias

    def named_leaves(self):
        return [('weight', self.conv.weight), ('bias', self.conv.bias)]

    def execute(self, x, meta=False):
        if meta:
            return nn.conv2d(x, self.weight_meta, self.bias_meta, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        return self.conv(x)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def execute(self, x, meta=False):
        return _norm(x) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def execute(self, x, meta=False):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x), meta=meta), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = MetaConv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = MetaConv2d(hidden * 2, hidden * 2, 3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = MetaConv2d(hidden, dim, 1, bias=bias)

    def execute(self, x, meta=False):
        x = self.project_in(x, meta=meta)
        x = self.dwconv(x, meta=meta)
        c = x.shape[1] // 2
        x1, x2 = x[:, :c, :, :], x[:, c:, :, :]
        x = jt.nn.gelu(x1) * x2
        return self.project_out(x, meta=meta)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = jt.ones((num_heads, 1, 1))
        self.qkv = MetaConv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = MetaConv2d(dim * 3, dim * 3, 3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = MetaConv2d(dim, dim, 1, bias=bias)

    def execute(self, x, meta=False):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x, meta=meta), meta=meta)
        q, k, v = qkv[:, :c, :, :], qkv[:, c:2*c, :, :], qkv[:, 2*c:, :, :]
        head = self.num_heads
        q = q.reshape(b, head, c // head, h * w)
        k = k.reshape(b, head, c // head, h * w)
        v = v.reshape(b, head, c // head, h * w)
        q = q / (q.norm(dim=-1, keepdims=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdims=True) + 1e-6)
        attn = jt.matmul(q, k.transpose(2, 3)) * self.temperature
        attn = jt.nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out, meta=meta)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def execute(self, x, meta=False):
        x = x + self.attn(self.norm1(x, meta=meta), meta=meta)
        x = x + self.ffn(self.norm2(x, meta=meta), meta=meta)
        return x


class MetaPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = jt.array([init] * num_parameters).float32()

    def named_leaves(self):
        return [('weight', self.weight)]

    def execute(self, x, meta=False):
        w = self.weight.reshape(1, -1, 1, 1)
        return jt.maximum(0, x) + w * jt.minimum(0, x)


class AFM(nn.Module):
    def __init__(self, dim=16, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = jt.ones((num_heads, 1, 1))
        self.temperature2 = jt.ones((num_heads, 1, 1))
        self.qkv1 = MetaConv2d(dim, dim * 3, 1, bias=bias)
        self.qkv1_dwconv = MetaConv2d(dim * 3, dim * 3, 3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv2 = MetaConv2d(dim, dim * 3, 1, bias=bias)
        self.qkv2_dwconv = MetaConv2d(dim * 3, dim * 3, 3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_mid1 = MetaConv2d(dim, dim, 1, bias=bias)
        self.project_mid2 = MetaConv2d(dim, dim, 1, bias=bias)
        self.project_E = MetaConv2d(dim * 2, dim * 8, 1, bias=bias)
        self.project_E_d = MetaConv2d(dim * 8, dim * 8, 3, stride=1, padding=1, groups=dim * 8, bias=bias)
        self.project_E1 = MetaConv2d(dim, dim * 4, 1, bias=bias)
        self.project_E1_d = MetaConv2d(dim * 4, dim * 4, 3, stride=1, padding=1, groups=dim * 4, bias=bias)
        self.project_E2 = MetaConv2d(dim, dim * 4, 1, bias=bias)
        self.project_E2_d = MetaConv2d(dim * 4, dim * 4, 3, stride=1, padding=1, groups=dim * 4, bias=bias)
        self.project_out1 = MetaConv2d(dim * 4, dim * 2, 1, bias=bias)
        self.project_out2 = MetaConv2d(dim * 4, dim * 2, 1, bias=bias)

    def execute(self, x1, x2, meta=False):
        b, c, h, w = x1.shape
        qkv1 = self.qkv1_dwconv(self.qkv1(x1, meta=meta), meta=meta)
        q1, k1, v1 = qkv1[:, :c, :, :], qkv1[:, c:2*c, :, :], qkv1[:, 2*c:, :, :]
        qkv2 = self.qkv2_dwconv(self.qkv2(x2, meta=meta), meta=meta)
        q2, k2, v2 = qkv2[:, :c, :, :], qkv2[:, c:2*c, :, :], qkv2[:, 2*c:, :, :]
        head = self.num_heads
        q1 = q1.reshape(b, head, c // head, h * w)
        k1 = k1.reshape(b, head, c // head, h * w)
        v1 = v1.reshape(b, head, c // head, h * w)
        q2 = q2.reshape(b, head, c // head, h * w)
        k2 = k2.reshape(b, head, c // head, h * w)
        v2 = v2.reshape(b, head, c // head, h * w)
        q1 = q1 / (q1.norm(dim=-1, keepdims=True) + 1e-6)
        k1 = k1 / (k1.norm(dim=-1, keepdims=True) + 1e-6)
        q2 = q2 / (q2.norm(dim=-1, keepdims=True) + 1e-6)
        k2 = k2 / (k2.norm(dim=-1, keepdims=True) + 1e-6)
        attn1 = jt.matmul(q2, k1.transpose(2, 3)) * self.temperature1
        attn2 = jt.matmul(q1, k2.transpose(2, 3)) * self.temperature2
        out1 = jt.matmul(jt.nn.softmax(attn1, dim=-1), v1).reshape(b, c, h, w)
        out2 = jt.matmul(jt.nn.softmax(attn2, dim=-1), v2).reshape(b, c, h, w)
        x1 = x1 + self.project_mid1(out1, meta=meta)
        x2 = x2 + self.project_mid2(out2, meta=meta)
        out = jt.concat([x1, x2], dim=1)
        eg = self.project_E_d(self.project_E(out, meta=meta), meta=meta)
        c2 = eg.shape[1] // 2
        g1, g2 = eg[:, :c2, :, :], eg[:, c2:, :, :]
        x1 = jt.nn.gelu(g1) * self.project_E1_d(self.project_E1(x1, meta=meta), meta=meta)
        x2 = jt.nn.gelu(g2) * self.project_E2_d(self.project_E2(x2, meta=meta), meta=meta)
        return out + self.project_out1(x1, meta=meta) + self.project_out2(x2, meta=meta)


class ReFusion(nn.Module):
    def __init__(self, dim=16, num_blocks=2, heads=8, ffn_expansion_factor=2, bias=False):
        super().__init__()
        self.project_in_1 = MetaConv2d(1, dim, 3, stride=1, padding=1, bias=bias)
        self.project_in_2 = MetaConv2d(1, dim, 3, stride=1, padding=1, bias=bias)
        self.encoder1 = nn.ModuleList([TransformerBlock(dim, heads, ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.encoder2 = nn.ModuleList([TransformerBlock(dim, heads, ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.FM = AFM(dim=dim, num_heads=heads, bias=bias)
        self.decoder = nn.ModuleList([TransformerBlock(dim * 2, heads, ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.project_out_0 = MetaConv2d(dim * 2, dim, 3, stride=1, padding=1, bias=bias)
        self.act = nn.LeakyReLU()
        self.project_out_1 = MetaConv2d(dim, 1, 3, stride=1, padding=1, bias=bias)

    def execute(self, x, meta=False):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = self.project_in_1(x1, meta=meta)
        x2 = self.project_in_2(x2, meta=meta)
        for layer in self.encoder1:
            x1 = layer(x1, meta=meta)
        for layer in self.encoder2:
            x2 = layer(x2, meta=meta)
        out = self.FM(x1, x2, meta=meta)
        for layer in self.decoder:
            out = layer(out, meta=meta)
        out = self.project_out_0(out, meta=meta)
        out = self.act(out)
        out = self.project_out_1(out, meta=meta)
        return jt.sigmoid(out)


class LPN(nn.Module):
    def __init__(self, dim=16, num_blocks=2, heads=[8, 8, 8], ffn_expansion_factor=2, bias=False):
        super().__init__()
        self.patch_embed1 = MetaConv2d(1, dim, 3, stride=1, padding=1, bias=bias)
        self.patch_embed2 = MetaConv2d(1, dim, 3, stride=1, padding=1, bias=bias)
        self.encoder1 = nn.ModuleList([TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.encoder2 = nn.ModuleList([TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.encoder3 = nn.ModuleList([TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks)])
        self.out1 = MetaConv2d(dim * 2, dim, 3, stride=1, padding=1, bias=bias)
        self.out2 = MetaPReLU()
        self.out3 = MetaConv2d(dim, 2, 3, stride=1, padding=1, bias=bias)

    def execute(self, x, meta=False):
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:, :, :]
        x1 = self.patch_embed1(x1, meta=meta)
        for subm in self.encoder1:
            x1 = subm(x1, meta=meta)
        x2 = self.patch_embed2(x2, meta=meta)
        for subm in self.encoder2:
            x2 = subm(x2, meta=meta)
        out = jt.concat((x1, x2), 1)
        for subm in self.encoder3:
            out = subm(out, meta=meta)
        out = self.out1(out, meta=meta)
        out = self.out2(out)
        out = self.out3(out, meta=meta)
        return jt.nn.softmax(out, dim=1)


def count_parameter(net):
    total = 0
    for p in net.parameters():
        total += p.numel()
    print('Number of parameter: %.3fM' % (total / 1e6))
    return total
