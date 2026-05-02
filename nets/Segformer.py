import jittor as jt
import jittor.nn as nn
from .backbone_SS import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv(input_dim, embed_dim, 1)

    def execute(self, x):
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv(c1, c2, k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm(c2)
        self.act = nn.ReLU() if act is True else act

    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1)
        self.linear_pred = nn.Conv(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout(dropout_ratio)

    def execute(self, inputs):
        c1, c2, c3, c4 = inputs
        c1 = self.linear_c1(c1)
        c2 = nn.interpolate(self.linear_c2(c2), size=c1.shape[2:], mode='bilinear')
        c3 = nn.interpolate(self.linear_c3(c3), size=c1.shape[2:], mode='bilinear')
        c4 = nn.interpolate(self.linear_c4(c4), size=c1.shape[2:], mode='bilinear')
        _c = self.linear_fuse(jt.concat([c4, c3, c2, c1], dim=1))
        return self.linear_pred(self.dropout(_c))


class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False):
        super().__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2, 'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {'b0': 256, 'b1': 256, 'b2': 768, 'b3': 768, 'b4': 768, 'b5': 768}[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def execute(self, inputs):
        H, W = inputs.shape[2], inputs.shape[3]
        x = self.backbone.execute(inputs)
        x = self.decode_head(x)
        return nn.interpolate(x, size=(H, W), mode='bilinear')
