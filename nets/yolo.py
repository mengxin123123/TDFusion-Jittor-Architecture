import jittor as jt
import jittor.nn as nn
from .backbone_OD import Backbone, C2f, Conv


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv(c1, 1, 1, bias=False)
        self.c1 = c1

    def execute(self, x):
        b, c, a = x.shape
        x = x.reshape(b, 4, self.c1, a).transpose(2, 1)
        x = jt.nn.softmax(x, dim=1)
        return self.conv(x).reshape(b, 4, a)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi, pretrained=False):
        super().__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00}
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.nl = len(ch)
        self.reg_max = 16
        self.no = num_classes + self.reg_max * 4
        self.num_classes = num_classes
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], num_classes)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv(c3, num_classes, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def execute(self, x):
        feat1, feat2, feat3 = self.backbone.forward(x)
        p5 = self.upsample(feat3)
        p4 = self.conv3_for_upsample1(jt.concat([p5, feat2], 1))
        p4_up = self.upsample(p4)
        p3 = self.conv3_for_upsample2(jt.concat([p4_up, feat1], 1))
        p3_down = self.down_sample1(p3)
        p4 = self.conv3_for_downsample1(jt.concat([p3_down, p4], 1))
        p4_down = self.down_sample2(p4)
        p5 = self.conv3_for_downsample2(jt.concat([p4_down, feat3], 1))
        x = [p3, p4, p5]
        out = []
        for i in range(self.nl):
            out.append(jt.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        box, cls = jt.concat([xi.reshape(xi.shape[0], self.no, -1) for xi in out], 2).split((self.reg_max * 4, self.num_classes), 1)
        dbox = self.dfl(box)
        return dbox, cls, out, None, None
