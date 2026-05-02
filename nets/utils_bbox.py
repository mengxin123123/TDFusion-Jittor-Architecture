import jittor as jt


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        y, x = jt.meshgrid(jt.arange(h), jt.arange(w))
        points = jt.stack([x, y], dim=-1).reshape(-1, 2).float32() + grid_cell_offset
        anchor_points.append(points)
        stride_tensor.append(jt.ones((points.shape[0], 1)) * stride)
    return jt.concat(anchor_points, dim=0), jt.concat(stride_tensor, dim=0)
