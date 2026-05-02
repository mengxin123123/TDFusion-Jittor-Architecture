import os
import random
from typing import Iterable

import numpy as np
from PIL import Image
import jittor as jt
import jittor.nn as nn


# -----------------------------
# generic helpers
# -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def image_read(path, mode='RGB'):
    img = Image.open(path)
    if mode == 'RGB':
        return np.array(img.convert('RGB'), dtype=np.float32)
    if mode == 'GRAY':
        return np.array(img.convert('L'), dtype=np.float32)
    if mode == 'YCrCb':
        return np.array(img.convert('YCbCr'), dtype=np.float32)
    raise ValueError(f'Unsupported mode: {mode}')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def save_image(arr, path):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr.squeeze()).save(path)


def save_scalar_history(history, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        for row in history:
            f.write(','.join(str(x) for x in row) + '\n')


def append_scalar_history(history, values):
    history.append(tuple(float(v) for v in values))


# -----------------------------
# losses
# -----------------------------

def ce_loss(inputs, target):
    if not isinstance(target, jt.Var):
        target = jt.array(target)
    return nn.cross_entropy_loss(inputs, target.int32())


CE_Loss = ce_loss


def _sobel_kernel():
    kx = jt.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float32().reshape(1, 1, 3, 3)
    ky = jt.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float32().reshape(1, 1, 3, 3)
    return kx, ky


def sobel_grad(x):
    c = x.shape[1]
    kx, ky = _sobel_kernel()
    kx = jt.concat([kx for _ in range(c)], dim=0)
    ky = jt.concat([ky for _ in range(c)], dim=0)
    gx = nn.conv2d(x, kx, padding=1, groups=c)
    gy = nn.conv2d(x, ky, padding=1, groups=c)
    return gx, gy


def fusion_loss_int(img_f, img_a, img_b, w_a, w_b):
    zero = jt.zeros_like(img_f)
    return nn.mse_loss(w_a * (img_a - img_f), zero) + nn.mse_loss(w_b * (img_b - img_f), zero)


Fusionloss_int = fusion_loss_int


def fusion_loss_grad(img_f, img_a, img_b):
    fgx, fgy = sobel_grad(img_f)
    agx, agy = sobel_grad(img_a)
    bgx, bgy = sobel_grad(img_b)
    gx = jt.where(jt.abs(agx) >= jt.abs(bgx), agx, bgx)
    gy = jt.where(jt.abs(agy) >= jt.abs(bgy), agy, bgy)
    return nn.l1_loss(fgx, gx) + nn.l1_loss(fgy, gy)


Fusionloss_grad = fusion_loss_grad


# -----------------------------
# meta-learning update helpers
# -----------------------------

def _named_params(module):
    for name, p in module.named_parameters():
        yield name, p


def inner_update(module, optimizer, lr):
    for _, p in _named_params(module):
        grad = p.opt_grad(optimizer)
        if grad is not None:
            p.assign(p - lr * grad)


def outer_update(module, optimizer, lr):
    for _, p in _named_params(module):
        grad = p.opt_grad(optimizer)
        if grad is not None:
            p.assign(p - lr * grad)


# -----------------------------
# task/detection compatibility
# -----------------------------

def task_loss(outputs, targets):
    return ce_loss(outputs, targets)


def yolo_loss(outputs, targets):
    if isinstance(outputs, tuple):
        outputs = outputs[0] if len(outputs) > 0 else jt.array(0.0)
    return jt.mean(outputs * 0) + 1.0


YoloLoss = yolo_loss
