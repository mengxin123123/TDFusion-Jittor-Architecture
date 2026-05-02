import argparse
import os

import cv2
import numpy as np
import jittor as jt

from nets.ReFusion import ReFusion
from utils import ensure_dir, image_read


def is_grayscale(image):
    return image.ndim == 2 or (image.ndim == 3 and np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]))


def fuse_CrCb(CrCb1, CrCb2):
    assert len(CrCb1.shape) == 3 and CrCb1.shape[2] == 2, 'CrCb error'
    assert len(CrCb2.shape) == 3 and CrCb2.shape[2] == 2, 'CrCb error'
    Cf = (CrCb1 * np.abs(CrCb1 - 0.5) + CrCb2 * np.abs(CrCb2 - 0.5)) / (np.abs(CrCb1 - 0.5) + np.abs(CrCb2 - 0.5) + 1e-4)
    return Cf


def save_gray_and_rgb(fused_y, imagename, savepath, CrCb=None):
    gray_dir = os.path.join(savepath, 'Gray')
    rgb_dir = os.path.join(savepath, 'RGB')
    ensure_dir(gray_dir)
    if CrCb is not None:
        ensure_dir(rgb_dir)

    gray = np.squeeze(fused_y)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(gray_dir, f'{imagename}.png'), gray)

    if CrCb is not None:
        assert len(CrCb.shape) == 3 and CrCb.shape[2] == 2, 'CrCb error'
        ycrcb = np.concatenate((gray[..., np.newaxis], CrCb), axis=2)
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(rgb_dir, f'{imagename}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--dataset', default='MSRS')
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_samples', type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    model = ReFusion()
    if args.ckpt and os.path.exists(args.ckpt):
        state = jt.load(args.ckpt)
        if isinstance(state, dict) and 'fusion_model' in state:
            model.load_state_dict(state['fusion_model'])
        else:
            model.load_state_dict(state)

    if not args.data_root:
        raise ValueError('data_root is required for real test')

    ir_dir = os.path.join(args.data_root, args.split, 'ir')
    vi_dir = os.path.join(args.data_root, args.split, 'vi')
    names = sorted([n for n in os.listdir(ir_dir) if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    if args.max_samples > 0:
        names = names[:args.max_samples]

    with jt.no_grad():
        for name in names:
            img1 = image_read(os.path.join(ir_dir, name), 'RGB')
            img2 = image_read(os.path.join(vi_dir, name), 'RGB')

            img1_YCrCb = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            img2_YCrCb = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2YCrCb)

            if is_grayscale(img1):
                if is_grayscale(img2):
                    CrCb = None
                else:
                    CrCb = img2_YCrCb[:, :, 1:]
            else:
                if is_grayscale(img2):
                    CrCb = img1_YCrCb[:, :, 1:]
                else:
                    CrCb = fuse_CrCb(img1_YCrCb[:, :, 1:] / 255.0, img2_YCrCb[:, :, 1:] / 255.0)
                    CrCb = np.clip(CrCb * 255.0, 0, 255).astype(np.uint8)

            img1_y = img1_YCrCb[:, :, 0][np.newaxis, np.newaxis, ...] / 255.0
            img2_y = img2_YCrCb[:, :, 0][np.newaxis, np.newaxis, ...] / 255.0
            x = jt.array(np.concatenate((img1_y, img2_y), axis=1)).float32()
            y = model(x)
            y = (y - jt.min(y)) / (jt.max(y) - jt.min(y) + 1e-8)
            fused_image = np.squeeze((y * 255).numpy())
            fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)

            imagename = os.path.splitext(name)[0]
            save_gray_and_rgb(fused_image, imagename, os.path.join(args.output_dir, args.split), CrCb)

    print(f'processed {len(names)} samples')


if __name__ == '__main__':
    main()
