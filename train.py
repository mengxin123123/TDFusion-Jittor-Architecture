import os
import argparse
import time
import random

import jittor as jt
import jittor.nn as nn

from data.dataloader import TrainsetSeg, TrainsetDet, yolo_dataset_collate
from nets.ReFusion import ReFusion, LPN
from nets.Segformer import SegFormer
from nets.yolo import YoloBody
from nets.yolo_training import yoloLoss
from utils import (
    seed_everything, Fusionloss_int, Fusionloss_grad, CE_Loss, inner_update,
    outer_update, ensure_dir, append_scalar_history, save_scalar_history, save_image
)


class StepLRScheduler:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = max(int(step_size), 1)
        self.gamma = gamma
        self.epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


def make_optimizer(params, lr, eps=None):
    return nn.Adam(params, lr=lr, eps=eps) if eps is not None else nn.Adam(params, lr=lr)


def _ensure_vis_dirs(root):
    for sub in ['model', 'logs', 'loss_curve', 'samples', 'results']:
        ensure_dir(os.path.join(root, sub))


def _tensor_to_image(x):
    arr = x.numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FMB')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', default='./exp')
    parser.add_argument('--resume', default='')
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--max_meta_step', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--vis_every', type=int, default=50)
    args = parser.parse_args()

    seed_everything(42)
    jt.flags.use_cuda = jt.has_cuda
    ensure_dir(args.output_dir)

    dataset = args.dataset.upper()
    if dataset == 'FMB':
        class_num, resize_h, resize_w = 14, 300, 400
    elif dataset == 'MSRS':
        class_num, resize_h, resize_w = 8, 240, 320
    elif dataset == 'M3FD':
        class_num, resize_h, resize_w = 6, 192, 256
    elif dataset == 'LLVIP':
        class_num, resize_h, resize_w = 1, 256, 320
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    switch = 1
    batch_size = args.batch_size // 2 if switch == 1 else args.batch_size

    fusion_model = ReFusion()
    task_model = SegFormer(class_num + 1) if dataset in ['FMB', 'MSRS'] else YoloBody(class_num, phi='n', pretrained=False)
    yolo_loss = yoloLoss(task_model)
    loss_model = LPN()

    optimizer1 = make_optimizer(fusion_model.parameters(), args.lr)
    scheduler1 = StepLRScheduler(optimizer1, args.step_size, args.gamma)
    optimizer2 = make_optimizer(task_model.parameters(), args.lr)
    scheduler2 = StepLRScheduler(optimizer2, args.step_size, args.gamma)
    optimizer3 = make_optimizer(loss_model.parameters(), args.lr, eps=1e-10)
    scheduler3 = StepLRScheduler(optimizer3, args.step_size, args.gamma)

    if dataset in ['FMB', 'MSRS']:
        taskloader = TrainsetSeg(dataset, args.data_root).set_attrs(batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        taskloader = TrainsetDet(dataset, args.data_root).set_attrs(batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=yolo_dataset_collate)

    exppath = os.path.join(args.output_dir, time.strftime('%m_%d_%H_%M', time.localtime()))
    ensure_dir(exppath)
    _ensure_vis_dirs(exppath)
    log_path = os.path.join(exppath, 'logs', 'train.log')
    loss_csv = os.path.join(exppath, 'loss_curve', 'loss_history.csv')

    if args.resume and os.path.exists(args.resume):
        state = jt.load(args.resume)
        if isinstance(state, dict):
            fusion_model.load_state_dict(state.get('fusion_model', state))
            if 'task_model' in state:
                task_model.load_state_dict(state['task_model'])
            if 'loss_model' in state:
                loss_model.load_state_dict(state['loss_model'])

    fusion_model.train()
    task_model.train()
    loss_model.train()

    history = []
    sample_saved = False

    for epoch in range(args.num_epochs):
        F_loss, F_loss_int, F_loss_grad, T_loss = [], [], [], []
        for i, (data_ir_ori, data_vis_ori, mask_t, index) in enumerate(taskloader):
            if i == args.max_meta_step * 2:
                break

            H, W = data_ir_ori.shape[2:]
            if H < 256 or W < 256:
                data_ir_crop, data_vis_crop = data_ir_ori, data_vis_ori
            else:
                h = int((H - 256) * random.random())
                w = int((W - 256) * random.random())
                data_ir_crop = data_ir_ori[:, :, h:h + 256, w:w + 256]
                data_vis_crop = data_vis_ori[:, :, h:h + 256, w:w + 256]

            data_ir_resize = nn.interpolate(data_ir_ori, size=(resize_h, resize_w), mode='bilinear')
            data_vis_resize = nn.interpolate(data_vis_ori, size=(resize_h, resize_w), mode='bilinear')

            if switch:
                data_ir_crop_ori, data_vis_crop_ori = data_ir_crop, data_vis_crop
                data_ir_resize_ori, data_vis_resize_ori = data_ir_resize, data_vis_resize
                data_ir_crop = jt.concat((data_ir_crop_ori, data_vis_crop_ori), 0)
                data_vis_crop = jt.concat((data_vis_crop_ori, data_ir_crop_ori), 0)
                data_ir_resize = jt.concat((data_ir_resize_ori, data_vis_resize_ori), 0)
                data_vis_resize = jt.concat((data_vis_resize_ori, data_ir_resize_ori), 0)
                mask_t = jt.concat((mask_t, mask_t), 0)
                index = jt.concat((index, index), 0)
                if dataset in ['M3FD', 'LLVIP']:
                    mask_t[-mask_t.shape[0] // 2:, 0] = mask_t[-mask_t.shape[0] // 2:, 0] + batch_size

            if i % 2 == 0:
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                data_fuse = fusion_model(jt.concat((data_ir_crop, data_vis_crop), 1))
                w = loss_model(jt.concat((data_ir_crop, data_vis_crop), 1))
                loss_f_int = Fusionloss_int(data_fuse, data_ir_crop, data_vis_crop, w[:, 0:1], w[:, 1:2])
                loss_f_grad = Fusionloss_grad(data_fuse, data_ir_crop, data_vis_crop)
                fusion_loss = loss_f_int + loss_f_grad
                optimizer1.backward(fusion_loss)
                optimizer1.step()
                inner_update(fusion_model, optimizer1, args.lr)

                data_fuse = fusion_model(jt.concat((data_ir_resize, data_vis_resize), 1)).detach()
                data_task = task_model(data_fuse)
                if dataset in ['FMB', 'MSRS']:
                    mask_ce = nn.interpolate(mask_t.unsqueeze(1).float32(), size=data_task.shape[2:], mode='nearest').squeeze(1).int32()
                    task_loss = CE_Loss(data_task, mask_ce)
                else:
                    task_loss = yolo_loss(data_task, mask_t)
                optimizer2.backward(task_loss)
                optimizer2.step()

                F_loss.append(float(fusion_loss.item()))
                F_loss_int.append(float(loss_f_int.item()))
                F_loss_grad.append(float(loss_f_grad.item()))
                T_loss.append(float(task_loss.item()))

                if not sample_saved:
                    save_image(_tensor_to_image(data_ir_ori), os.path.join(exppath, 'samples', 'ir.png'))
                    save_image(_tensor_to_image(data_vis_ori), os.path.join(exppath, 'samples', 'vi.png'))
                    save_image(_tensor_to_image(data_fuse), os.path.join(exppath, 'samples', 'fused.png'))
                    sample_saved = True
            else:
                optimizer3.zero_grad()
                data_fuse = fusion_model(jt.concat((data_ir_resize, data_vis_resize), 1), meta=True)
                data_task = task_model(data_fuse)
                if dataset in ['FMB', 'MSRS']:
                    mask_ce = nn.interpolate(mask_t.unsqueeze(1).float32(), size=data_task.shape[2:], mode='nearest').squeeze(1).int32()
                    task_loss = CE_Loss(data_task, mask_ce)
                else:
                    task_loss = yolo_loss(data_task, mask_t)
                optimizer3.backward(task_loss)
                optimizer3.step()
                outer_update(loss_model, optimizer3, args.lr)

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        ckpt = {
            'fusion_model': fusion_model.state_dict(),
            'task_model': task_model.state_dict(),
            'loss_model': loss_model.state_dict(),
            'epoch': epoch + 1,
        }
        jt.save(ckpt, os.path.join(exppath, 'model', f'ckpt_{epoch + 1}.pkl'))

        epoch_row = (
            epoch + 1,
            sum(F_loss) / max(len(F_loss), 1),
            sum(F_loss_int) / max(len(F_loss_int), 1),
            sum(F_loss_grad) / max(len(F_loss_grad), 1),
            sum(T_loss) / max(len(T_loss), 1),
        )
        append_scalar_history(history, epoch_row)
        save_scalar_history(history, loss_csv)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"Epoch {epoch + 1}/{args.num_epochs} "
                f"FusionLoss={epoch_row[1]:.6f} "
                f"F_int={epoch_row[2]:.6f} "
                f"F_grad={epoch_row[3]:.6f} "
                f"TaskLoss={epoch_row[4]:.6f}\n"
            )
        print(f'Epoch {epoch + 1}/{args.num_epochs} done')


if __name__ == '__main__':
    main()
