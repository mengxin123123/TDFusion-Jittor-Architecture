import os
import numpy as np
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset

from utils import image_read


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(name)
    return files


class TrainsetSeg(Dataset):
    def __init__(self, dataname, data_root):
        super().__init__()
        self.dataname = dataname.upper()
        self.data_root = data_root
        if self.dataname == 'FMB':
            self.ir_path = os.path.join(data_root, 'train', 'ir')
            self.vi_path = os.path.join(data_root, 'train', 'vi')
            self.mask_path = os.path.join(data_root, 'train', 'label')
        elif self.dataname == 'MSRS':
            self.ir_path = os.path.join(data_root, 'train', 'ir')
            self.vi_path = os.path.join(data_root, 'train', 'vi')
            self.mask_path = os.path.join(data_root, 'train', 'label')
        else:
            raise ValueError(f'Unknown segmentation dataset: {dataname}')

        self.file_name_list = _list_image_files(self.ir_path)
        if not self.file_name_list:
            self.file_name_list = _list_image_files(self.vi_path)
        self.set_attrs(total_len=len(self.file_name_list), shuffle=True)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        name = self.file_name_list[index]
        ir_path = os.path.join(self.ir_path, name)
        vi_path = os.path.join(self.vi_path, name)
        mask_path = os.path.join(self.mask_path, name)
        ir = image_read(ir_path, 'GRAY')[np.newaxis, ...] / 255.0
        vi = image_read(vi_path, 'GRAY')[np.newaxis, ...] / 255.0
        mask = image_read(mask_path, 'GRAY').astype(np.int64)
        return jt.array(ir).float32(), jt.array(vi).float32(), jt.array(mask).int64(), jt.array([index]).int32()


class TrainsetDet(Dataset):
    def __init__(self, data_name, data_root):
        super().__init__()
        self.data_name = data_name.upper()
        self.data_root = data_root
        if self.data_name == 'M3FD':
            list_path = os.path.join(data_root, 'M3FD_train.txt')
            self.ir_path = os.path.join(data_root, 'M3FD', 'ir')
            self.vi_path = os.path.join(data_root, 'M3FD', 'vi')
            self.h, self.w = 768, 1024
        elif self.data_name == 'LLVIP':
            list_path = os.path.join(data_root, 'LLVIP_train.txt')
            self.ir_path = os.path.join(data_root, 'LLVIP', 'ir')
            self.vi_path = os.path.join(data_root, 'LLVIP', 'vi')
            self.h, self.w = 1024, 1280
        else:
            raise ValueError(f'Unknown detection dataset: {data_name}')

        with open(list_path, encoding='utf-8') as f:
            self.annotation_lines = f.readlines()
        self.set_attrs(total_len=len(self.annotation_lines), shuffle=True)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        name = line[0]
        ir = Image.open(os.path.join(self.ir_path, name)).convert('L')
        vi = Image.open(os.path.join(self.vi_path, name)).convert('L')
        if ir.size != (self.w, self.h):
            ir = ir.resize((self.w, self.h), Image.BICUBIC)
            vi = vi.resize((self.w, self.h), Image.BICUBIC)

        ir = np.array(ir, dtype=np.float32)[None, ...] / 255.0
        vi = np.array(vi, dtype=np.float32)[None, ...] / 255.0
        box = np.array([list(map(int, b.split(','))) for b in line[1:]], dtype=np.float32)
        np.random.shuffle(box)
        labels_out = np.zeros((len(box), 6), dtype=np.float32)
        if len(box):
            box[:, [0, 2]] = box[:, [0, 2]] / self.h
            box[:, [1, 3]] = box[:, [1, 3]] / self.w
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
        return jt.array(ir).float32(), jt.array(vi).float32(), jt.array(labels_out).float32(), jt.array([index]).int32()


def yolo_dataset_collate(batch):
    ir_images, vi_images, bboxes, index_list = [], [], [], []
    for i, (ir, vi, box, index) in enumerate(batch):
        ir_images.append(ir)
        vi_images.append(vi)
        box_np = box.numpy() if isinstance(box, jt.Var) else box
        box_np[:, 0] = i
        bboxes.append(box_np)
        index_list.append(index.numpy() if isinstance(index, jt.Var) else index)
    ir_images = jt.array(np.array([x.numpy() for x in ir_images], dtype=np.float32))
    vi_images = jt.array(np.array([x.numpy() for x in vi_images], dtype=np.float32))
    bboxes = jt.array(np.concatenate(bboxes, 0).astype(np.float32)) if len(bboxes) else jt.array(np.zeros((0, 6), dtype=np.float32))
    index_list = jt.array(np.concatenate(index_list, 0).astype(np.int32))
    return ir_images, vi_images, bboxes, index_list
