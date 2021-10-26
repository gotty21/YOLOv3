import os
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset


# input     >> img(cv2で読み込んだndarray), imasize(リサイズ後のサイズ)
# output    >> img(ndarray), info_img([元のh, 元のw, リサイズ後画像が存在するh, リサイズ後画像が存在するw, 127で埋めた横方向の片側量, 127で埋めた縦方向の片側量])
def pad_resize(img, img_size: int):
    h, w, _ = img.shape
    img = img[:, :, ::-1]       # BGR >> RGB

    # 比率
    r = w / h

    ## 縦長の場合
    if r < 1:
        new_h = img_size        # new_h = imgsize
        new_w = new_h * r       # new_w = new_h * w / h << h:w = new_h : new_w
    ## 横長の場合
    else:
        new_w = img_size
        new_h = new_w / r

    new_w, new_h = int(new_w), int(new_h)


    # 拡大する量(片側)
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2


    # resize
    img = cv2.resize(img, (new_w, new_h))                           # 一旦new_wとnew_hでリサイズ

    sized = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
    sized[dy:dy+new_h, dx:dx+new_w, :] = img                        # 余白を127で埋めて入れる

    img_info = (h, w, new_h, new_w, dx, dy)
    return sized, img_info


# input     >> labls: [x(左上), y(左上), h, w]
# output    >> labes: [xcenter, ycenter, h, w] (pad_resizeに合わせて変換)
def label2yolobox(labels, info_img, maxsize):
    h, w, nh, nw, dx, dy = info_img

    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h

    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize

    return labels   


class COCODataset(Dataset):
    def __init__(self, img_size=416, min_size=1, max_labels=10):
        
        self.data_dir = '/content/drive/MyDrive/project/YOLOv3/code/COCO/'   # COCOまでのpath
        self.json_file = 'instances_val2017.json'
        self.img_dir = 'val2017'
        
        self.coco = COCO(self.data_dir + 'annotations/' + self.json_file)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

        self.img_size = img_size
        self.min_size = min_size
        self.max_labels = max_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]

        # load image
        img_file = os.path.join(self.data_dir, self.img_dir, '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        img, info_img = pad_resize(img, self.img_size)

        img = np.transpose(img / 255., (2, 0, 1))       # px値0-1化 + [ch, h, w]に並び替え


        # load labels
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        labels = []

        for anno in annotations:
            # anno: [x(左上), y(左上), h, w]

            # h, w が min_sizeを超えている場合
            if (anno['bbox'][2] > self.min_size) and (anno['bbox'][3] > self.min_size):
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))

        # ラベルが見つかったら
        if len(labels) > 0:
            labels = np.stack(labels)
            labels = label2yolobox(labels, info_img, self.img_size)
            padded_labels[range(len(labels))[:self.max_labels]] = labels[:self.max_labels]
        
        padded_labels = torch.from_numpy(padded_labels)


        return img, padded_labels, info_img, id_



if __name__ == '__main__':
    dataset = COCODataset()
    
    index = 1
    img, labels, _, _ = dataset[index]
    print(img.shape)
    print(labels.size())

