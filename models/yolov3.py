import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from dataset.cocodataset import COCODataset
from utils.utils import bboxes_iou


# Conv -> batch_norm -> leakyReLU
def add_conv(in_ch, out_ch, ksize, stride):
    stage = nn.Sequential()

    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksize, stride=stride, padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))

    return stage

# Residual block
class resblock(nn.Module):
    
    def __init__(self, in_ch, nblocks=1, shortcut=True):
        super().__init__()

        self.shortcut = shortcut 
        self.module_list = nn.ModuleList()

        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(in_ch,    in_ch//2, 1, 1))
            resblock_one.append(add_conv(in_ch//2, in_ch,    3, 1))

            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x

            for res in module:
                h = res(h)

            x = x + h if self.shortcut else h

        return x

# YOLO Layer
class YOLOLayer(nn.Module):
    def __init__(self, layer_no, in_ch, ignore_thre=0.7):
        super(YOLOLayer, self).__init__()
        layer_no = layer_no - 1

        # fixed
        stride = [32, 16, 8]
        anch_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [
            [ 10,  13], [ 16,  30], [ 33,  23],
            [ 30,  61], [ 62,  45], [ 59, 119],
            [116,  90], [156, 198], [373, 326]
            ]
        
        # 元々はcfgファイルから読み込むけどベタ打ち
        self.stride = stride[layer_no]
        self.anchors = anchors
        self.anch_mask = anch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)                # = 3

        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]

        self.n_classes = 80
        self.ignore_thre = ignore_thre

        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=self.n_anchors * (self.n_classes + 5), kernel_size=1, stride=1, padding=0)

        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)


    def forward(self, x, labels=None):
        output = self.conv(x)

        batch_size = output.shape[0]
        fsize = output.shape[2]
        n_ch = self.n_classes + 5
        dtype = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        # サイズ変換                                                                    # [b, 255, fsize, fsize]
        output = output.view(batch_size, self.n_anchors, n_ch, fsize, fsize)            # [b, 3, 85, fsize, fsize]
        output = output.permute(0, 1, 3, 4, 2)                                          # [b, 3, fsize, fsize, 85]

        # sigmoid
        #output[:, :, :, 4:] = torch.sigmoid(output[:, :, :, 4:])                        # class確率部分を0-1化
        #output[:, :, :, :2] = torch.sigmoid(output[:, :, :, :2])                        # x, y を0-1化
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])   # 上記の書き変え

        # 予測値変換
        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4]))                      # [b, 3, fsize, fsize] 横方向に0, 1...416
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))    # [b, 3, fsize, fsize] 縦方向に0, 1...416
        
        anchors = np.array(self.masked_anchors)
        w_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))  # [b, 3, fsize, fsize] 3×全要素anchor
        h_anchors = dtype(np.broadcast_to(np.reshape(anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))  # [b, 3, fsize, fsize] 3×全要素anchor

        pred = output.clone()
        pred[..., 0] += x_shift                             # 各のgrid原点+pred >> grid座標系
        pred[..., 1] += y_shift                             # 各のgrid原点+pred >> grid座標系
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors


        # not training
        if labels is None:  
            pred[..., :4] *= self.stride                    # 画像座標系変換 [b, 3, 13, 13, 85]
            pred = pred.contiguous()
            return pred.view(batch_size, -1, n_ch).data     # [b, 3*fsize*fsize, 85]

   
   
   
        # 損失計算
        pred = pred[..., :4].data

        tgt_mask = torch.zeros(batch_size, self.n_anchors, fsize, fsize, self.n_classes + 4).type(dtype)    # zeros [b, 3, fsize, fsize, 84]
        obj_mask = torch.ones(batch_size, self.n_anchors, fsize, fsize).type(dtype)                         # ones  [b, 3, fsize, fsize]
        tgt_scale = torch.zeros(batch_size, self.n_anchors, fsize, fsize, 2).type(dtype)                    # zeros [b, 3, fsize, fsize, 2]
        target = torch.zeros(batch_size, self.n_anchors, fsize, fsize, n_ch).type(dtype)                    # zeros [b, 3, fsize, fsize, 85]

   
   
   
   
        labels = labels.cpu().data                                                                          # [b, 10, 5]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        # 各座標をgrid座標軸に変換して取得
        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize

        truth_i_all = truth_x_all.to(torch.int16).numpy()   # 小数点以下を切り捨てた各グリッドの原点
        truth_j_all = truth_y_all.to(torch.int16).numpy()   # 小数点以下を切り捨てた各グリッドの原点


        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize



        for b in range(batch_size):
            n = int(nlabel[b])

            if n == 0: continue

            truth_box = dtype(np.zeros((n, 4)))     # [n, 4]
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]

            # anchor-boxとのIoU
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors) # nlabel(各正解box) × 9(anchor)のIoU出力
            best_n_all = np.argmax(anchor_ious_all, axis=1)                 # 一番IoUが高いanchorの番号
            best_n = best_n_all % 3                                        
            ##  一番IoUが高いanchorの番号が今回のyolo layerのstageと同じ場合True
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))
            
            if sum(best_n_mask) == 0: continue


            # pred-truth IoU
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            pred_ious = bboxes_iou(pred[b].contiguous().view(-1, 4), truth_box, xyxy=False) # nlabel(正解box) × 507(3*13*13)予測boxのIoU
            pred_best_iou, _ = pred_ious.max(dim=1)                                         # 各予測がどの正解レベルに対応しているか [507]
            pred_best_iou = (pred_best_iou > self.ignore_thre)      # iouが閾値を超えているか判断       
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])   # [3, 13, 13]
            obj_mask[b] = 1 - pred_best_iou.to(torch.int64)         # [3, 13, 13] 閾値を超えているgridを0にする
            
            
            # 1batch内の正解ラベルが所属するgrid原点
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]
    
            for ti in range(best_n.shape[0]):   # 1batch内に付与されているラベル分ループ
                if best_n_mask[ti] == 1:        # anchorと同じジャンルのサイズの場合
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]              # 0, 1, 2 anchor no

                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1

                    # x
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    # y
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)

                    # w
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    # h
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    
                    # obj-p
                    target[b, a, j, i, 4] = 1

                    # class
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).numpy()] = 1

                    # 重みづけ　大きな物体はおおざっぱな矩形　小さな物体は細かい違いも検出
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)


        # 最終損失計算
        output[..., 4] *= obj_mask                      # obj-p 
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask     # x, y, w, h, class >> 対象anchorと同じジャンルの矩形
        output[..., 2:4] *= tgt_scale                   # w, h 重みづけ

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale


        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale, size_average=False)

        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2



def create_yolov3_modules(ignore_thre):
    mlist = nn.ModuleList()

    # なんやかんやLayer
    mlist.append(add_conv(in_ch=   3, out_ch=  32, ksize=3, stride=1))      # 0
    mlist.append(add_conv(in_ch=  32, out_ch=  64, ksize=3, stride=2))      # 1
    mlist.append(resblock(in_ch=  64))                                      # 2
    mlist.append(add_conv(in_ch=  64, out_ch= 128, ksize=3, stride=2))      # 3
    mlist.append(resblock(in_ch= 128, nblocks=2))                           # 4
    mlist.append(add_conv(in_ch= 128, out_ch= 256, ksize=3, stride=2))      # 5
    mlist.append(resblock(in_ch= 256, nblocks=8))                           # 6
    mlist.append(add_conv(in_ch= 256, out_ch= 512, ksize=3, stride=2))      # 7
    mlist.append(resblock(in_ch= 512, nblocks=8))                           # 8
    mlist.append(add_conv(in_ch= 512, out_ch=1024, ksize=3, stride=2))      # 9
    mlist.append(resblock(in_ch=1024, nblocks=4))                           # 10

    
    mlist.append(resblock(in_ch=1024, nblocks=2, shortcut=False))           # 11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       # 12    torch.Size([b, 512, 13, 13])   
    
    # 1st YOLO branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))       # 13    torch.Size([b, 1024, 13, 13])
    mlist.append(YOLOLayer(layer_no=1, in_ch=1024, ignore_thre=ignore_thre))# 14

    # branch
    mlist.append(add_conv(in_ch= 512, out_ch= 256, ksize=1, stride=1))      # 15    12層目の出力を入力
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))               # 16    8層目を結合して入力
    mlist.append(add_conv(in_ch= 768, out_ch= 256, ksize=1, stride=1))      # 17
    mlist.append(add_conv(in_ch= 256, out_ch= 512, ksize=3, stride=1))      # 18
    mlist.append(resblock(in_ch= 512, nblocks=1, shortcut=False))           # 19
    mlist.append(add_conv(in_ch= 512, out_ch= 256, ksize=1, stride=1))      # 20

    # 2nd YOLO branch
    mlist.append(add_conv(in_ch= 256, out_ch= 512, ksize=3, stride=1))      # 21
    mlist.append(YOLOLayer(layer_no=2, in_ch= 512, ignore_thre=ignore_thre))# 22

    # branch
    mlist.append(add_conv(in_ch= 256, out_ch= 128, ksize=1, stride=1))      # 23    20層目の出力を入力
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))               # 24    6層目を結合して入力
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))        # 25
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        # 26
    mlist.append(resblock(in_ch= 256, nblocks=2, shortcut=False))               # 27
    mlist.append(YOLOLayer(layer_no=3, in_ch= 256, ignore_thre=ignore_thre))# 28

    return mlist



class YOLOv3(nn.Module):
    def __init__(self, ignore_thre):
        super(YOLOv3, self).__init__()

        self.module_list = create_yolov3_modules(ignore_thre)

    def forward(self, x, targets=None):
        train = targets is not None                     # targets: 学習データ

        output = []
        route_layers = []
        self.loss_dict = defaultdict(float)             # 空のdict用意

        for i, module in enumerate(self.module_list):

            if i in [14, 22, 28]:          # yolo layer
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                
                output.append(x)

            else: x = module(x)

            # route layer
            if i in [6, 8, 12, 20]: route_layers.append(x)
            if i == 14: x = route_layers[2]
            if i == 16: x = torch.cat((x, route_layers[1]), 1)
            if i == 22: x = route_layers[3]
            if i == 24: x = torch.cat((x, route_layers[0]), 1)


            print(i, x.size())



        if train: 
            return sum(output) 
            
        else: 
            return torch.cat(output, 1)


if __name__ == '__main__':
    cuda = False
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    dataset = COCODataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    data_iterator = iter(dataloader)
    imgs, labels, _, _ = next(data_iterator)

    #print('imgs.size():', imgs.size())
    #print('labels.size():', labels.size())

    imgs = Variable(imgs.type(dtype))
    labels = Variable(labels.type(dtype), requires_grad=False)
    
    model = YOLOv3(ignore_thre=0.002)
    model(imgs, labels)

