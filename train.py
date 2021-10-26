import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.cocodataset import COCODataset
from models.yolov3 import YOLOv3


def main():

    cuda = torch.cuda.is_available() and args.use_cuda
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    iter_size = 10
    batch_size = 4
    decay = 0.0005
    subdivision = 16
    momentum = 0.9
    base_lr = 0.001 / batch_size / subdivision
    weight_decay = decay * batch_size * subdivision

    # data
    dataset = COCODataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataiterator =iter(dataloader)

    # model
    model = YOLOv3(ignore_thre=0.5)
    model.train()
    ## weigths解析　実装予定

    # optimizer
    params_dict = dict(model.named_parameters())

    params = []
    for key, value in params_dict.items():
        # convの場合
        if 'conv.weight' in key:   
            params += [{'params': value, 'weight_decay': weight_decay}]
        # その他
        else:
            params += [{'params': value, 'weight_decay': 0.0}]

    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum, dampening=0, weight_decay=weight_decay)


    # start training
    iter_state = 0

    for i in range(iter_state, iter_size+1):
        optimizer.zero_grad()

        for j in range(subdivision):
            # データ取得
            try:
                imgs, targets, _, _ = next(dataiterator)
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)

            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)

            # 損失計算
            loss = model(imgs, targets)
            loss.backward()

        optimizer.step()

        print('epoch:', i)






if __name__ == '__main__':
    # GPUまたはCPUのどちらが使用できるか表示
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用プロセッサ:", DEVICE)

    main()