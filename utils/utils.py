import torch
import numpy as np

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4: raise IndexError


    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)        
    else:
        # top left
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),    # x-w/2, y-h/2
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)                 # x-w/2, y-h/2
        )
        # bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),    # x-w/2, y-h/2
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)                 # x-w/2, y-h/2
        )

        #print(tl.size(), br.size())                                # [a, 9(anchor), 2(tl, br)]           

        area_a = torch.prod(bboxes_a[:, 2:], 1)                     # w*h = 面積
        area_b = torch.prod(bboxes_b[:, 2:], 1)



    area_i = torch.prod(br - tl, 2) * (tl < br).all()

    return area_i / (area_a[:, None] + area_b - area_i)
    


    