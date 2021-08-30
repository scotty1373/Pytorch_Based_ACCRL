# -*- coding: utf-8 -*-
import numpy as np
import torch
torch.hub.set_dir('./')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# 单目标斜对角坐标
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = int(x[:, 0])
    y[:, 1] = int(x[:, 1])
    y[:, 2] = int(x[:, 2])
    y[:, 3] = int(x[:, 3])
    return y

result = model('./zidane.jpg')

data = {}

# 使用字典存储锚框信息, xyhw modeul
# for index, x in enumerate(result.pred):
#     auchor = np.zeros((1, len(result.pred[index]), 5))
#     auchor_num = len(result.pred[index])
#     auchor[0, :, :4] = xyxy2xywh(x)[:, :4]
#     auchor[0, :, 4] = x[:, 5]
#     data[f'auchor{index}'] = auchor

# xyxy module
for index, x in enumerate(result.pred):
    auchor = np.zeros((1, len(result.pred[index]), 5))
    auchor_num = len(result.pred[index])

    auchor[0, :, :4] = xyxy2xywh(x)[:, :4]
    auchor[0, :, 4] = x[:, 5]
    data[f'auchor{index}'] = auchor

print(f'auchor data: {auchor}')


