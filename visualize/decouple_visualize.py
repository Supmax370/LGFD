import cv2
import matplotlib.pyplot as plt
import torch
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
import numpy as np
from PIL import Image
from nets.yolo import YoloBody
from yolo import YOLO

def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y
# ---------------------------------------------------------#
#   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
# 图像路径
image_path = '/data/user7/wt/datasets/FLIR-Aligned/VOCdevkit/VOC2007/JPEGImages/FLIR_08907.jpg'
caption = 'AAAA'
heatmap_save_path = "visualize/heatmap_vision.png"

image = Image.open(image_path)
image = cvtColor(image)
# ---------------------------------------------------------#
#   给图像增加灰条，实现不失真的resize
#   也可以直接resize进行识别
# ---------------------------------------------------------#
image_data = resize_image(image, (640, 640), True)
# ---------------------------------------------------------#
#   添加上batch_size维度
# ---------------------------------------------------------#
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

with torch.no_grad():
    images = torch.from_numpy(image_data)
    if True:
        images = images.cuda()
    # ---------------------------------------------------------#
    #   将图像输入网络当中进行预测！
    # ---------------------------------------------------------#
    yolo = YOLO()
    yolo.detect_heatmap(image, heatmap_save_path)

