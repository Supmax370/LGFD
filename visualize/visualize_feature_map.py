import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from nets.yolo import YoloBody

input_shape = [640, 640]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 8
phi = 'l'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YoloBody(anchors_mask, num_classes, phi, False).to(device)
model.load_state_dict(torch.load("/data/user7/wt/code/FLIR-Aligned/YOLO-DECOUPLE/logs/ep090-loss0.326-val_loss0.467.pth", map_location=device), strict=False)
model = model.eval()

# 图像路径
# image_path = r'D:\Deep_Learning_folds\Datasets\State-Air\VOC2007\JPEGImages\1697446959026_h_30.50m_roll_-1.70_pitch_-1.00_yaw_-2.30.jpg'
image_path = '/data/user7/wt/datasets/FLIR-Aligned/VOCdevkit/VOC2007/JPEGImages/FLIR_08907.jpg'
caption = 'AAAA'
heatmap_save_path = "visualize/heatmap_vision.jpg"
# r'D:\Deep_Learning_folds\Datasets\State-Air\VOC2007\JPEGImages\1698715065588_h_30.90m_roll_3.70_pitch_22.60_yaw_63.80.jpg'
# r'D:\Deep_Learning_folds\Datasets\State-Air\VOC2007\JPEGImages\1697446959026_h_30.50m_roll_-1.70_pitch_-1.00_yaw_-2.30.jpg'
 # 'VOCdevkit/VOC2007/JPEGImages/frame_20190905143505_x_0003197.jpg'

# 加载图像
image = Image.open(image_path)
iw, ih  = image.size
w, h    = input_shape
new_image = image.resize((w, h), Image.BICUBIC)

# 将图像转换为numpy数组
image_array = np.array(new_image)
# 将numpy数组转换为PyTorch张量
input_tensor = torch.from_numpy(image_array)
# 获取模型权重的数据类型
weight_dtype = next(model.parameters()).dtype
# 添加batch维度
input_tensor = torch.unsqueeze(input_tensor, dim=0).to(dtype=weight_dtype, device=device)
input_tensor = input_tensor.permute(0, 3, 1, 2)
# 打印输入张量的形状
print(input_tensor.shape)

[out0, out1, out2], object_feature, noise_feature, logits_per_image = model(input_tensor,caption)
object_feature3, object_feature4 = object_feature
noise_feature3, noise_feature4 = noise_feature
logits_per_image3, logits_per_image4 = logits_per_image
print(object_feature3)
# 假设特征层的张量名为feature_map，shape为(batch_size, num_channels, height, width)
feature_map = object_feature3.cpu()

# 将特征层的张量转换为NumPy数组
feature_map_np = feature_map.detach().numpy()

# 叠加通道并显示
combined_feature_map = np.sum(feature_map_np[0], axis=0)  # 叠加第一个样本的所有通道


# 创建一个自定义的colormap
colors = [[0.26666667, 0.00392157, 0.32941176], "blue"]  # 紫色用于背景，绿色用于特征
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# 使用plt.imshow绘制图像，并应用自定义的colormap
plt.imshow(combined_feature_map, cmap=cmap)
#plt.imshow(combined_feature_map)
plt.colorbar()  # 可选，显示颜色条
plt.show()

"""def onclick(event):
    ix, iy = int(event.xdata), int(event.ydata)
    print('x = %d, y = %d' % (ix, iy))
    print('RGB =', combined_feature_map[iy, ix])  # 显示点击位置的像素值

fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()"""

#plt.imshow(combined_feature_map)

"""# 假设特征层的张量名为feature_map，shape为(batch_size, num_channels, height, width)
feature_map2 = Scale_feature.cpu()

# 将特征层的张量转换为NumPy数组
feature_map_np2 = feature_map2.detach().numpy()

# 叠加通道并显示
combined_feature_map2 = np.sum(feature_map_np2[0], axis=0)  # 叠加第一个样本的所有通道
plt.imshow(combined_feature_map2)
plt.show()"""