import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from nets.backbone import Conv


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#
# class DecoupleModule(nn.Module):
#     # ---------------------------------------------------#
#     #   split
#     # ---------------------------------------------------#
#     def __init__(self, n_in):
#         super(DecoupleModule, self).__init__()
#         self.noise_conv = Conv(n_in // 2, n_in)
#         self.object_conv = Conv(n_in // 2, n_in)
#
#     def forward(self, x):
#         noise_feature_map0, object_feature_map0 = torch.split(x, 64, dim=1)
#         noise_feature_map = self.noise_conv(noise_feature_map0)
#         object_feature_map = self.object_conv(object_feature_map0)
#         return noise_feature_map, object_feature_map, noise_feature_map0, object_feature_map0
#
# # ACCR模块
# class ACCR_regularizer(nn.Module):
#     def __init__(self, n_in):
#         super(ACCR_regularizer, self).__init__()
#         self.noise_gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.object_gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.w_noise = nn.Linear(n_in, 1, bias=False)
#         self.w_object = nn.Linear(n_in, 1, bias=False)
#
#     def forward(self, features_noise, features_object):
#         # 例如特征层shape为:bs, 80, 80, 128
#         # 80, 80, 128 => 1, 1, 128
#         noise_embedding = self.noise_gap(features_noise)
#         # 80, 80, 128 => 1, 1, 128
#         object_embedding = self.object_gap(features_object)
#         # 1, 1, 128 => 1, 128
#         noise_embedding = noise_embedding.view(-1, 128)
#         object_embedding = object_embedding.view(-1, 128)
#         # 1, 128 => 1, 1
#         vs_noise = self.w_noise(noise_embedding)
#         # 1, 128 => 1, 1
#         vs_object = self.w_object(object_embedding)
#         # 计算相关系数
#         # rho = ((vs_scale - vs_scale.mean(dim=0)) * (vs_invariant - vs_invariant.mean(dim=0))).mean(dim=0).pow(2) \
#         #       / ((vs_scale.var(dim=0) + 1e-6) * (vs_invariant.var(dim=0) + 1e-6))
#         # # print(rho)
#         # return rho, scale_embedding
#         return noise_embedding,object_embedding  # invariant_embedding


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#

        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #input->2048,7,7
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.decouple = DecoupleModule(n_in=decouple_channel)
        # self.accr = ACCR_regularizer(n_in=decouple_channel)
        # self.scale_classifier = nn.Sequential(nn.Linear(decouple_channel, decouple_channel),
        #                                       nn.LeakyReLU(inplace=True),
        #                                       nn.Linear(decouple_channel, num_classes_scale))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # features1    = list([model.conv1, model.bn1, model.relu, model.maxpool])
    # features2    = list([model.layer1,model.layer2,model.layer3,model.layer4])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    features    = nn.Sequential(*features)
    # features1    = nn.Sequential(*features1)
    # features2    = nn.Sequential(*features2)
    classifier   = nn.Sequential(*classifier)
    return features, classifier
