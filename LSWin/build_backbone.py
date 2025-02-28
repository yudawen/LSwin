from models.LSwin_backbone import LSwin_backbone
import torch

backbone=LSwin_backbone()

# print(backbone)

# 模拟输入数据
inputs = torch.randn(2, 3, 512, 512)# B × C × H × W

# 将模型和数据移动到CPU上
device = torch.device("cpu")
backbone.to(device)
inputs = inputs.to(device)

# 提取特征
features = backbone(inputs)

print("Output feature size of the backbone:")
print(features.size())# B × C × H × W
