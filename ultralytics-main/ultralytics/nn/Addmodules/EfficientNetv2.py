import timm
import torch.nn as nn


class EfficientNetV2Backbone(nn.Module):
    def __init__(self, model_name='efficientnetv2_s', out_indices=(3, 4, 5)):
        super().__init__()
        self.model = timm.create_model(model_name, features_only=True, out_indices=out_indices)

        # 获取通道数（示例：EfficientNetV2-S 的特定层输出通道）
        self.channels = self.model.feature_info.channels

    def forward(self, x):
        features = self.model(x)
        return features  # 输出多尺度特征图