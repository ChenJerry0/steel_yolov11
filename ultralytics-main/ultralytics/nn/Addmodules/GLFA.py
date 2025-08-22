import math
import torch
from torch import nn
import torch.nn.functional as F
__all__ = ['GLFA','GLFAResBlock']

class GLFA(nn.Module):
    """
    Global-Local Fusion Attention with Channel Reward Mechanism
    结合全局上下文与局部细节的多尺度注意力模块

    主要改进点：
    1. 双路注意力分支（全局通道+局部空间）
    2. 通道奖励重校准机制
    3. 多尺度特征金字塔融合
    4. 动态特征门控
    """

    def __init__(self, ch_in, dim, reduction_ratio=16, kernel_sizes=[3, 5, 7], groups=4):
        super().__init__()
        self.dim = dim
        self.groups = groups

        # 通道压缩
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(ch_in, dim, 1, groups=groups),  # 分组卷积压缩
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 全局通道奖励分支 (使用通道相关性)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            # ChannelRewardMechanism(dim)  # 新增奖励机制
        )

        # 局部多尺度空间分支
        self.local_branch = MultiScaleSpatial(dim, kernel_sizes)  # 使用 dim 作为输入通道数

        # 混合注意力机制
        self.attn = nn.ModuleDict({
            'qkv': nn.Conv2d(dim, 3 * dim, 1, bias=False),
            'oper_q': nn.Sequential(
                SpatialOperation(dim),
                ChannelOperation(dim),
            ),
            'oper_k': nn.Sequential(
                SpatialOperation(dim),
                ChannelOperation(dim),
            ),
            'dwc': nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            'proj': nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        })

        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, 3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        # 调整通道数
        self.adjust_channels = nn.Conv2d(dim, ch_in, 1)

    def forward(self, x):
        # 原始特征保留
        identity = x

        # 特征压缩
        reduced = self.conv_reduce(x)

        # 全局通道奖励路径
        global_feat = self.global_branch(reduced)

        # 局部多尺度路径
        local_feat = self.local_branch(reduced)

        # 确保全局特征的空间尺寸与局部特征一致
        global_feat = F.interpolate(global_feat, size=local_feat.shape[-2:], mode='nearest')

        # 动态门控融合
        combined = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(combined)
        fused_feat = gate[:, 0:1] * global_feat + gate[:, 1:2] * local_feat

        # 混合注意力机制
        q, k, v = self.attn['qkv'](fused_feat).chunk(3, dim=1)
        q = self.attn['oper_q'](q)
        k = self.attn['oper_k'](k)
        out = self.attn['proj'](self.attn['dwc'](q + k) * v)

        # 调整通道数
        out = self.adjust_channels(out)

        # 残差连接
        return identity + out


class ChannelRewardMechanism(nn.Module):
    """通道奖励机制：动态调整特征重要性"""

    def __init__(self, dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 通道级奖励权重
        channel_weights = F.softmax(x / self.temperature, dim=1)
        # 自适应缩放
        return x * self.alpha * channel_weights


class MultiScaleSpatial(nn.Module):
    """多尺度空间特征提取"""

    def __init__(self, dim, kernel_sizes):
        super().__init__()
        self.scale_convs = nn.ModuleList()
        for k in kernel_sizes:
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(True)
                )
            )
        self.fusion = nn.Conv2d(dim * len(kernel_sizes), dim, 1)

    def forward(self, x):
        features = [conv(x) for conv in self.scale_convs]
        return self.fusion(torch.cat(features, dim=1))


class FeaturePyramid(nn.Module):
    """特征金字塔增强模块"""

    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )
        self.conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        down_feat = self.down1(x)
        up_feat = self.up1(down_feat)
        return self.conv(torch.cat([x, up_feat], dim=1))


class GLFAResBlock(nn.Module):
    """集成GLFA的增强残差块"""

    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            GLFA(dim, dim),  # 修复 GLFA 初始化参数
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

