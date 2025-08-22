from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3, A2C2f
from ultralytics.nn.modules.block import C3k

try:
    from mmcv.cnn import ConvModule, build_norm_layer
    from mmengine.model import BaseModule
    from mmengine.model import constant_init
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError as e:
    pass

__all__ = ['MoCAA', 'C3k2_Mona', 'C2f_Mona', 'C3k2_DyTMona', 'C2f_DyTMona', 'C3k2_MoCAA', 'C2f_MoCAA', 'A2C2f_Mona']
# 论文地址：https://arxiv.org/abs/2408.08345
'''
来自CVPR2025顶会
即插即用模块: Mona 多认知视觉适配器模块 （全称：Multi-cognitive Visual Adapter）
两个二次创新模块: DyTMona, MoCAA 好好编故事，可以直接拿去冲SCI一区、二区、三区

本文核心内容：
预训练和微调可以提高视觉任务中的传输效率和性能。最近的增量调整方法为视觉分类任务提供了更多选择。
尽管他们取得了成功，但现有的视觉增量调整艺术未能超过在对象检测和分割等具有挑战性的任务上完全微调的上限。
为了找到完全微调的有竞争力的替代方案，我们提出了多认知视觉适配器 （Mona） 调整，这是一种新颖的基于适配器的调整方法。
首先，我们在适配器中引入了多个视觉友好型滤波器，以增强其处理视觉信号的能力，而以前的方法主要依赖于语言友好的线性耳罩。
其次，我们在适配器中添加缩放的归一化层，以调节虚拟滤波器的输入特征分布。
为了充分展示 Mona 的实用性和通用性，我们对多个表征视觉任务进行了实验，包括 COCO 上的实例分割、
ADE20K 上的语义分割、Pas cal VOC 上的目标检测、DOTA/STAR 上的定向对象检测和三个常见数据集上的年龄分类。
令人兴奋的结果表明，Mona 在所有这些任务上都超过了完全微调，并且是唯一一种在上述各种任务上优于完全微调的增量调优方法。
例如，与完全微调相比，Mona 在 COCO 数据集上实现了 1% 的性能提升。
综合结果表明，与完全微调相比，Mona 调优更适合保留和利用预训练模型的能力。

适用于：语义分割、目标检测、实例分割、图像分类、图像增强等等所有CV任务都用的上，通用的即插即用模块
'''


class CAA(nn.Module):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        AiFHG = x
        conv1_x = self.conv1_AiFHG(x)
        conv2_x = self.conv2_AiFHG(x)
        conv3_x = self.conv3_AiFHG(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + AiFHG

        AiFHG = x

        x = self.projector(x)

        return AiFHG + x


class Mona(nn.Module):
    def __init__(self, in_dim, AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)

        self.dropout_AiFHG = nn.Dropout(p=0.1)

        self.adapter_conv_AiFHG = MonaOp(64)

        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)
        b, n, c = project1.shape
        h, w = H, W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# 二次创新模块：MoCAA
class MoCAA(nn.Module):  # Multi-cognitive Context Anchor Attention
    def __init__(self, in_dim, AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)

        self.dropout_AiFHG = nn.Dropout(p=0.1)

        self.adapter_conv_AiFHG = CAA(64)

        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(-1, -2)
        AiFHG = x
        x = self.norm_AiFHG(x) * self.gamma_AiFHG + x * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x)
        b, n, c = project1.shape
        h, w = H, W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


# 二次创新模块：DyTMona
class DyTMona(nn.Module):
    def __init__(self, in_dim, AiFHG=4):
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)
        self.dropout_AiFHG = nn.Dropout(p=0.1)
        self.adapter_conv_AiFHG = MonaOp(64)
        self.norm_AiFHG = DynamicTanh(normalized_shape=in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B, C, H, W = x.shape

        x_dyt = self.norm_AiFHG(x).reshape(B, C, -1).transpose(-1, -2)
        x = x.reshape(B, C, -1).transpose(-1, -2)
        x = x_dyt * self.gamma_AiFHG + x * self.gammax_AiFHG
        AiFHG = x
        project1 = self.project1_AiFHG(x)
        b, n, c = project1.shape
        h, w = H, W
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv_AiFHG(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


# class Bottleneck_Mona(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#         self.Attention = Mona(c2)
#
#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
#


class C2f_Mona(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Mona(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_Mona(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Mona(c_) for _ in range(n)))


class C3k2_Mona(C2f_Mona):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_Mona(self.c, self.c, 2, shortcut, g) if c3k else Mona(self.c) for _ in range(n)
        )


class C2f_DyTMona(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(DyTMona(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_DyTMona(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DyTMona(c_) for _ in range(n)))


class C3k2_DyTMona(C2f_Mona):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_DyTMona(self.c, self.c, 2, shortcut, g) if c3k else DyTMona(self.c) for _ in range(n)
        )


class C2f_MoCAA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MoCAA(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_MoCAA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MoCAA(c_) for _ in range(n)))


class C3k2_MoCAA(C2f_Mona):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_MoCAA(self.c, self.c, 2, shortcut, g) if c3k else MoCAA(self.c) for _ in range(n)
        )


# yolov12中的改进
class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class ABlock_Mona(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.mona1 = Mona(dim)
        self.mona2 = Mona(dim)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = self.mona1(x + self.attn(x))
        return self.mona2(x + self.mlp(x))


class ABlock_DyT(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.dyt1 = DynamicTanh(normalized_shape=dim)
        self.dyt2 = DynamicTanh(normalized_shape=dim)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(self.dyt1(x))
        return x + self.mlp(self.dyt2(x))


class A2C2f_Mona(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)

        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_Mona(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )


class A2C2f_DyT(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)

        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_DyT(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 创建Mona模块实例，32代表通道维度
    Mona = Mona(80)
    #   输入 B C H W, 输出 B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input = torch.randn(1, 80, 32, 32)
    # 运行前向传递
    output = Mona(input)
    # 输出输入图片张量和输出图片张量的形状
    print("CV_Mona_input size:", input.size())
    print("CV_Mona_Output size:", output.size())

    # 创建DyTMona模块实例，32代表通道维度
    DyTMona = DyTMona(80)
    # 随机生成输入4维度张量：B, C, H, W
    input = torch.randn(1, 80, 32, 32)
    # 运行前向传递
    output = DyTMona(input)
    # 输出输入图片张量和输出图片张量的形状
    print("何恺明大神之作—DyTMona_input size:", input.size())
    print("何恺明大神之作—DyTMona_Output size:", output.size())

    # 创建MCAA模块实例，32代表通道维度
    MoCAA = MoCAA(32)
    # 随机生成输入4维度张量：B, C, H, W
    input = torch.randn(1, 32, 32, 32)
    # 运行前向传递
    output = MoCAA(input)
    # 输出输入图片张量和输出图片张量的形状
    print("MoCAA_input size:", input.size())
    print("MoCAA_Output size:", output.size())
