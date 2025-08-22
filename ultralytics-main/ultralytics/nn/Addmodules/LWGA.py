import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import List
from torch import Tensor
import antialiased_cnns
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from ultralytics.nn.modules import C3

# 论文地址： https://arxiv.org/pdf/2501.10040
'''
来自Arxiv 2025最新论文  LWGA 轻量级分组注意力模块

遥感（RS）视觉任务在学术研究和实际应用中具有重要意义。然而，这些任务面临着多个挑战，尤其是在单幅图像中检测和识别尺度变化显著的多个目标，
这严重影响了特征提取的有效性。尽管以往的双分支或多分支架构在一定程度上缓解了尺度变化的问题，但也显著增加了计算开销和参数量，
使得这些架构难以部署在资源受限的设备上。此外，当前主流的轻量级骨干网络多为自然图像设计，在应对多尺度目标方面表现不佳，从而限制了其在遥感视觉任务中的表现。

本文提出了一种专门面向遥感视觉任务设计的轻量级骨干网络——LWGANet，其中引入了一种新颖的注意力机制模块：轻量级分组注意力（LWGA）模块。
LWGA 模块针对遥感图像特点，充分利用特征图中的冗余信息，在不增加额外复杂度和计算负担的前提下，从局部到全局提取多尺度空间信息，从而在保持高效率的同时实现精确的特征提取。

LWGANet 在场景分类、定向目标检测、语义分割和变化检测四类关键遥感视觉任务中共计12个数据集上进行了系统评估。
实验结果表明，LWGANet 在多个数据集上实现了先进性能，并且在高性能与低复杂度之间实现了优异的平衡，展现出良好的通用性与实用性。
总体而言，LWGANet 为资源受限场景下对遥感图像进行高效处理提供了一种新颖而有效的解决方案。
'''
__all__ = ['LWGA','C2f_LWGA','C3k2_LWGA']
class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1, bias=False),
            build_norm_layer(norm_layer, dim*4)[1],
            act_layer(),
            nn.Conv2d(dim*4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x
class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = build_norm_layer(norm_layer, channel)[1]
    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x
    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x
    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)
    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)
class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x
class D_GA(nn.Module):

    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x
class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LWGA(nn.Module):
    def __init__(self,
                 dim=64,  # 输入特征图的通道数
                 stage=1,  # 网络的stage，决定使用哪种GA（GA12）
                 att_kernel=7,  # MRA中的注意力卷积核大小
                 mlp_ratio=4.0,  # MLP通道扩展比例
                 drop_path=0.1,  # 不使用DropPath
                 act_layer=nn.GELU,  # 激活函数
                 norm_layer=dict(type='BN', requires_grad=True)  # 使用 BatchNorm
                 ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention

        # 用于特征融合的1x1卷积
        self.fuse_conv1 = nn.Conv2d(self.dim_split * 2, self.dim_split, 1, bias=False)
        self.fuse_conv2 = nn.Conv2d(self.dim_split * 2, self.dim_split, 1, bias=False)

        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)  # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)  # GA4 is global attention (stage of 4)
            self.norm = build_norm_layer(norm_layer, self.dim_split)[1]
        else:
            self.GA12 = GA12(self.dim_split, act_layer)  # GA12 is global attention (stages of 1 and 2)
            self.norm = build_norm_layer(norm_layer, self.dim_split)[1]

        self.norm1 = build_norm_layer(norm_layer, dim)[1]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # 保存原始输入作为残差连接
        shortcut = x.clone()

        # 分割输入特征为四个组
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)

        # 1. 点注意力路径
        x1_pa = self.PA(x1)
        x1 = x1 + x1_pa  # 残差连接

        # 2. 局部注意力路径：融合点注意力输出和原始x2
        # 将PA的输出与x2连接，然后融合
        x2_fused = torch.cat([x1_pa, x2], dim=1)
        x2_fused = self.fuse_conv1(x2_fused)
        x2 = self.LA(x2_fused)

        # 3. 中程注意力路径
        x3_mra = self.MRA(x3)
        x3 = x3 + x3_mra  # 残差连接

        # 4. 全局注意力路径：融合中程注意力输出和原始x4
        # 将MRA的输出与x4连接，然后融合
        x4_fused = torch.cat([x3_mra, x4], dim=1)
        x4_fused = self.fuse_conv2(x4_fused)

        # 根据阶段选择不同的全局注意力
        if self.stage == 2:
            x4 = x4 + self.GA3(x4_fused)
        elif self.stage == 3:
            x4_ga = self.GA4(x4_fused)
            x4 = self.norm(x4 + x4_ga)
        else:
            x4_ga = self.GA12(x4_fused)
            x4 = self.norm(x4 + x4_ga)

        # 合并所有分支的输出
        x_att = torch.cat((x1, x2, x3, x4), 1)

        # 应用MLP和残差连接
        x_mlp = self.mlp(x_att)
        x_mlp = self.norm1(x_mlp)
        x = shortcut + self.drop_path(x_mlp)

        return x

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
class Bottleneck_LWGA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True,stage=1, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = LWGA(c2,stage)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_LWGA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, stage=1,g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_LWGA(self.c, self.c, shortcut,stage, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True,stage=1,  g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut,  g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_LWGA(c_, c_, shortcut,stage,  g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))


class C3k2_LWGA(C2f_LWGA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, stage=1, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut,stage,  g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, stage, g) if c3k else Bottleneck_LWGA(self.c, self.c, shortcut,stage,  g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

if __name__ == "__main__":
    # 定义输入特征图大小：Batch=1, Channel=64, H=32, W=32
    input= torch.randn(1, 64, 32, 32)
    # 初始化 LWGA
    block = LWGA(
        dim=64,            # 输入特征图的通道数
        stage=1,           # 网络的stage，决定使用哪种GA（GA12）
        att_kernel=7,      # MRA中的注意力卷积核大小
        mlp_ratio= 4.0 ,   # MLP通道扩展比例
        drop_path= 0.1 ,   # 不使用DropPath
        act_layer= nn.GELU , # 激活函数
        norm_layer= dict(type='BN', requires_grad=True)    # 使用 BatchNorm
    )
    # 前向传播
    output = block(input)
    # 输出张量信息
    print(f"Ai缝合怪永久更新中—LWGA Input shape:  {input.shape}")
    print(f"Ai缝合怪永久更新中—LWGA Output shape: {output.shape}")