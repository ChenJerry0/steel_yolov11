import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
from ultralytics.nn.modules import C3, Bottleneck

try:
    from mmcv.cnn import ConvModule, build_norm_layer
    from mmengine.model import BaseModule
    from mmengine.model import constant_init
    from mmengine.model.weight_init import trunc_normal_init, normal_init
except ImportError as e:
    pass



__all__ =['MGLFM','C3k2_GLMM','C2f_GLMM','C3k2_Mona']
#论文题目： STMNet: Single-Temporal Mask-based Network for Self-Supervised Hyperspectral Change Detection
'''
来自TGRS 2025顶刊
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        # H, W
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]
        return embeddings

class LMM(nn.Module):
    def __init__(self, channels):
        super(LMM, self).__init__()
        self.channels = channels
        dim = self.channels
        # 3*7conv
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 2, dim * 3)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        N, C, H, W = x.shape
        x_w = self.fc_h(x)
        x_h = self.fc_w(x)
        x_add = x_h + x_w + x
        att = F.adaptive_avg_pool2d(x_add, output_size=1)
        att = self.reweight(att).reshape(N, C, 3).permute(2, 0, 1)
        att = self.swish(att).unsqueeze(-1).unsqueeze(-1)
        x_att = x_h * att[0] + x_w * att[1] + x * att[2]

        return x_att

class GMM(nn.Module):
    def __init__(self, channels, H, W):
        super(GMM, self).__init__()
        # 添加通道校验
        assert channels % 4 == 0, "Channels must be divisible by 4"
        self.channels = channels
        patch = 4
        self.C = int(channels / patch)
        # 添加自适应padding
        self.proj_h = nn.Conv2d(H * self.C, self.C * H, 3,
                              padding=1, groups=self.C, bias=True)
        self.proj_w = nn.Conv2d(W * self.C, self.C * W, 3,
                              padding=1, groups=self.C, bias=True)
        self.fuse_h = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        self.fuse_w = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)

        self.relate_pos_h = RelativePosition(channels, H)
        self.relate_pos_w = RelativePosition(channels, W)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)

    def forward(self, x):
        N, C, H, W = x.shape
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        C1 = int(C / self.C)

        x_h = x + pos_h
        # Splitting & Concatenate
        x_h = x_h.view(N, C1, self.C, H, W)
        # Column
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w
        # Row
        x_w = self.proj_w(x_h.view(N, C1, H * self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)
        x = self.fuse_w(torch.cat([x, x_w], dim=1))
        return x

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class GLMonaOp(nn.Module):
    def __init__(self, in_features,H,W):
        super().__init__()
        self.conv1_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3_AiFHG = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
        self.lmm = LMM(in_features)
        self.gmm = GMM(in_features,H,W)
    def forward(self, x):
        AiFHG = x
        x=self.lmm(self.gmm(x))
        conv1_x = self.conv1_AiFHG(x)
        conv2_x = self.conv2_AiFHG(x)
        conv3_x = self.conv3_AiFHG(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + AiFHG

        AiFHG = x

        x = self.projector(x)

        return AiFHG + x

class GLMona(nn.Module):
    def __init__(self, in_dim, H, W):  # 新增空间维度参数
        super().__init__()
        self.project1_AiFHG = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2_AiFHG = nn.Linear(64, in_dim)
        self.dropout_AiFHG = nn.Dropout(p=0.1)
        # 传入空间维度参数
        self.adapter_conv_AiFHG = GLMonaOp(64, H, W)
        self.norm_AiFHG = nn.LayerNorm(in_dim)
        self.gamma_AiFHG = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax_AiFHG = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        AiFHG = x_reshaped
        x_norm = self.norm_AiFHG(x_reshaped) * self.gamma_AiFHG + x_reshaped * self.gammax_AiFHG
        project1 = self.project1_AiFHG(x_norm)
        # 精确维度变换
        project1_reshaped = project1.reshape(B, H, W, 64).permute(0, 3, 1, 2)
        project1_conv = self.adapter_conv_AiFHG(project1_reshaped)
        project1_conv_reshaped = project1_conv.permute(0, 2, 3, 1).reshape(B, -1, 64)
        nonlinear = self.nonlinear(project1_conv_reshaped)
        nonlinear = self.dropout_AiFHG(nonlinear)
        project2 = self.project2_AiFHG(nonlinear)
        out = AiFHG + project2
        # 精确恢复维度
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

class GLMM(nn.Module):
    def __init__(self, dim,H,W):
        super(GLMM,self).__init__()
        self.lmm = LMM(dim)
        self.gmm = GMM(dim,H,W)

    def forward(self, x):
        return self.lmm(self.gmm(x))
'''二次创新模块:MGLFM 多尺度全局局部特征融合模块'''
class MGLFM(nn.Module):
    def __init__(self, dim, H,W):
        super(MGLFM, self).__init__()
        self.LLM = LMM(dim)
        self.GMM = GMM(dim, H,W)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data[0],data[1]
        initial = x + y
        # LLM = self.LLM(initial)
        # GMM = self.GMM(initial)
        # pattn1 = LLM+GMM
        pattn1 = self.LLM(self.GMM(initial))
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

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

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, H=20,shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

class C2f_GLMM(C2f):
    def __init__(self, c1, c2, n=1, H=20, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n,H, shortcut, g, e)
        self.m = nn.ModuleList(GLMM(dim=self.c, H=H,W=H) for _ in range(n))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, H=20,n=1, g=1, e=0.5):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n,  g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLMM(c_,H,H) for _ in range(n)))

class C3k2_GLMM(C2f_GLMM):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2,  n=1,H=20, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n,H, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k2_GLMM(self.c, self.c,H, 2, g) if c3k else GLMM(self.c, H,H) for _ in range(n)
        )




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

class C2f_Mano(C2f):
    def __init__(self, c1, c2, n=1, H=20, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n,H, shortcut, g, e)
        self.m = nn.ModuleList(GLMona(in_dim=self.c, H=H,W=H) for _ in range(n))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, H=20,n=1, g=1, e=0.5):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n,  g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLMona(c_,H,H) for _ in range(n)))

class C3k2_Mona(C2f_Mano):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2,  n=1,H=20, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n,H, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k2_Mona(self.c, self.c, H, 2,n=2, c3k=False) if c3k else GLMona(self.c, H, H) for _ in range(n)
        )
# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W

    input= torch.randn(1, 32, 64, 64)
    # 创建 GMM 模块
    gmm = GMM(channels=32,H=64,W=64)
    # 将输入图像传入GMM 模块进行处理
    output = gmm(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('全局混合模块_GMM_input_size:', input.size())
    print('全局混合模块_GMM_output_size:', output.size())

    # 创建 LMM 模块
    lmm = LMM(channels=32)
    # 将输入图像传入LMM模块进行处理
    output = lmm(input)
    # 输出结果的形状
    # 打印输入和输出的形状
    print('局部混合模块LMM_input_size:', input.size())
    print('局部混合模块LMM_output_size:', output.size())

    #二次创新模块MGLFM 多尺度全局局部特征融合模块
    block = MGLFM(dim=32,H=64,W=64)
    input1 = torch.rand(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)
    output = block([input1, input2])
    print('二次创新模块——MGLFM_input_size:', input.size())
    print('二次创新模块——MGLFM_output_size:', output.size())

    Mona = GLMona(80,32,32)
    #   输入 B C H W, 输出 B C H W
    # 随机生成输入4维度张量：B, C, H, W
    input= torch.randn(1, 80,32,32)
    # 运行前向传递
    output = Mona(input)
    # 输出输入图片张量和输出图片张量的形状
    print("CV_Mona_input size:", input.size())
    print("CV_Mona_Output size:", output.size())