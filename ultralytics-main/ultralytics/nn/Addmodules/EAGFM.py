import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
# https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Li_Multi-Level_Feature_Fusion_Network_for_Lightweight_Stereo_Image_Super-Resolution_CVPRW_2024_paper.pdf
# 代码地址论文里面有
__all__ =['CVIM','EAGFM']


'''
来自CVPR 2024 顶会 
即插即用模块：CVIM 跨视图交互模块 （特征融合模块）
含二次创新模块 EAGFM 有效注意力引导特征融合模块 
EGAFM是CVIM模块的二次创新，效果优于CVIM ，可以直接拿去发小论文，冲SCI一区或是二区

CVIM模块的主要目标是通过有效的左右视图交互，实现图像的跨视图信息融合，
从而提高图像超分辨率的性能，增强图像细节特征。具体来说：
1.增强跨视图信息共享：通过左右视图之间的特征交互，提取视图之间的互补信息。
2.减少计算复杂度：通过优化输入特征的维度和去除冗余操作，显著降低了计算开销，使其适合于轻量化网络。
3.提高跨视图融合效率：结合左右视图的特征，提高重建图像细节和纹理的准确性。
CVIM通过轻量化的跨视图交互机制，高效特征提取和融合左右视图的互补信息，
显著提升图像的超分辨率性能，同时保持低计算复杂度和高效集成能力。

特征融合模块适用于所有计算机视觉CV任务，通用的即插即用模块
'''
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
class CVIM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )

        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )

        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x ):
        x_l, x_r =x
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
        F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))
        print((x_l).shape)
        return x_l + F_r2l + x_r + F_l2r


# 顶会二次创新模块 EAGFM 有效注意力引导融合模块  可以直接去发小论文 冲sci一区和二区
# 二次创新的思路：增加了一个注意力模块，使用门控机制获取权重，通过权重值进一步增强x_l和x_r特征进行残差连接
class EAGFM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )

        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )

        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pa = PixelAttention(c)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_l, x_r = x
        initial = x_l + x_r
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c
        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c
        # scale
        F_r2l = self.l_proj3(F_r2l.permute(0, 3, 1, 2))
        F_l2r = self.r_proj3(F_l2r.permute(0, 3, 1, 2))
        a = self.sigmoid(self.pa(initial, F_l2r + F_r2l))
        print((a * x_l + (1 - a) * x_r).shape)
        return a * x_l + (1 - a) * x_r


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 初始化CVIM模块并设定通道维度
    CVIM_module = CVIM(32)
    output = CVIM_module([input1, input2])
    # 输出结果的形状
    print("CVIM_输入张量的形状：", input1.shape)
    print("CVIM_输出张量的形状：", output.shape)
    # 初始化EAGFM模块并设定通道维度
    EGAFM_module = EAGFM(32)
    output = EGAFM_module([input1, input2])
    # 输出结果的形状
    print("二次创新EAGFM_输入张量的形状：", input1.shape)
    print("二次创新EAGFM_输出张量的形状：", output.shape)

