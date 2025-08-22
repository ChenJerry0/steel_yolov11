from ultralytics.nn.modules import Conv
import torch
import torch.nn as nn

__all__ = ['Pzconv']

class Pzconv(nn.Module):
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.conv2 = Conv(dim, dim, k=1, s=1)
        self.conv3 = nn.Conv2d(
            dim, dim, 5,
            1, 2, groups=dim
        )
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(
            dim, dim, 7,
            1, 3, groups=dim
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6
if __name__ == '__main__':
    # 定义输入张量的形状为 B, C, H, W
    input = torch.randn(1, 32, 64,64)
    # 创建一个 LRSA 模块实例
    lrsa = Pzconv(dim=32)  # 这里 dim=3 对应输入通道数
    # 执行前向传播
    output = lrsa(input)
    # 打印输入和输出的形状
    print('input_size:', input.size())
    print('output_size:', output.size())
