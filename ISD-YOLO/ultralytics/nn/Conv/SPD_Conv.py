######################  SPD-Conv  ####     start ###############################
import math
import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

######################  SPD-Conv  ####     start ###############################


import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class LightweightAttention(nn.Module):
    """轻量级通道注意力（尺寸安全版）"""

    def __init__(self, channels, ratio=0.25):
        super().__init__()
        mid_channels = max(8, channels // 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = Conv(channels, mid_channels, k=1, act=True)
        self.out_conv = Conv(mid_channels, channels, k=1, act=False)
        self.ratio = ratio

    def forward(self, x):
        b, c, _, _ = x.shape
        k = max(1, int(c * self.ratio))
        attn = self.avg_pool(x)
        attn = self.conv(attn)
        attn = self.out_conv(attn)
        attn = torch.sigmoid(attn)
        return x * attn.expand_as(x)


class GhostModule(nn.Module):
    """尺寸匹配的Ghost模块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        init_channels = max(8, out_channels // 2)
        self.primary_conv = Conv(
            in_channels, init_channels, k=kernel_size, s=stride,
            p=kernel_size // 2 + (stride - 1)  # 确保stride=2时尺寸正确
        )
        self.cheap_operation = Conv(init_channels, init_channels, k=3, s=1, p=1, g=init_channels)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class ELSPDConv(nn.Module):
    """尺寸严格匹配的ELSPDConv"""

    def __init__(self, c1, c2, k=3, s=2, ratio=0.25):
        super().__init__()
        self.stride = s
        self.scale = s
        spd_channels = c1 * s * s

        # 通道安全处理
        reduction = max(1, min(2, spd_channels // 128))
        reduced_channels = max(32, spd_channels // reduction)

        self.ghost = GhostModule(spd_channels, reduced_channels, k)
        self.conv_reduce = Conv(reduced_channels, c2, k=1)
        self.attention = LightweightAttention(c2, ratio)
        self.bn = nn.BatchNorm2d(c2)
        self.stride_tensor = torch.tensor([s])  # 显式设置stride

    def forward(self, x):
        batch, c, h, w = x.shape
        pad_h = (self.scale - h % self.scale) % self.scale
        pad_w = (self.scale - w % self.scale) % self.scale
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # SPD转换
        x = x.view(batch, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch, c * self.scale * self.scale, h // self.scale, w // self.scale)

        # 特征提取
        x = self.ghost(x)
        x = self.conv_reduce(x)
        x = self.attention(x)
        # 如果批次大小为1，使用评估模式
        if batch == 1:
            self.bn.eval()
        x = self.bn(x)

        # 尺寸验证（关键修复）
        target_h, target_w = h // self.scale, w // self.scale
        if x.shape[2:] != (target_h, target_w):
            x = torch.nn.functional.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return x
