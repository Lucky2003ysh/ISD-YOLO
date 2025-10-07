import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, c1, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(c1 * 2, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        pooled = torch.cat([avg_out, max_out], dim=1)
        att_weight = self.mlp(pooled).view(b, c, 1, 1)
        return att_weight


class SpatialAttention(nn.Module):

    def __init__(self, c1, groups=4, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c1 // groups, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1 // groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1 // groups, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class GLAM(nn.Module):
    """
    Global-Light Attention Mechanism (GLAM)
    """

    def __init__(self, c1, c2=None, reduction=16, groups=4, kernel_size=7):
        super().__init__()
        c2 = c1 if c2 is None else c2

        self.channel_att = ChannelAttention(c1, reduction)

        self.spatial_att = SpatialAttention(c1, groups, kernel_size)

        self.groups = groups

    def forward(self, x):
        channel_weight = self.channel_att(x)
        x_channel = x * channel_weight

        spatial_weight = self.spatial_att(x_channel)
        out = x_channel * spatial_weight

        if self.groups > 1:
            out = self._channel_shuffle(out, self.groups)



        return out

    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
