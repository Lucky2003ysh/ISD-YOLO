import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class LightP_SPD_Conv(nn.Module):
    """修复尺寸不匹配的轻量化P_SPD_Conv模块"""

    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.stride = s
        self.scale = s  # 仅支持s=2（YOLO标准下采样步长）
        self.k = k  # 卷积核大小（必须为奇数，确保填充对称）
        self.pad_size = (k - 1) // 2  # 计算对称填充基数，确保卷积后尺寸不变

        # 1. 统一两个分支的填充策略（确保卷积后尺寸一致）
        self.pad_h = nn.ZeroPad2d(padding=(self.pad_size, self.pad_size, 0, 0))  # 水平方向对称填充
        self.pad_v = nn.ZeroPad2d(padding=(0, 0, self.pad_size, self.pad_size))  # 垂直方向对称填充

        # 2. SPD转换后通道压缩（平滑特征过渡）
        spd_channels = c1 * self.scale * self.scale  # s=2时为c1*4
        self.reduce = Conv(spd_channels, c1 * 2, 1, s=1)  # 压缩至2*c1

        # 3. 多方向卷积（1×k水平，k×1垂直，确保输出尺寸一致）
        self.cv_cw = Conv(c1 * 2, c2 // 2, (1, k), s=1, p=0)  # 水平卷积（1×k）
        self.cv_ch = Conv(c1 * 2, c2 // 2, (k, 1), s=1, p=0)  # 垂直卷积（k×1）

        # 4. 融合与激活
        self.cat_conv = Conv(c2, c2, 1, s=1)
        self.act = Conv.default_act if act else nn.Identity()

    def forward(self, x):
        batch, c, h, w = x.shape

        # Step 1: SPD转换（确保空间尺寸可被scale整除）
        pad_h = (self.scale - h % self.scale) % self.scale
        pad_w = (self.scale - w % self.scale) % self.scale
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))  # 补0后尺寸：(h+pad_h, w+pad_w)
        h_padded, w_padded = x.shape[2], x.shape[3]

        # Step 2: 空间→通道转换（无信息损失）
        x = x.view(batch, c, h_padded // self.scale, self.scale, w_padded // self.scale, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch, c * self.scale ** 2, h_padded // self.scale,
                   w_padded // self.scale)  # 形状：(b, c*4, h_split, w_split)

        # Step 3: 通道压缩（减少维度跳变）
        x = self.reduce(x)  # 形状：(b, 2*c1, h_split, w_split)

        # Step 4: 多方向特征提取（关键修复：确保两分支尺寸一致）
        # 水平分支：1×k卷积 + 水平对称填充
        y1 = self.pad_h(x)  # 水平填充后宽度：w_split + 2*pad_size
        y1 = self.cv_cw(y1)  # 1×k卷积后宽度：(w_split + 2*pad_size) - k + 1 = w_split（因pad_size=(k-1)/2）
        # 垂直分支：k×1卷积 + 垂直对称填充
        y2 = self.pad_v(x)  # 垂直填充后高度：h_split + 2*pad_size
        y2 = self.cv_ch(y2)  # k×1卷积后高度：(h_split + 2*pad_size) - k + 1 = h_split（因pad_size=(k-1)/2）

        # 强制检查尺寸（确保拼接前一致）
        if y1.shape[2:] != y2.shape[2:]:
            # 若仍有差异，以y1尺寸为准进行插值
            y2 = nn.functional.interpolate(y2, size=y1.shape[2:], mode="bilinear", align_corners=False)

        # Step 5: 特征拼接与融合
        x = torch.cat([y1, y2], dim=1)  # 通道维度拼接（dim=1）
        x = self.cat_conv(x)
        x = self.act(x)

        # Step 6: 确保最终输出尺寸与原始Conv一致
        target_h, target_w = h // self.stride, w // self.stride
        if x.shape[2:] != (target_h, target_w):
            x = nn.functional.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x
