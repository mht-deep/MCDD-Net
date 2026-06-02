import math
import torch
import torch.nn as nn
from network.resnet import resnet50  # Fix import to match package path when running from project root.
import torch.nn.functional as F

from utils.addWays import DFEBlock, MSCA


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = f_fg-f_bg
        return f_fg, f_bg, f_uc
"""Auxiliary Head"""


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


class FBDCA(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_diff = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)

        diff_feature = self.compute_differential_feature(fg, bg, B, H, W)
        attn_diff = self.compute_differential_attention(diff_feature, B, H, W, C)
        x_weighted = self.apply_attention(attn_diff, v_unfolded, B, H, W, C)
        x_weighted = x_weighted.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted)
        return out

    def compute_differential_feature(self, fg, bg, B, H, W):
        fg_pooled = self.pool(fg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        bg_pooled = self.pool(bg.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        eps = 1e-6
        diff_abs = torch.abs(fg_pooled - bg_pooled)
        diff_norm = diff_abs / (torch.norm(diff_abs, dim=-1, keepdim=True) + eps)
        return diff_norm

    def compute_differential_attention(self, diff_feature, B, H, W, C):
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        attn = self.attn_diff(diff_feature).reshape(
            B, h * w, self.num_heads,
               self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size
        ).permute(0, 2, 1, 3, 4)  # [B, num_heads, h*w, k*k, k*k]
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
        self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.up_2x2(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x

class FBDCAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x



class MCDD(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]

        self.enhancer1 = MSCA(64)
        self.enhancer2 = MSCA(256)

        """ FEM """
        self.dconv1 = CBR(64, 128)
        self.dconv2 = CBR(256, 128)
        self.dconv3 = CBR(512, 128)
        self.dconv4 = CBR(1024, 128)

        self.dfe1 = DFEBlock(128,128)
        self.dfe2 = DFEBlock(128, 128)
        self.dfe3 = DFEBlock(128, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = FBDCAPreprocess(128, 128, 1)  # 1/16
        self.preprocess_bg4 = FBDCAPreprocess(128, 128, 1)  # 1/16

        self.preprocess_fg3 = FBDCAPreprocess(128, 128, 2)  # 1/8
        self.preprocess_bg3 = FBDCAPreprocess(128, 128, 2)  # 1/8

        self.preprocess_fg2 = FBDCAPreprocess(128, 128,4)  # 1/4
        self.preprocess_bg2 = FBDCAPreprocess(128, 128, 4)  # 1/4

        self.preprocess_fg1 = FBDCAPreprocess(128, 128,8)  # 1/2
        self.preprocess_bg1 = FBDCAPreprocess(128, 128, 8)  # 1/2

        """ Auxiliary Head """
        self.aux_head = AuxiliaryHead(128)

        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cdfa4 = FBDCA(128, 128, 4)
        self.cdfa3 = FBDCA(128 + 128, 128, 4)
        self.cdfa2 = FBDCA(128 + 128, 128, 4)
        self.cdfa1 = FBDCA(128 + 128, 128, 4)

        """ Output Block """
        self.output_block = output_block(512, 1)
        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x1 = self.enhancer1(x1)
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x2 = self.enhancer2(x2)
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        """ Dilated Conv """
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x2)
        d3 = self.dconv3(x3)
        d4 = self.dconv4(x4)

        d3 = self.dfe3(self.up_2x2(d4),d3)
        d2 = self.dfe2(self.up_2x2(d3), d2)
        d1 = self.dfe1(self.up_2x2(d2), d1)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x4)
        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)
        f4 = self.cdfa4(d4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([d3, f4_up], dim=1)
        f3 = self.cdfa3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([d2, f3_up], dim=1)
        f2 = self.cdfa2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([d1, f2_up], dim=1)
        f1 = self.cdfa1(f_2_1, f_fg1, f_bg1)
        final = torch.cat([f1, self.up_2x2(f2), self.up_4x4(f3), self.up_8x8(f4)], dim=1)
        """ Output Block """
        mask = self.output_block(final)

        return mask, mask_fg, mask_bg, mask_uc