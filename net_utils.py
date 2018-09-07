import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpMatchingCat(nn.Module):
    """
    Pad to the largest of the two and concatenate along channels
    """
    def forward(self, x1, x2):
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        if diffX > 0:
            x2 = F.pad(x2, (diffX // 2, int(np.ceil(diffX / 2)), 0, 0))
        elif diffX < 0:
            diffX = abs(diffX)
            x1 = F.pad(x1, (diffX // 2, int(np.ceil(diffX / 2)), 0, 0))

        if diffY > 0:
            x2 = F.pad(x2, (0, 0, diffY // 2, int(np.ceil(diffY / 2))))
        elif diffY < 0:
            diffY = abs(diffY)
            x1 = F.pad(x1, (0, 0, diffY // 2, int(np.ceil(diffY / 2))))

        return torch.cat([x2, x1], dim=1)


class DownMatchingCat(nn.Module):
    """
    Crop to the smallest of the two and concatenate along channels
    """
    def forward(self, x1, x2):
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        if diffX > 0:
            x_slice = slice((diffX // 2), -int(np.ceil(diffX / 2)))
            x1 = x1[..., x_slice, :]
        elif diffX < 0:
            diffX = abs(diffX)
            x_slice = slice((diffX // 2), -int(np.ceil(diffX / 2)))
            x2 = x2[..., x_slice, :]

        if diffY > 0:
            y_slice = slice((diffY // 2), -int(np.ceil(diffY / 2)))
            x1 = x1[..., y_slice]
        elif diffY < 0:
            diffY = abs(diffY)
            y_slice = slice((diffY // 2), -int(np.ceil(diffY / 2)))
            x2 = x2[..., y_slice]

        return torch.cat([x2, x1], dim=1)


class BlockV0(nn.Module):
    """double convolution with pre-activation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x):
        return self.block(x)


class BlockV1(nn.Module):
    """Triple convolution with pre-activation and residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
        self.match_channels = None if in_channels == in_channels \
                                  else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

    def forward(self, x):
        if self.match_channels is None:
            return x + self.block(x)
        else:
            return self.match_channels(x) + self.block(x)


class DecoderBlockV0(nn.Module):
    """
    crop to the smallest of two inputs and concatenate
    one convolution and one deconvolution to 2x size
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.match_cat = DownMatchingCat()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2=None):
        x = x1 if x2 is None else self.match_cat(x1, x2)
        return self.block(x)


class DecoderBlockV1(nn.Module):
    """
    - Use bilinear upsample on the first input
    - Option to select which concatenation method to use with optional second input after upsample
    Followed by double convolution with batch norm and pre-activation
    """
    def __init__(self, in_channels, out_channels, upmatch=False, upsample_scale=None, upsample_size=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=upsample_scale, size=upsample_size, mode='bilinear', align_corners=True)
        self.match_cat = UpMatchingCat() if upmatch else DownMatchingCat()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            x = self.match_cat(x, x2)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class DecoderBlockGatedV1(nn.Module):
    """
    - Same as DecoderBlockV1, but with spatial and channel gates
    """
    def __init__(self, in_channels, out_channels, upmatch=False, upsample_scale=None, upsample_size=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=upsample_scale, size=upsample_size, mode='bilinear', align_corners=True)
        self.match_cat = UpMatchingCat() if upmatch else DownMatchingCat()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)

        self.spatial_gate = SpatialGate(out_channels)
        self.channel_gate = ChannelGate(out_channels, self._gate_reduction(out_channels))

    def _gate_reduction(self, channels):
        p = int(np.log2(channels) / 2)
        p = 2 if p < 2 else p
        return 2**p

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            x = self.match_cat(x, x2)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        sg = self.spatial_gate(x)
        cg = self.channel_gate(x)
        c = x * sg + x * cg
        return x


class BasicBlock(nn.Module):
    """
    Copy from pytorch
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        if self.downsample is not None:
            out = self.downsample(out)

        out = self.relu(out)

        return out


def space_to_depth(input, factor=2):
    """
    all pixels in each square of size (factor x factor), stacked as channels
    0, 1, 2, 3              (0, 1, 4, 5), (2, 3, 6, 7)
    4, 5, 6, 7      -->>    (8, 9, 12,13),(10,11,14,15)
    8, 9, 10,11
    12,13,14,15

    thus, number of channels is multiplied by factor^2, while
    width and height of an image is reduced by factor
    """
    in_size = input.size()
    assert (in_size[2] % factor == 0) and (in_size[3] % factor == 0)
    out_size = [in_size[0], factor * factor * in_size[1], in_size[2] // factor, in_size[3] // factor]

    out = input
    out = out.view(in_size[0], in_size[1], in_size[2], out_size[3], factor)
    out = out.permute(0, 1, 3, 4, 2)
    out = out.view((in_size[0], in_size[1], out_size[3], factor, out_size[2], factor))
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()
    out = out.view(in_size[0], out_size[3], out_size[2], -1)
    out = out.permute(0, 3, 2, 1)
    return out


def depth_to_space(input, factor=2):
    """
    the opposite of space_to_depth

    number of channels is reduced by factor^2, and width and height of an image is multiplied by factor
    """
    in_size = input.size()
    assert in_size[1] % (factor**2) == 0
    out_size = [in_size[0], in_size[1]//(factor**2), in_size[2] * factor, in_size[3] * factor]
    out = input
    out = out.permute(0, 3, 2, 1)
    out = out.view(in_size[0], in_size[3], in_size[2], factor, factor, out_size[1])
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
    out = out.view(in_size[0], out_size[1], in_size[3], factor, out_size[2])
    out = out.permute(0, 1, 4, 2, 3)
    out = out.view(in_size[0], out_size[1], out_size[2], out_size[3])
    return out


class ChannelGate(nn.Module):
    """
    https://arxiv.org/pdf/1709.01507.pdf
    Essentially calculates and applies a gate for each channel for the whole image
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = lambda x: x.view(x.size(0), x.size(1), -1).mean(dim=-1)

        self.ch_gate = nn.Sequential(nn.Linear(in_channels, in_channels // reduction),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                     nn.Linear(in_channels // reduction, in_channels),
                                     nn.Sigmoid())

    def forward(self, x):
        bs, ch, _, _ = x.size()
        ch_avg = self.avg_pool(x).view(bs, ch)
        ch_gate = self.ch_gate(ch_avg).view(bs, ch, 1, 1)
        return x * ch_gate


class SpatialGate(nn.Module):
    """
    calculate and apply gate per pixel using all channels for that pixel
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sp_gate = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                                     nn.Sigmoid())

    def forward(self, x):
        bs, ch, _, _ = x.size()
        sp_gate = self.sp_gate(x)
        return x * sp_gate


class CenterBlockV1(nn.Module):
    """
    just a double convolution with downsample by maxpool
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x


class FinalBlockV1(nn.Module):
    """
    Add drop layer and double convolution at the end
    """
    def __init__(self, in_channels, out_channels, p_drop=0.5):
        super().__init__()
        self.drop = nn.Dropout2d(p_drop)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.drop(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x