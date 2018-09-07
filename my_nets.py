import torch
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn

from net_utils import DownMatchingCat, BasicBlock, DecoderBlockV0, DecoderBlockV1, BlockV0, \
    CenterBlockV1, FinalBlockV1, depth_to_space, DecoderBlockGatedV1


class MyResNetV0(nn.Module):

    def __init__(self, in_size=1, num_classes=1):

        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        encoder = tv.models.resnet18(pretrained=True)

        self.incoming = nn.Conv2d(in_size, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        try:
            self.final_size = self.layer4[-1].bn3.num_features
        except:
            self.final_size = self.layer4[-1].bn2.num_features

        self.mid_layer = BasicBlock(
            self.final_size, self.final_size, downsample=nn.Sequential(
                nn.Conv2d(self.final_size, 2 * self.final_size, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(2 * self.final_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ))

        self.mid_dec = DecoderBlockV0(2 * self.final_size, self.final_size)
        self.dec4 = DecoderBlockV0(2 * self.final_size, self.final_size // 2)
        self.dec3 = DecoderBlockV0(self.final_size, self.final_size // 4)
        self.dec2 = DecoderBlockV0(self.final_size // 2, self.final_size // 8)
        self.dec1 = DecoderBlockV0(self.final_size // 4, self.final_size // 8)
        self.dec0 = DecoderBlockV0(self.final_size // 4, self.final_size // 32)
        self.cat_in = DownMatchingCat()
        self.final = nn.Conv2d(3 + self.final_size // 32, 1, kernel_size=1)

    def forward(self, x):
        out = x
        out_in = self.incoming(out)     # -> (3, 101, 101)
        out_0 = self.layer0(out_in)     # -> (64, 51, 51)
        out_1 = self.layer1(out_0)      # -> (64, 26, 26)
        out_2 = self.layer2(out_1)      # -> (128, 13, 13)
        out_3 = self.layer3(out_2)      # -> (256, 7, 7)
        out_4 = self.layer4(out_3)      # -> (512, 4, 4)

        out = self.mid_layer(out_4)     # -> (1024, 2, 2)

        out = self.mid_dec(out)         # -> (512, 4, 4)
        out = self.dec4(out, out_4)     # -> (256, 8, 8)
        out = self.dec3(out, out_3)     # -> (128, 14, 14)
        out = self.dec2(out, out_2)     # -> (64, 26, 26)
        out = self.dec1(out, out_1)     # -> (64, 52, 52)
        out = self.dec0(out, out_0)     # -> (16, 102, 102)
        out = self.cat_in(out, out_in)  # -> (19, 101, 101)
        out = self.final(out)
        return out


class MyNetV0(nn.Module):
    """Didn't work well, pre-trained nets are better"""
    def __init__(self, in_size=1, num_classes=1, num_filters=4, blocks_per_layer=1):
        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.layer0 = nn.Sequential(
            BlockV0(in_size, num_filters),
            *[BlockV0(num_filters, num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            BlockV0(num_filters, 2 * num_filters),
            *[BlockV0(2 * num_filters, 2 * num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            BlockV0(2 * num_filters, 4 * num_filters),
            *[BlockV0(4 * num_filters, 4 * num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            BlockV0(4 * num_filters, 8 * num_filters),
            *[BlockV0(8 * num_filters, 8 * num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            BlockV0(8 * num_filters, 16 * num_filters),
            *[BlockV0(16 * num_filters, 16 * num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.center = nn.Sequential(
            BlockV0(16 * num_filters, 32 * num_filters),
            *[BlockV0(32 * num_filters, 32 * num_filters) for _ in range(1, blocks_per_layer)],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.center_dec = DecoderBlockV0(32 * num_filters, 16 * num_filters)
        self.dec4 = DecoderBlockV0(32 * num_filters, 8 * num_filters)
        self.dec3 = DecoderBlockV0(16 * num_filters, 4 * num_filters)
        self.dec2 = DecoderBlockV0(8 * num_filters, 2 * num_filters)
        self.dec1 = DecoderBlockV0(4 * num_filters, 1 * num_filters)
        self.dec0 = DecoderBlockV0(2 * num_filters, 1 * num_filters)
        self.final_concat = DownMatchingCat()
        self.final_conv = nn.Conv2d(in_size + num_filters, 1, kernel_size=1)

    def forward(self, x):
        out = x
        out_0 = self.layer0(out)  # -> (nf, 51, 51)
        out_1 = self.layer1(out_0)  # -> (2*nf, 26, 26)
        out_2 = self.layer2(out_1)  # -> (4*nf, 13, 13)
        out_3 = self.layer3(out_2)  # -> (8*nf, 7, 7)
        out_4 = self.layer4(out_3)  # -> (16*nf, 4, 4)

        out = self.center(out_4)  # -> (32*nf, 2, 2)

        out = self.center_dec(out)  # -> (16*nf, 4, 4)
        out = self.dec4(out, out_4)  # -> (8*nf, 8, 8)
        out = self.dec3(out, out_3)  # -> (4*nf, 14, 14)
        out = self.dec2(out, out_2)  # -> (2*nf, 26, 26)
        out = self.dec1(out, out_1)  # -> (nf, 52, 52)
        out = self.dec0(out, out_0)  # -> (nf, 102, 102)
        out = self.final_concat(out, x)  # -> (nf+1, 101, 101)
        out = self.final_conv(out)
        out = out
        return out


class MyResNetV1(nn.Module):
    """
    Add options to
        - select which ResNet to use for encoder
        - which decoder block
        - what is the base number of channels to use for encoder, independent from the encoder size
        - add or not another transformation between encoder and decoder
    """
    encoder_sizes = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, in_size=1, num_classes=1, num_filters=16,
                 encoder_net='resnet18', decoder_version=0, mid_layer=True):

        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.mid_layer = mid_layer
        encoder = getattr(tv.models, encoder_net)(pretrained=True)
        layer_sizes = self.encoder_sizes[encoder_net]
        if decoder_version == 0:
            decoder_block = DecoderBlockV0
        elif decoder_version == 1:
            decoder_block = DecoderBlockV1

        self.incoming = nn.Conv2d(in_size, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if mid_layer:
            self.mid_layer = BasicBlock(
                layer_sizes[4], layer_sizes[4], downsample=nn.Sequential(
                    nn.Conv2d(layer_sizes[4], 2 * layer_sizes[4],
                              kernel_size=(1, 1), stride=(2, 2), bias=False),
                    nn.BatchNorm2d(2 * layer_sizes[4], eps=1e-05,
                                   momentum=0.1, affine=True, track_running_stats=True)
                ))

            self.mid_dec = decoder_block(2 * layer_sizes[4], 32 * num_filters)
            self.dec4 = decoder_block(layer_sizes[4] + 32 * num_filters, 16 * num_filters)
        else:
            self.dec4 = decoder_block(layer_sizes[4], 16 * num_filters)

        self.dec3 = decoder_block(layer_sizes[3] + 16 * num_filters, 8 * num_filters)
        self.dec2 = decoder_block(layer_sizes[2] + 8 * num_filters, 4 * num_filters)
        self.dec1 = decoder_block(layer_sizes[1] + 4 * num_filters, 2 * num_filters)
        self.dec0 = decoder_block(layer_sizes[0] + 2 * num_filters, 1 * num_filters)
        self.cat_in = DownMatchingCat()
        self.final = nn.Conv2d(3 + num_filters, 1, kernel_size=1)

    def forward(self, x):
        out = x
        out_in = self.incoming(out)  # -> (3, 101, 101)
        out_0 = self.layer0(out_in)  # -> (64, 51, 51)
        out_1 = self.layer1(out_0)  # -> (64, 26, 26)
        out_2 = self.layer2(out_1)  # -> (128, 13, 13)
        out_3 = self.layer3(out_2)  # -> (256, 7, 7)
        out_4 = self.layer4(out_3)  # -> (512, 4, 4)

        if self.mid_layer:
            out = self.mid_layer(out_4)  # -> (1024, 2, 2)
            out = self.mid_dec(out)  # -> (512, 4, 4)
            out = self.dec4(out, out_4)  # -> (256, 8, 8)
        else:
            out = self.dec4(out_4)  # -> (256, 8, 8)

        out = self.dec3(out, out_3)  # -> (128, 14, 14)
        out = self.dec2(out, out_2)  # -> (64, 26, 26)
        out = self.dec1(out, out_1)  # -> (64, 52, 52)
        out = self.dec0(out, out_0)  # -> (16, 102, 102)
        out = self.cat_in(out, out_in)  # -> (19, 101, 101)
        out = self.final(out)
        return out


class MyResNetV2(nn.Module):
    """
    - Add option to use lstm to go from 1 to 3 channels at the beginning
    - Add an option to freeze encoder weights
    """
    encoder_sizes = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, in_size=1, num_classes=1, num_filters=16, incoming='cnn', freeze_encoder=False,
                 encoder_net='resnet18', decoder_version=0, mid_layer=True):

        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.mid_layer = mid_layer
        encoder = getattr(tv.models, encoder_net)(pretrained=True)

        self.ignore_params = list(encoder.parameters()) if freeze_encoder else []

        layer_sizes = self.encoder_sizes[encoder_net]
        if decoder_version == 0:
            decoder_block = DecoderBlockV0
        elif decoder_version == 1:
            decoder_block = DecoderBlockV1

        self.incoming = incoming
        if incoming == 'cnn':
            self.incoming_layer = nn.Conv2d(in_size, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif incoming == 'lstm':
            self.incoming_layer = nn.LSTM(101, 101, bidirectional=True, batch_first=True)
            self.h0 = torch.zeros(2, 128, 101)
            self.c0 = torch.zeros(2, 128, 101)

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if mid_layer:
            self.mid_layer = BasicBlock(
                layer_sizes[4], layer_sizes[4], downsample=nn.Sequential(
                    nn.Conv2d(layer_sizes[4], 2 * layer_sizes[4],
                              kernel_size=(1, 1), stride=(2, 2), bias=False),
                    nn.BatchNorm2d(2 * layer_sizes[4], eps=1e-05,
                                   momentum=0.1, affine=True, track_running_stats=True)
                ))

            self.mid_dec = decoder_block(2 * layer_sizes[4], 32 * num_filters)
            self.dec4 = decoder_block(layer_sizes[4] + 32 * num_filters, 16 * num_filters)
        else:
            self.dec4 = decoder_block(layer_sizes[4], 16 * num_filters)

        self.dec3 = decoder_block(layer_sizes[3] + 16 * num_filters, 8 * num_filters)
        self.dec2 = decoder_block(layer_sizes[2] + 8 * num_filters, 4 * num_filters)
        self.dec1 = decoder_block(layer_sizes[1] + 4 * num_filters, 2 * num_filters)
        self.dec0 = decoder_block(layer_sizes[0] + 2 * num_filters, 1 * num_filters)
        self.cat_in = DownMatchingCat()
        self.final = nn.Conv2d(3 + num_filters, 1, kernel_size=1)

    def forward(self, x):
        out = x

        if self.incoming == 'cnn':
            out_in = self.incoming_layer(out)  # -> (3, 101, 101)
        elif self.incoming == 'lstm':
            out_in = out.squeeze(1)
            h0 = torch.zeros(2, x.size(0), x.size(-1)).to(x.device)
            c0 = torch.zeros(2, x.size(0), x.size(-1)).to(x.device)
            out_in, _ = self.incoming_layer(out_in, (h0, c0))
            out_in = out_in.view(x.size(0), x.size(2), x.size(3), 2).permute(0, 3, 1, 2) # -> [bs, 2, 101, 101]

            # raise Exception("{}  {}".format(out.size(), out_in.size()))
            out_in = torch.cat([out, out_in], dim=1)

        out_0 = self.layer0(out_in)  # -> (64, 51, 51)
        out_1 = self.layer1(out_0)  # -> (64, 26, 26)
        out_2 = self.layer2(out_1)  # -> (128, 13, 13)
        out_3 = self.layer3(out_2)  # -> (256, 7, 7)
        out_4 = self.layer4(out_3)  # -> (512, 4, 4)

        if self.mid_layer:
            out = self.mid_layer(out_4)  # -> (1024, 2, 2)
            out = self.mid_dec(out)  # -> (512, 4, 4)
            out = self.dec4(out, out_4)  # -> (256, 8, 8)
        else:
            out = self.dec4(out_4)  # -> (256, 8, 8)

        out = self.dec3(out, out_3)  # -> (128, 14, 14)
        out = self.dec2(out, out_2)  # -> (64, 26, 26)
        out = self.dec1(out, out_1)  # -> (64, 52, 52)
        out = self.dec0(out, out_0)  # -> (16, 102, 102)
        out = self.cat_in(out, out_in)  # -> (19, 101, 101)
        out = self.final(out)
        return out

    def pararemters(self):
        return (p for p in super().parameters() if p not in self.ignore_params)


class MyResNetV3(nn.Module):
    """
    - Change decoder block to new one with an option to use channel and spatial gates (or SE - squeeze and excitation)
    - remove encoder freezing because that does not give good results
    - instead have an option to get encoder parameters separately from
        the rest (can use different optimizer with different lr)
    - change center and final blocks

    """
    encoder_sizes = {
        'resnet18': [64, 64, 128, 256, 512],
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }

    def __init__(self, in_size=1, num_classes=1, num_filters=16, incoming='cnn',
                 encoder_net='resnet18', p_drop=0.2, add_gates=False):

        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.add_gates = add_gates
        encoder = getattr(tv.models, encoder_net)(pretrained=True)

        decoder_block = DecoderBlockGatedV1 if add_gates else DecoderBlockV1

        layer_sizes = self.encoder_sizes[encoder_net]

        self.incoming = incoming
        if incoming == 'cnn':
            self.incoming_layer = nn.Conv2d(in_size, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif incoming == 'lstm':
            self.incoming_layer = nn.LSTM(101, 101, bidirectional=True, batch_first=True)
            self.h0 = torch.zeros(2, 128, 101)
            self.c0 = torch.zeros(2, 128, 101)

        # start with size 101 x 101
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)   # 51
        self.layer1 = encoder.layer1                                            # 51
        self.layer2 = encoder.layer2                                            # 26
        self.layer3 = encoder.layer3                                            # 13
        self.layer4 = encoder.layer4                                            # 7

        self.center = CenterBlockV1(layer_sizes[4], 16 * num_filters)           # 3

        self.dec4 = decoder_block(layer_sizes[4] + 16 * num_filters, 8 * num_filters, upsample_scale=2)    # -> 6
        self.dec3 = decoder_block(layer_sizes[3] + 8 * num_filters, 4 * num_filters, upsample_scale=2)     # -> 12
        self.dec2 = decoder_block(layer_sizes[2] + 4 * num_filters, 2 * num_filters, upsample_scale=2)     # -> 24
        self.dec1 = decoder_block(
            layer_sizes[0] + layer_sizes[1] + 2 * num_filters, 1 * num_filters, upsample_scale=2)           # -> 48
        self.dec0 = decoder_block(3 + 1 * num_filters, 1 * num_filters, upsample_size=(101, 101))          # -> 101

        self.final = FinalBlockV1(7 * num_filters // 4, num_classes, p_drop=p_drop)

    def forward(self, x):
        if self.incoming == 'cnn':
            enc_in = self.incoming_layer(x)  # -> (3, 101, 101)
        elif self.incoming == 'lstm':
            enc_in = x.squeeze(1)
            h0 = torch.zeros(2, x.size(0), x.size(-1)).to(x.device)
            c0 = torch.zeros(2, x.size(0), x.size(-1)).to(x.device)
            enc_in, _ = self.incoming_layer(enc_in, (h0, c0))
            # -> [bs, 2, 101, 101]
            enc_in = enc_in.view(x.size(0), x.size(2), x.size(3), 2).permute(0, 3, 1, 2)
            # -> [bs, 3, 101, 101]
            enc_in = torch.cat([x, enc_in], dim=1)

        enc_0 = self.layer0(enc_in)
        enc_1 = self.layer1(enc_0)
        enc_2 = self.layer2(enc_1)
        enc_3 = self.layer3(enc_2)
        enc_4 = self.layer4(enc_3)

        centr = self.center(enc_4)

        dec_4 = self.dec4(centr, enc_4)                             #;print('dec_4', dec_4.size())
        dec_3 = self.dec3(dec_4, enc_3)                             #;print('dec_3', dec_3.size())
        dec_2 = self.dec2(dec_3, enc_2)                             #;print('dec_2', dec_2.size())
        dec_1 = self.dec1(dec_2, torch.cat([enc_0, enc_1], 1))      #;print('dec_1', dec_1.size())
        dec_0 = self.dec0(dec_1, enc_in)                            #;print('dec_0', dec_0.size())


        # cat on size 96 first
        # the output channels are:
        # num_filters / 4
        # num_filters / 8
        # num_filters / 4
        # num_filters / 8
        out = torch.cat([
            depth_to_space(dec_1, factor=2),
            depth_to_space(dec_2, factor=4),
            F.upsample(depth_to_space(dec_3, factor=4), scale_factor=2, mode='bilinear', align_corners=True),
            F.upsample(depth_to_space(dec_4, factor=8), scale_factor=2, mode='bilinear', align_corners=True),
        ], dim=1)                                                   #;print(out.size())

        # now cat to 101
        # channels total (1 + 3/4) * num_filters
        out = torch.cat([dec_0, F.pad(out, (2, 3, 2, 3))], dim=1)

        out = self.final(out)
        return out

    def encoder_pararemters(self):
        return (p
                for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
                for p in layer.parameters())

    def other_pararemters(self):
        return (p
                for layer in [self.incoming_layer, self.center, self.final,
                              self.dec0, self.dec1, self.dec2, self.dec3, self.dec4]
                for p in layer.parameters())

