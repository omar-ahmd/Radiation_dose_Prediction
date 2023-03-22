import torch
import torch.nn as nn
import sys

sys.path.append("./DCNN")
from transformer import PixelwiseViT
import torch.nn.functional as F
import numpy as np


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, with_dropout=False):
        super(SingleConv, self).__init__()
        if with_dropout:
            self.single_conv = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.Dropout(0.02),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.single_conv(x)


class DenseFeaureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DenseFeaureAggregation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=1 * in_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch,
                base_ch,
                dilation=2,
                kernel_size=3,
                padding=2,
                stride=1,
                bias=True,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch + base_ch,
                base_ch,
                dilation=3,
                kernel_size=3,
                padding=3,
                stride=1,
                bias=True,
            ),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 2 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch + 2 * base_ch,
                base_ch,
                dilation=5,
                kernel_size=3,
                padding=5,
                stride=1,
                bias=True,
            ),
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 3 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch + 3 * base_ch,
                base_ch,
                dilation=7,
                kernel_size=3,
                padding=7,
                stride=1,
                bias=True,
            ),
        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 4 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch + 4 * base_ch,
                base_ch,
                dilation=9,
                kernel_size=3,
                padding=9,
                stride=1,
                bias=True,
            ),
        )

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 5 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_ch + 5 * base_ch,
                out_ch,
                dilation=1,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
            ),
        )

    def forward(self, x):
        out_ = self.conv1(x)
        concat_ = torch.cat((out_, x), dim=1)
        out_ = self.conv2(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv3(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv4(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv5(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv_out(concat_)
        return out_


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch, bottleneck="DFA", with_dropout=False):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[1],
                list_ch[1],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[2],
                list_ch[2],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[3],
                list_ch[3],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[4],
                list_ch[4],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )

        if bottleneck == "DFA":
            self.DFA = DenseFeaureAggregation(list_ch[4], list_ch[4], list_ch[4])
        elif bottleneck == "Vit":
            self.DFA = PixelwiseViT(
                192,
                6,
                12,
                1536,
                192,
                "leakyrelu",
                "layer",
                (256, 16, 16),
                (256, 16, 16),
            )
        else:
            self.DFA = None

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        if not self.DFA is None:
            out_encoder_4 = self.DFA(out_encoder_4)

        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]


class Decoder(nn.Module):
    def __init__(
        self, out_ch, list_ch, Unet=True, with_dropout=False, PTV_estimator=False
    ):
        super(Decoder, self).__init__()
        self.connect = Unet
        self.PTV_estimator = PTV_estimator

        self.upconv_3_1 = nn.ConvTranspose2d(
            list_ch[4], list_ch[3], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[3],
                list_ch[3],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        self.upconv_2_1 = nn.ConvTranspose2d(
            list_ch[3], list_ch[2], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[2],
                list_ch[2],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        self.upconv_1_1 = nn.ConvTranspose2d(
            list_ch[2], list_ch[1], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(
                list_ch[1],
                list_ch[1],
                kernel_size=3,
                stride=1,
                padding=1,
                with_dropout=with_dropout,
            ),
        )
        if self.PTV_estimator:
            self.conv_out = nn.Sequential(nn.Conv2d(list_ch[1], out_ch, kernel_size=1))
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
            )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder

        if self.connect:
            out_decoder_3_1 = self.decoder_conv_3_1(
                torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1)
            )
            out_decoder_2_1 = self.decoder_conv_2_1(
                torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1)
            )
            out_decoder_1_1 = self.decoder_conv_1_1(
                torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1)
            )
        else:
            out_decoder_3_1 = self.upconv_3_1(out_encoder_4)

            out_decoder_2_1 = self.upconv_2_1(out_decoder_3_1)

            out_decoder_1_1 = self.upconv_1_1(out_decoder_2_1)

        output = self.conv_out(out_decoder_1_1)

        if self.PTV_estimator:
            output = F.softmax(
                output, dim=1
            )  # Use softmax activation for multi-class segmentation

        return [output]


class Decoder_sep(nn.Module):
    def __init__(self, out_ch, list_ch, Unet=True):
        super(Decoder_sep, self).__init__()
        self.connect = Unet

        self.upconv_3_1 = nn.ConvTranspose2d(
            2 * list_ch[4], 2 * list_ch[3], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(
                4 * list_ch[3], 2 * list_ch[3], kernel_size=3, stride=1, padding=1
            ),
            SingleConv(
                2 * list_ch[3], 2 * list_ch[3], kernel_size=3, stride=1, padding=1
            ),
        )
        self.upconv_2_1 = nn.ConvTranspose2d(
            2 * list_ch[3], 2 * list_ch[2], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(
                4 * list_ch[2], 2 * list_ch[2], kernel_size=3, stride=1, padding=1
            ),
            SingleConv(
                2 * list_ch[2], 2 * list_ch[2], kernel_size=3, stride=1, padding=1
            ),
        )
        self.upconv_1_1 = nn.ConvTranspose2d(
            2 * list_ch[2], 2 * list_ch[1], kernel_size=2, stride=2, bias=True
        )
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(
                4 * list_ch[1], 2 * list_ch[1], kernel_size=3, stride=1, padding=1
            ),
            SingleConv(
                2 * list_ch[1], 2 * list_ch[1], kernel_size=3, stride=1, padding=1
            ),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, out_encoder):

        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder
        if self.connect:
            out_decoder_3_1 = self.decoder_conv_3_1(
                torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1)
            )
            out_decoder_2_1 = self.decoder_conv_2_1(
                torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1)
            )
            out_decoder_1_1 = self.decoder_conv_1_1(
                torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1)
            )
        else:
            out_decoder_3_1 = self.upconv_3_1(out_encoder_4)

            out_decoder_2_1 = self.upconv_2_1(out_decoder_3_1)

            out_decoder_1_1 = self.upconv_1_1(out_decoder_2_1)

        output = self.conv_out(out_decoder_1_1)
        return [output]


class Model(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        list_ch,
        bottleneck="DFA",
        Unet=True,
        with_dropout=False,
        PTV_estimator=False,
    ):
        super(Model, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, bottleneck, with_dropout=with_dropout)
        self.decoder = Decoder(out_ch, list_ch, Unet=Unet, PTV_estimator=PTV_estimator)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def initialize(self):
        print("# random init encoder weight using nn.init.kaiming_uniform !")
        self.init_conv_deconv_BN(self.decoder.modules)
        print("# random init decoder weight using nn.init.kaiming_uniform !")
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder


class Model_sep(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch, bottleneck="DFA", Unet=True):
        super(Model_sep, self).__init__()
        self.connect = Unet
        self.encoder_masks = Encoder(in_ch - 1, list_ch, bottleneck)
        self.encoder_ct = Encoder(1, list_ch, bottleneck)

        self.decoder = Decoder_sep(out_ch, list_ch, Unet=self.connect)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def initialize(self):
        print("# random init encoder weight using nn.init.kaiming_uniform !")
        self.init_conv_deconv_BN(self.decoder.modules)
        print("# random init decoder weight using nn.init.kaiming_uniform !")
        self.init_conv_deconv_BN(self.encoder_masks.modules)
        print("# random init decoder weight using nn.init.kaiming_uniform !")
        self.init_conv_deconv_BN(self.encoder_ct.modules)

    def forward(self, x):
        masks = x[:, :3]
        ct = x[:, 3:]
        out_encoder_masks = self.encoder_masks(masks)
        out_encoder_ct = self.encoder_ct(ct)

        out_encoder = []
        for i in range(len(out_encoder_ct)):
            out_enc = torch.cat((out_encoder_masks[i], out_encoder_ct[i]), dim=1)
            out_encoder.append(out_enc)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.freeze_models()

    def freeze_models(self):
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False
            m = m.eval()

    def forward(self, input):
        models_outputs = []
        for m in self.models:
            try:
                outp = m(input)[0]
            except:
                outp = m(input[:, :-1])[0]

            models_outputs.append(outp)
        models_out_concat = torch.cat(models_outputs, axis=1)
        return [models_out_concat.mean(axis=1)[:,None]]
