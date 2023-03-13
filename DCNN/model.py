import torch
import torch.nn as nn
import sys
sys.path.append('./DCNN')
from transformer import PixelwiseViT
import torch.nn.functional as F

# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in discriminator
ndf = 128



class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, with_dropout=False):
        super(SingleConv, self).__init__()
        if with_dropout:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.Dropout(0.02),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.single_conv(x)

class DenseFeaureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DenseFeaureAggregation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=1 * in_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, base_ch, dilation=2, kernel_size=3, padding=2, stride=1, bias=True),

        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + base_ch, base_ch, dilation=3, kernel_size=3, padding=3, stride=1, bias=True),

        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 2 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 2 * base_ch, base_ch, dilation=5, kernel_size=3, padding=5, stride=1, bias=True),

        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 3 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 3 * base_ch, base_ch, dilation=7, kernel_size=3, padding=7, stride=1, bias=True),

        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 4 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 4 * base_ch, base_ch, dilation=9, kernel_size=3, padding=9, stride=1, bias=True),

        )

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch + 5 * base_ch, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch + 5 * base_ch, out_ch, dilation=1, kernel_size=1, padding=0, stride=1, bias=True),
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
    def __init__(self, in_ch, list_ch, bottleneck='DFA', with_dropout=False):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )

        if bottleneck=='DFA':
            self.DFA = DenseFeaureAggregation(list_ch[4], list_ch[4], list_ch[4])
        elif bottleneck=='Vit':
            self.DFA = PixelwiseViT(192, 6, 12, 1536, 192, 'leakyrelu', 'layer', (256, 16, 16), (256, 16, 16))
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
    def __init__(self, out_ch, list_ch, Unet=True, with_dropout=False, PTV_estimator=False):
        super(Decoder, self).__init__()
        self.connect = Unet
        self.PTV_estimator = PTV_estimator

        self.upconv_3_1 = nn.ConvTranspose2d(list_ch[4], list_ch[3], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        self.upconv_2_1 = nn.ConvTranspose2d(list_ch[3], list_ch[2], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        self.upconv_1_1 = nn.ConvTranspose2d(list_ch[2], list_ch[1], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1, with_dropout=with_dropout)
        )
        if self.PTV_estimator:
            self.conv_out = nn.Sequential(
                nn.Conv2d(list_ch[1], out_ch, kernel_size=1)
            )
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
            output = F.softmax(output, dim=1)  # Use softmax activation for multi-class segmentation
              
        return [output]

class Decoder_sep(nn.Module):
    def __init__(self, out_ch, list_ch):
        super(Decoder_sep, self).__init__()

        self.upconv_3_1 = nn.ConvTranspose2d(2*list_ch[4], 2*list_ch[3], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(4 * list_ch[3], 2*list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(2*list_ch[3], 2*list_ch[3], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_2_1 = nn.ConvTranspose2d(2*list_ch[3], 2*list_ch[2], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(4 * list_ch[2], 2*list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(2*list_ch[2], 2*list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.upconv_1_1 = nn.ConvTranspose2d(2*list_ch[2], 2*list_ch[1], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(4 * list_ch[1], 2*list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(2*list_ch[1], 2*list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(2*list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder
        
        out_decoder_3_1 = self.decoder_conv_3_1(
            torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2_1 = self.decoder_conv_2_1(
            torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1)
        )
        out_decoder_1_1 = self.decoder_conv_1_1(
            torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1)
        )
        
        output = self.conv_out(out_decoder_1_1)
        return [output]

class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch, 
                bottleneck='DFA', 
                Unet=True, 
                with_dropout=False, 
                PTV_estimator=False):
        super(Model, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, bottleneck, with_dropout=with_dropout)
        self.decoder = Decoder(out_ch, list_ch, Unet=Unet, PTV_estimator=PTV_estimator)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)  # is a list

        return out_decoder

class Model_sep(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch, bottleneck='DFA'):
        super(Model_sep, self).__init__()
        self.encoder_masks = Encoder(in_ch-1, list_ch, bottleneck)
        self.encoder_ct = Encoder(1, list_ch, bottleneck)

        self.decoder = Decoder_sep(out_ch, list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder_masks.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder_ct.modules)

    def forward(self, x):
        masks = x[:,:3]
        ct = x[:,3:]
        out_encoder_masks = self.encoder_masks(masks)
        out_encoder_ct = self.encoder_ct(ct)
        
        out_encoder=[]
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
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.freeze_models()
        in_ch = len(models)
        self.conv = nn.Sequential(
            SingleConv(in_ch, 1, kernel_size=1, stride=1, padding=0)
        )
    def freeze_models(self):
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False
            m=m.eval()
        
    def forward(self, input):
        models_outputs = []
        for m in self.models:
            models_outputs.append(m(input)[0])
        models_out_concat = torch.cat(models_outputs, axis=1)
        return torch.mean(models_out_concat, axis=1)



class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0


    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
