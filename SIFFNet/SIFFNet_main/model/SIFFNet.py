import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from CAMixing import TransformerBlock
from SIE import UUNet



## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, nc, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
##---------- SS Attention ----------
class SSBlockNaive(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.conv1_act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride*2, dilation=stride*2)
        self.conv2_act = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def _get_ff(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward(self, x):
        return x + self._get_ff(x)


class SSBlock(SSBlockNaive):
    def __init__(self, stride, in_ch, f_scale=2, ss_exp_factor=1.):
        super().__init__(stride, in_ch)
        self.embed_size = int(in_ch * ss_exp_factor)
        self.conv_layer = nn.Conv2d(in_channels=65, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.wqk = nn.Parameter(torch.zeros(size=(in_ch, self.embed_size)))
        self.wqk.requires_grad = True
        nn.init.xavier_uniform_(self.wqk.data, gain=1.414)

        self.stride = stride
        self.f_scale = f_scale



    def _pixel_unshuffle(self, x, c, f):
        x = rearrange(x, 'b c h w -> b 1 (c h) w')
        x = F.pixel_unshuffle(x, f)
        x = rearrange(x, 'b k (c h) w -> b (k c) h w', c=c)
        return x

    def _pixel_shuffle(self, x, c, f):
        x = rearrange(x, 'b (f c) h w -> b f (c h) w', f=f ** 2, c=c)
        x = F.pixel_shuffle(x, f)
        x = rearrange(x, 'b f (c h) w -> b (f c) h w', f=1, c=c)
        return x

    def _pad_for_shuffle(self, x, f):
        _, _, h, w = x.shape
        pad_h = 0
        pad_w = 0
        if h % f != 0:
            pad_h = f - h % f
            x = F.pad(x, (0, 0, 0, pad_h), mode='constant', value=0)
        if w % f != 0:
            pad_w = f - w % f
            x = F.pad(x, (0, pad_w, 0, 0), mode='constant', value=0)
        return x, pad_h, pad_w

    def _get_attention(self, x, f, unet_output):
        _, c, _, _ = x.shape
        xx = F.layer_norm(x, x.shape[-3:])  # layer normalization

        xx, ph, pw = self._pad_for_shuffle(xx, f)
        xx = self._pixel_unshuffle(xx, c, f)
        xx = rearrange(xx, 'b (f c) h w -> (b f) c h w', c=c, f=f ** 2)

        v, ph, pw = self._pad_for_shuffle(x, f)
        v = self._pixel_unshuffle(v, c, f)
        v = rearrange(v, 'b (f c) h w -> (b f) c h w', c=c, f=f ** 2)

        b, _, sh, sw = xx.shape

        # embed by wqk
        qk = rearrange(xx, 'b c h w -> (b h w) c')
        v = rearrange(v, 'b c h w -> (b h w) c')
        qk = torch.mm(qk, self.wqk)##矩阵乘法

        # process per image
        qk = rearrange(qk, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)
        v = rearrange(v, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)

        # get cosine similarity
        qk_norm = torch.linalg.norm(qk, dim=-1).unsqueeze(dim=-1) + 1e-8
        qk = qk / qk_norm

        attn = torch.bmm(qk, torch.transpose(qk, 1, 2))
        attn += 1.
        attn /= self.embed_size ** 0.5
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out =out.unsqueeze(0)
        out =out.reshape(unet_output.size(0),-1,unet_output.size(2),unet_output.size(3))
        out = torch.cat((out, unet_output), dim=1)
        out= self.conv_layer(out)

        if ph > 0:
            out = out[:, :, :-ph, :]
        if pw > 0:
            out = out[:, :, :, :-pw]

        return out

    def forward(  self , x ,  unet_output):
        f = self.f_scale * self.stride
        return x + self._get_ff(x + self._get_attention(x, f, unet_output))

class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size,  relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))


        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out





class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


class SIFFNet(nn.Module):

    def __init__(self, in_nc=1, out_nc=1, nc=64, bias=True):
        super(SIFFNet, self).__init__()
        kernel_size = 3
        reduction = 8

        self.unet = UUNet(num_classes=2)

        self.SSBlock = SSBlock(1, nc, f_scale=2, ss_exp_factor=1.)

        self.CA = CALayer(nc, reduction, bias=bias)

        self.conv_head2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.conv1 = UNetConvBlock(nc, nc, relu_slope=0.2)


        self.conv2 = UNetConvBlock(nc, nc, relu_slope=0.2)


        self.conv3 = UNetConvBlock(nc, nc, relu_slope=0.2)


        self.conv4 = nn.Sequential(TransformerBlock(dim=nc, num_heads=2, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias'))

        self.mp1 = nn.MaxPool2d(2)

        self.conv5 = UNetConvBlock(nc, nc,  relu_slope=0.2)


        self.conv6 = UNetConvBlock(nc, nc,  relu_slope=0.2)


        self.conv7 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.mp2 = nn.MaxPool2d(2)

        self.conv8 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv9 = UNetConvBlock(nc, nc,  relu_slope=0.2)


        # upsample

        self.conv10 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv11 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv12 = nn.Sequential(TransformerBlock(dim=nc, num_heads=2, ffn_expansion_factor=2.66, bias=bias, LayerNorm_type='WithBias'))


        # upsample

        self.conv13 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv14 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv15 = UNetConvBlock(nc, nc,  relu_slope=0.2)

        self.conv16 = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.m_dilatedconv1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.m_bn1 = nn.BatchNorm2d(nc)
        self.m_relu1 = nn.ReLU(inplace=True)

        self.m_dilatedconv2 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)
        self.m_bn2 = nn.BatchNorm2d(nc)
        self.m_relu2 = nn.ReLU(inplace=True)

        self.m_dilatedconv3 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.m_bn3 = nn.BatchNorm2d(nc)
        self.m_relu3 = nn.ReLU(inplace=True)

        self.m_dilatedconv4 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)
        self.m_bn4 = nn.BatchNorm2d(nc)
        self.m_relu4 = nn.ReLU(inplace=True)

        self.m_dilatedconv5 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=5, dilation=5, bias=bias)
        self.m_bn5 = nn.BatchNorm2d(nc)
        self.m_relu5 = nn.ReLU(inplace=True)

        self.m_dilatedconv6 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=6, dilation=6, bias=bias)
        self.m_bn6 = nn.BatchNorm2d(nc)
        self.m_relu6 = nn.ReLU(inplace=True)

        self.m_dilatedconv7 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=7, dilation=7, bias=bias)
        self.m_bn7 = nn.BatchNorm2d(nc)
        self.m_relu7 = nn.ReLU(inplace=True)

        self.m_dilatedconv8 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=8, dilation=8, bias=bias)
        self.m_bn8 = nn.BatchNorm2d(nc)
        self.m_relu8 = nn.ReLU(inplace=True)

        self.m_dilatedconv7_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=7, dilation=7, bias=bias)
        self.m_bn7_1 = nn.BatchNorm2d(nc)
        self.m_relu7_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv6_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=6, dilation=6, bias=bias)
        self.m_bn6_1 = nn.BatchNorm2d(nc)
        self.m_relu6_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv5_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=5, dilation=5, bias=bias)
        self.m_bn5_1 = nn.BatchNorm2d(nc)
        self.m_relu5_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv4_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)
        self.m_bn4_1 = nn.BatchNorm2d(nc)
        self.m_relu4_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv3_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.m_bn3_1 = nn.BatchNorm2d(nc)
        self.m_relu3_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv2_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)
        self.m_bn2_1 = nn.BatchNorm2d(nc)
        self.m_relu2_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv1_1 = nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.m_bn1_1 = nn.BatchNorm2d(nc)
        self.m_relu1_1 = nn.ReLU(inplace=True)

        self.m_dilatedconv = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)

        self.conv_tail = nn.Conv2d(2*out_nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.up = Up()

    def forward(self, x0):

        unet_output = self.unet(x0)

        x0_0 = self.conv_head(x0)

        x0_1 = self.SSBlock(x0_0, unet_output)

        ca_branch = self.CA(x0_1)

        x0_2 = self.conv_head2(ca_branch)

        x1 = self.conv1(x0_2)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x4_1 = self.mp1(x4)

        x5 = self.conv5(x4_1)

        x6 = self.conv6(x5)

        x7 = self.conv7(x6)

        x7_1 = self.mp2(x7)

        x8 = self.conv8(x7_1)

        x9 = self.conv9(x8)

        x9_1 = self.up(x9, x7)

        x10 = self.conv10(x9_1+x7)

        x11= self.conv11(x10)

        x12 = self.conv12(x11)

        x12_1 = self.up(x12, x4)

        x13 = self.conv13(x12_1+x4)

        x14 = self.conv14(x13)

        x15 = self.conv15(x14)

        x = self.conv16(x15)

        X = x + x0

        y1 = self.m_dilatedconv1(x0_2)
        y1_1 = self.m_bn1(y1)
        y1_1 = self.m_relu1(y1_1)

        y2 = self.m_dilatedconv2(y1_1)
        y2_1 = self.m_bn2(y2)
        y2_1 = self.m_relu2(y2_1)

        y3 = self.m_dilatedconv3(y2_1)
        y3_1 = self.m_bn3(y3)
        y3_1 = self.m_relu3(y3_1)

        y4 = self.m_dilatedconv4(y3_1)
        y4_1 = self.m_bn4(y4)
        y4_1 = self.m_relu4(y4_1)

        y5 = self.m_dilatedconv5(y4_1)
        y5_1 = self.m_bn5(y5)
        y5_1 = self.m_relu5(y5_1)

        y6 = self.m_dilatedconv6(y5_1)
        y6_1 = self.m_bn6(y6)
        y6_1 = self.m_relu6(y6_1)

        y7 = self.m_dilatedconv7(y6_1)
        y7_1 = self.m_bn7(y7)
        y7_1 = self.m_relu7(y7_1)

        y8 = self.m_dilatedconv8(y7_1)
        y8_1 = self.m_bn8(y8)
        y8_1 = self.m_relu8(y8_1)

        y9 = self.m_dilatedconv7_1(y8_1)
        y9 = self.m_bn7_1(y9)
        y9 = self.m_relu7_1(y9)

        y10 = self.m_dilatedconv6_1(y9+y7)
        y10 = self.m_bn6_1(y10)
        y10 = self.m_relu6_1(y10)

        y11 = self.m_dilatedconv5_1(y10+y6)
        y11 = self.m_bn5_1(y11)
        y11 = self.m_relu5_1(y11)

        y12 = self.m_dilatedconv4_1(y11+y5)
        y12 = self.m_bn4_1(y12)
        y12 = self.m_relu4_1(y12)

        y13 = self.m_dilatedconv3_1(y12+y4)
        y13 = self.m_bn3_1(y13)
        y13 = self.m_relu3_1(y13)

        y14 = self.m_dilatedconv2_1(y13+y3)
        y14 = self.m_bn2_1(y14)
        y14 = self.m_relu2_1(y14)

        y15 = self.m_dilatedconv1_1(y14+y2)
        y15 = self.m_bn1_1(y15)
        y15 = self.m_relu1_1(y15)

        y = self.m_dilatedconv(y15+y1)

        Y = y + x0

        z0 = torch.cat([X, Y], dim=1)
        z = self.conv_tail(z0)
        Z = z + x0
        return unet_output , Z

if __name__ == "__main__":
    model = SIFFNet()
    input_tensor = torch.randn(1, 1, 160 ,160)
    output_tensor = model(input_tensor)
    print(output_tensor[0].shape)