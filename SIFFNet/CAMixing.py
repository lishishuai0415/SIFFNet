import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import os
sys.path.append(os.getcwd())

# m_seed = 1
# # set seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Multi-Scale Feed-Forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*3, kernel_size=(1,1), bias=bias)

        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=(1,1), bias=bias)

    def forward(self, x):

        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1)
        x2 = self.dwconv2(x2)
        x3 = self.dwconv3(x3)
        x = F.gelu(x1)*(x2+x3)
        x = self.project_out(x)
        return x



##########################################################################
## Convolution and Attention Fusion Module  (CAFM)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv         = nn.Conv2d(dim,   dim*3, kernel_size=(1,1), bias=bias)
        self.qkv_dwconv  = nn.Conv2d(dim*3, dim*3, kernel_size=(3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim,   dim,   kernel_size=(1,1), bias=bias)
        self.fc          = nn.Conv2d(3*self.num_heads, 9, kernel_size=(1,1), bias=True)
        self.dep_conv    = nn.Conv2d(9*dim//self.num_heads, dim, kernel_size=(3,3), bias=True, groups=dim//self.num_heads, padding=1)
        self.pa = PA(dim)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        f_conv = qkv.permute(0,2,3,1)

        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3)

        f_all = self.fc(f_all)


        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)

        out_conv = self.dep_conv(f_conv) # B, C, H, W

        #PA
        out = self.pa(x)
        #out
        output = out + out_conv

        return output

class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        out = x * y
        return out



##########################################################################
## CAMixing Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=31, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(3,3), stride=1, padding=1, bias=bias)

    def forward(self, x):
        #x = x.unsqueeze(2)
        x = self.proj(x)
        #x = x.squeeze(2)
        return x



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        # x = x.unsqueeze(2)
        x = self.body(x)
        # x = x.squeeze(2)
        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        # x = x.unsqueeze(2)
        x = self.body(x)
        # x = x.squeeze(2)
        return x

##########################################################################
##---------- HCANet -----------------------
class HCANet(nn.Module):
    def __init__(self, 
        inp_channels=31, 
        out_channels=31, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 1,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(HCANet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) 
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) 
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) 
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=(1,1,1), bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
            
        self.output = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=(3,3,3), stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)


        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = out_enc_level3
        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2.unsqueeze(2))
        inp_dec_level2 = inp_dec_level2.squeeze(2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        
        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1.unsqueeze(2)).squeeze(2) + inp_img



        return out_dec_level1

if __name__ == "__main__":
    model = HCANet()
    # print(model)
    # summary(model, (1,31,128,128))
    inputs = torch.ones([2,31,128,128]) #[b,c,h,w]
    outputs = model(inputs)
    print(outputs.size())
