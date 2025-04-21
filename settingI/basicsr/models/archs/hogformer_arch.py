import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

#########################################################################

Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
#        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) #* self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
#        self.weight = nn.Parameter(torch.ones(normalized_shape))
#        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
###############################################################################
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

def inverse_channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )
    def forward(self, x):
        return x * self.scale
    
class FFN_DIFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.667, bias=False):
        super(FFN_DIFF, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.sigma = ElementScale(
            hidden_features//4, init_value=1e-5, requires_grad=True)
        self.decompose = nn.Conv2d(
            in_channels=hidden_features//4,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.decompose_act = nn.GELU()
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, 
                                stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=3, 
                                         stride=1, padding=2, groups=hidden_features//4, 
                                         bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x
    
    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x = channel_shuffle(x, groups=1) 
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.feat_decompose(x)
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x

##########################################################################
# import time
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_experts=3):
        super(TransformerBlock, self).__init__()
        self.attn_g_spatial = Attention_DHOGSA(dim, num_heads, bias)#dim, num_heads, num_experts=4, bias=True
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FFN_DIFF(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        f_spatial = self.attn_g_spatial(self.norm_g(x))
        x = x + f_spatial
        x_out = x + self.ffn(self.norm_ff1(x))
        return x_out

##########################################################################
class Attention_DHOGSA(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True, patch_size=8, clip_limit=1.0, n_bins=9):
        super(Attention_DHOGSA, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.bin_proj = Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)
        self.patch_size = patch_size
        self.n_bins = n_bins
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1))
        
    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    
    def unpad(self, x, t_pad):
        *_, hw = x.shape
        return x[:,:,t_pad[0]:hw-t_pad[1]]
    
    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit
    
    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5)
    
    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out
        
    def split_into_patches(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        n_h, n_w = (h + pad_h)//self.patch_size, (w + pad_w)//self.patch_size
        return patches, (b, c, h, w, pad_h, pad_w, n_h, n_w)
    
    def merge_patches(self, patches, shape_info):
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info
        patches = rearrange(patches, 'b (h w) c (p1 p2) -> b c (h p1) (w p2)', h=n_h, w=n_w, p1=self.patch_size, p2=self.patch_size)
        if pad_h > 0 or pad_w > 0:
            patches = patches[:, :, :h, :w]
        return patches
    
    def apply_hog_to_patch(self, x_half):
        b, c, h, w = x_half.shape
        gx = F.conv2d(x_half, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(x_half, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        orientation = torch.atan2(gy, gx)  # [-pi, pi]
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins
        patches_x, shape_info = self.split_into_patches(x_half)
        patches_mag, _ = self.split_into_patches(magnitude)
        patches_ori, _ = self.split_into_patches(orientation_bin.float())
        b, n_patches, c, patch_pixels = patches_x.shape
        sort_values = torch.zeros_like(patches_x)
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=x_half.device)
        for i in range(self.n_bins):
            bin_mask = (patches_ori == i).float()
            bin_magnitude = patches_mag * bin_mask
            sort_values += bin_magnitude * (i + 1)
            hog_features[..., i] = bin_magnitude.mean(dim=[-1,-2])
        
        hog_features = hog_features / (hog_features.sum(dim=-1, keepdim=True) + 1e-8)
        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)
        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)
        return x_half_processed, sort_indices, hog_features, shape_info
    
    def forward(self, x):
        b, c, h, w = x.shape
        half_c = c // 2
        x_half = x[:, :half_c]
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)
        b, n_patches, n_bins = hog_features.shape
        n_h = shape_info[-2] #int(math.sqrt(n_patches))
        n_w = shape_info[-1]
        hog_map = rearrange(hog_features, 'b (nh nw) bins -> b bins nh nw', nh=n_h, nw=n_w).contiguous()
        hog_map = self.bin_proj(hog_map)
        hog_map = F.interpolate(hog_map, size=(h, w), mode='bilinear')
        x = torch.cat((x_half_processed + hog_map, x[:, half_c:]), dim=1)
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x
        gx = F.conv2d(v, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(v, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6).view(b, c, -1)
        orientation = torch.atan2(gy, gx).view(b, c, -1)  # [-pi, pi]
        
        orientation_norm = ((orientation + torch.pi) / (2 * torch.pi))
        weighted_magnitude = magnitude * orientation_norm
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)
        idx = idx.unsqueeze(1).expand(b, c, -1)
        v = torch.gather(v.view(b, c, -1), dim=2, index=idx)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        
        out_replace = out[:,:half_c]
        patches_out, shape_info = self.split_into_patches(out_replace)
        patches_out = torch.scatter(patches_out, -1, idx_patch, patches_out)
        out_replace = self.merge_patches(patches_out, shape_info)
        out[:,:half_c] = out_replace
        return out

    
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super(SkipPatchEmbed, self).__init__()

        self.proj = nn.Sequential(
            nn.AvgPool2d( 2, stride=2, padding=0 , ceil_mode=False , count_include_pad=True , divisor_override=None ),
            Conv2d(in_c, dim, kernel_size=1, bias=bias),
            Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

    def forward(self, x, ):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
class HOGformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(HOGformer, self).__init__()
        num_experts = [2,3,3,4]
        self.num_blocks_total = sum(num_blocks)+num_refinement_blocks+num_blocks[2]+num_blocks[1]+num_blocks[0]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0]) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[1]) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[2]) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[3]) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[2]) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[1]) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0]) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_experts=num_experts[0]) for i in range(num_refinement_blocks)])

        self.skip_patch_embed1 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = Conv2d(int(dim*2**1)+3, int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = Conv2d(int(dim*2**2)+3, int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = Conv2d(int(dim*2**3)+3, int(dim*2**3), kernel_size=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, ):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # c,h,w

        inp_enc_level2 = self.down1_2(out_enc_level1) # 2c, h/2, w/2
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))

        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        ###########################
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img
        

