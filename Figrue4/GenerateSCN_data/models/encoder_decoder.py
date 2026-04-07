import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import numpy as np
import torch.nn.functional as F
import math



class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

 
class LDProjector(torch.nn.Module):
    def __init__(
        self,
        mices_dict, 
        in_dim=1,
        proj_neuron=1024, 
        base_channel=128,
        dropout=0.0,
    ):
        super().__init__()

        self.proj_neuron = proj_neuron
        self.neuron_expand = nn.Conv2d(in_channels=in_dim,
                                       out_channels=base_channel,
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=(0,1),
                                       bias=True
                                       )
        self.norm1 = nn.GroupNorm(num_groups=2, num_channels=base_channel, eps=1e-6, affine=True) 
        self.act = nn.LeakyReLU()
        #! enhance this refine module with ResNetBlock.
        # the first resblock
        self.resblock1 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # atten block
        self.atten = AttnBlock(base_channel)
        # the second resblock
        self.resblock2 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        self.norm_out = nn.GroupNorm(num_groups=base_channel, num_channels=base_channel, eps=1e-6, affine=True)  #nn.ModuleDict({key:nn.GroupNorm(num_groups=32, num_channels=base_channel, eps=1e-6, affine=True) for key in mices_dict.keys()})
        # self.key_id = {key : torch.tensor([id_+1]) for id_, key in enumerate(mices_dict.keys())}
            
    def forward(self, x):
        # id_ = self.key_id[mice].float().to(x.device)
        len_t = x.shape[-1]
        x = self.neuron_expand(x.unsqueeze(1))    # N 128 N t       
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(self.proj_neuron, len_t)) # N 128 1024 t       
        x = self.norm1(x)
        x = self.act(x)
        # the first resblock
        x = self.resblock1(x)
        # the attention block
        x = self.atten(x)
        # the second resblock
        x = self.resblock2(x)

        x = self.norm_out(x)
        x = self.act(x)
        return  x       # N 128 1024 t
         

class LearnableSoftplus(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x):
        xb = x * self.beta
        return (torch.clamp(xb, 0) + torch.minimum(xb, -xb).exp().log1p()) / self.beta


class Interpolate(nn.Module):
    def __init__(self,  num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
    
    def forward(self, x):
        len_t = x.shape[-1]
        return torch.nn.functional.interpolate(x, size=(self.num_neurons, len_t), mode="bicubic")



class invLDProjector(torch.nn.ModuleDict):

    def __init__(
        self,
        mices_dict, 
        in_dim=1, 
        base_channel=128,
        dropout=0.1
    ):
        super().__init__()
        self.interpolate = nn.ModuleDict({ key: Interpolate(num_neurons) for key, num_neurons in mices_dict.items()})
        # the first resblock
        self.resblock1 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # the second resblock
        self.resblock2 = ResnetBlock(base_channel,
                                    base_channel,
                                    conv_shortcut=False,
                                    dropout=dropout,
                                    )
        # final out
        self.process1 = nn.Conv2d(base_channel,
                                     base_channel,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=base_channel, eps=1e-6, affine=True) 
        self.act = nn.LeakyReLU()
        self.process2 = nn.Conv2d(base_channel,
                                     base_channel//2,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=base_channel//2, eps=1e-6, affine=True)
        self.out = nn.Conv2d(base_channel//2,
                                     in_dim,
                                     kernel_size=1)
        # self.out_act = LearnableSoftplus(1.0)

    def forward(self, x, mice):
        # id_ = self.key_id[mice].float().to(x.device)
        
        x = self.interpolate[mice](x)
        # the first resblock
        x = self.resblock1(x)
        # the second resblock
        x = self.resblock2(x)
        
        # final out
        x = self.process1(x) 
        x = self.norm1(x) 
        x = self.act(x) 
        x = self.process2(x) 
        x = self.norm2(x) 
        x = self.act(x) 
        return self.out(x).squeeze(1)#self.out_act(self.out(x))



def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels<=32:
        num_groups=in_channels
    else:
        num_groups=32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        behavior_t: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(BehaviorMLP, self).__init__()
        self.behavior_t = behavior_t
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim // 2, bias=use_bias),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_dim // 2, out_features=out_dim // self.behavior_t, bias=use_bias),
            nn.Tanh(),
        )
    
    def forward(self, inputs: torch.Tensor):
        b, t = inputs.shape[0], inputs.shape[1]
        return self.model(inputs).reshape(b, t//self.behavior_t, -1) # B C T


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, *, ch=64, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=128,
                 resolution=1024, z_channels=512, double_z=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block 
            down.attn = attn
            
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)                
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)               

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch=64, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=128,
                 resolution=1024, z_channels=512, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out, 
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   


class EMAQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.re_embed = n_embed

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened            
            self.embedding.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.sum((z_q.detach()-z)**2) + \
                   torch.sum((z_q - z.detach()) ** 2)
        else:
            loss = torch.sum((z_q.detach()-z)**2) + self.beta * \
                   torch.sum((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss
    
    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    

# class PreservedLatentDynamics(nn.Module):
#     def __init__(self, n_e=1024, code_dim=256, z_channels=4):
#         super().__init__()    
        
#         self.quant_conv = torch.nn.Conv2d(z_channels, code_dim, 1)
#         # self.quantize = EMAQuantizer(n_embed=n_e, 
#         #                              embedding_dim=code_dim, 
#         #                              beta=0.25,
#         #                              decay=0.99)
#         self.quantize = VectorQuantizer(n_e=n_e, 
#                                      e_dim=code_dim, 
#                                      beta=0.25)
#         self.post_quant_conv = torch.nn.Conv2d(code_dim, z_channels, 1)

#     def forward(self, feat_ca):
#         quant_ca = self.quant_conv(feat_ca)
#         quant, codebook_loss = self.quantize(quant_ca)
#         quant_ref = self.post_quant_conv(quant)
#         return quant_ref, codebook_loss


class AttnBlock_1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,n = q.shape
        q = q.permute(0,2,1)   # b,n,c
        # k = k.reshape(b,c,n) # b,c,n
        w_ = torch.bmm(q,k)     # b,n,n    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,n)
        w_ = w_.permute(0,2,1)   # b,n,n (first n of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,n (n of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,n)

        h_ = self.proj_out(h_)

        return x+h_


class ResnetBlock_1d(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        if in_channels>=32:
            num_groups=32
        else:
            num_groups=in_channels
        self.norm1 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True) 
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.leaky_relu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class Pos2Embedding(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_channel=16,
        output_channel=16,
        proj_neuron=128,
        dropout=0.1
    ):
        super().__init__()
        
        self.mapping = nn.Conv1d(input_channels, hidden_channel, kernel_size=1, bias=True)
        self.norm_act = nn.Sequential(
            torch.nn.GroupNorm(num_groups=hidden_channel, num_channels=hidden_channel, eps=1e-6, affine=True),
            nn.LeakyReLU()
        )
        self.feat_extract = nn.Sequential(
            ResnetBlock_1d(hidden_channel, out_channels=hidden_channel, dropout=dropout),
            AttnBlock_1d(hidden_channel),
            ResnetBlock_1d(hidden_channel, out_channels=hidden_channel, dropout=dropout),
        )
        self.out_mean = nn.Sequential(
            ResnetBlock_1d(hidden_channel, out_channels=hidden_channel),
            nn.Conv1d(hidden_channel, output_channel, kernel_size=1, bias=True)
        )
        self.out_logvar = nn.Sequential(
            ResnetBlock_1d(hidden_channel, out_channels=hidden_channel),
            nn.Conv1d(hidden_channel, output_channel, kernel_size=1, bias=True)
        )
        self.proj_neuron=proj_neuron

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdims=True) #1 3 N 
        x = x / torch.abs(x).max()
        x = self.mapping(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x, output_size=(self.proj_neuron)) # N 128 128      
        x = self.norm_act(x)
        x = self.feat_extract(x)
        u_mean = self.out_mean(x)
        u_log_var = self.out_logvar(x)
        return u_mean, u_log_var


class IdiosyncraticLatentDynamicsConv(nn.Module):
    def __init__(
        self,
        mices_dict,
        channels=4, 
        latent_channel=256,
        dropout=0.0,
        grid_dict=None
    ):
        super().__init__()
        self.block_mean = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        self.block_std = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        
        self.grids_per_mice = {mice : torch.from_numpy(grid).to(torch.float32) for mice, grid in grid_dict['source_grid'].items()}
        if grid_dict is not None:
            self.grid_pridictor = Pos2Embedding(
                    input_channels=grid_dict['input_channels'],
                    hidden_channel=grid_dict['hidden_channel'],
                    output_channel=grid_dict['output_channel'],
                    proj_neuron=grid_dict['proj_neuron'],
                    dropout = dropout,
            )

    def forward(self, x, mice, flag_train=True):
        # for distributed training, the grid should be in all GPUs.
        positions = self.grids_per_mice[mice].unsqueeze(0).permute(0,2,1).to(x.device)
        u_mean, u_log_var  = self.grid_pridictor(positions) # 1 c N
        u_mean = u_mean.unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1])
        u_log_var = u_log_var.unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1])
        z_mean = self.block_mean(x)
        z_log_var = self.block_std(x)
        
        post_mean = (z_mean/(1+torch.exp(z_log_var-u_log_var))) + (u_mean/(1+torch.exp(u_log_var-z_log_var)))
        post_log_var = z_log_var + u_log_var - torch.log(torch.exp(z_log_var) + torch.exp(u_log_var))
        # x_sample = post_mean + torch.exp(0.5 * post_log_var) * torch.randn_like(post_mean).to(x.device)
        
        # kl_loss = 1 + post_log_var - u_log_var - ((torch.square(post_mean-u_mean) + torch.exp(post_log_var)) / torch.exp(u_log_var))
        # kl_loss = torch.sum(kl_loss) * (-1)
        # return x_sample, kl_loss
        return post_mean, post_log_var, z_mean, u_mean
    
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class PreservedLatentDynamics(nn.Module):
    def __init__(
        self,
        channels=16, 
        latent_channel=16,
        dropout=0.0,
    ):
        super().__init__()
        self.block_mean = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        self.block_std = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        
        self.embedding_dim = channels
        self.label_mapping = nn.Sequential(
            torch.nn.Linear(self.embedding_dim*8, 64),
            nn.SiLU(),
            torch.nn.Linear(64, 64),
        )
        # self.u_mean = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        # self.u_logvar = ResnetBlock(in_channels=latent_channel, out_channels=channels, dropout=dropout)
        self.u_mean = nn.Conv1d(64, channels, kernel_size=1, bias=True)
        self.u_logvar = nn.Conv1d(64, channels, kernel_size=1, bias=True)

    def forward(self, x, time_label, flag_train=True):
        time_embed = get_timestep_embedding(time_label, self.embedding_dim*8)
        label_mapped = self.label_mapping(time_embed).unsqueeze(-1)
        u_mean = self.u_mean(label_mapped)
        u_log_var = self.u_logvar(label_mapped)
        u_mean = u_mean.unsqueeze(-1).expand(x.shape[0],-1,x.shape[-2],x.shape[-1])
        u_log_var = u_log_var.unsqueeze(-1).expand(x.shape[0],-1,x.shape[-2],x.shape[-1])
        z_mean = self.block_mean(x)
        z_log_var = self.block_std(x)
        
        post_mean = (z_mean/(1+torch.exp(z_log_var-u_log_var))) + (u_mean/(1+torch.exp(u_log_var-z_log_var)))
        post_log_var = z_log_var + u_log_var - torch.log(torch.exp(z_log_var) + torch.exp(u_log_var))
        # x_sample = post_mean + torch.exp(0.5 * post_log_var) * torch.randn_like(post_mean).to(x.device)
        
        return post_mean, post_log_var, z_mean, u_mean, time_embed