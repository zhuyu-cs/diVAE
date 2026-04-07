import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import (LDProjector, invLDProjector,
                              Encoder, Decoder, 
                              PreservedLatentDynamics,
                              IdiosyncraticLatentDynamicsConv)


class diVAE(nn.Module):
    def __init__(self, 
                 mices_dict,
                 proj_dict,
                 encoder_decoder_dict,
                 pld_dict,
                 ild_dict,
                 grid_dict,
    ):
        super().__init__()
        
        # input projectors.
        self.in_projectors = LDProjector(mices_dict=mices_dict,
                                         in_dim=proj_dict['in_dim'], 
                                         proj_neuron=proj_dict['proj_neuron'], 
                                         base_channel=proj_dict['base_channel'], 
                                         dropout=proj_dict['dropout'])
        
        # feature encoder
        self.encoder = Encoder(
            ch=encoder_decoder_dict['base_channel'],
            ch_mult=encoder_decoder_dict['ch_mult'],
            num_res_blocks=encoder_decoder_dict['num_res_blocks'],
            attn_resolutions=encoder_decoder_dict['attn_resolutions'],
            dropout=encoder_decoder_dict['dropout'],
            resamp_with_conv=encoder_decoder_dict['resamp_with_conv'], 
            in_channels=proj_dict['base_channel'], 
            resolution=proj_dict['proj_neuron'],
            z_channels=encoder_decoder_dict['z_channels'], 
            double_z=False
        )

        # idiosyncratic latent dynamics
        self.ilds = IdiosyncraticLatentDynamicsConv(
            mices_dict=mices_dict,
            channels=encoder_decoder_dict['z_channels'], 
            latent_channel=ild_dict['latent_channel'],
            dropout=ild_dict['dropout'],
            grid_dict=grid_dict
        )
        
        # preserved latent dynamics
        self.pld = PreservedLatentDynamics(
            channels=encoder_decoder_dict['z_channels'], 
            latent_channel=ild_dict['latent_channel'],
            dropout=ild_dict['dropout'],
            # code_dim=pld_dict['code_dim'],   
            # n_e=pld_dict['n_e'],               
            # z_channels=encoder_decoder_dict['z_channels']  
        )
        # feature decoder
        self.decoder = Decoder(
            ch=encoder_decoder_dict['base_channel']*2, 
            ch_mult=encoder_decoder_dict['ch_mult'], 
            num_res_blocks=encoder_decoder_dict['num_res_blocks'], 
            attn_resolutions=encoder_decoder_dict['attn_resolutions'],  
            dropout=encoder_decoder_dict['dropout'], 
            resamp_with_conv=encoder_decoder_dict['resamp_with_conv'],  
            in_channels=proj_dict['base_channel'], 
            resolution=proj_dict['proj_neuron'],
            z_channels=encoder_decoder_dict['z_channels']*2, 
        )

        # output projectors
        self.output_projectors = invLDProjector(mices_dict=mices_dict,
                                                in_dim=proj_dict['in_dim'],
                                                base_channel=proj_dict['base_channel'], 
                                                dropout=proj_dict['dropout'],
                                                )
    
    def forward(self, cal_signal, time_label, mice): 
        # ca: B N t
        # mice: id of mouse.
        # project
        proj_ca = self.in_projectors(cal_signal) # compress original trace to the same spatial size (Neuron), eg., B 1 7440 32 -> B 128 1024 32 
        # encode
        embed_ca = self.encoder(proj_ca) # B c N t
        
        plds_ca, pld_log_var, zt_mean, lt_embed, time_embed = self.pld(embed_ca, time_label)   # B c N t
        ilds_ca, ild_log_var, zc_mean, id_embed = self.ilds(embed_ca, mice)           # B c N t
        
        return plds_ca, ilds_ca