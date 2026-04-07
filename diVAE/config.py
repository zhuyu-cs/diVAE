# model settings
model = 'diVAE'
seed = 42

model_dict = dict(  
    proj_dict=dict(
        in_dim=1,
        proj_neuron=256,             
        frames=16,
        num_param=4,
        dropout=0.1,
        base_channel=64,
    ),
    encoder_decoder_dict=dict(  
        base_channel=32, 
        ch_mult=(1,2,2,4), 
        num_res_blocks=1,      
        attn_resolutions=[],          
        dropout=0.1, 
        resamp_with_conv=True,
        double_z=False,
        give_pre_end=False,
        z_channels=2,             
    ),
    pld_dict=dict(
        n_e=256,                   
        code_dim=2,               
    ),
    ild_dict=dict(
        latent_channel=2,          
        dropout=0.1
    ),
    grid_dict=dict(
        input_channels=3,
        hidden_channel=2,
        output_channel=2,
    )
)

# dataset settings
batch_size = 16
mixup=False
coeffKL=0.5
# optimizer and learning rate
optimizer = dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None) 
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    periods=[400,400],
    restart_weights=[1,1],
    min_lr=[2e-4,1e-7],
)


# runtime settings
gpus = [0]
dist_params = dict(backend='nccl')
data_workers = 2 
checkpoint_config = dict(interval=800) 
workflow = [('train', 800), ('val', 1)]
total_epochs = 800
resume_from =  None
load_from = None
work_dir = './work_dir/diVAE'

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=8, 
    hooks=[
        dict(type='TextLoggerHook'),
    ])
