from .divae import diVAE


def make_diVAE(
    dataloaders,
    proj_dict=dict(),
    encoder_decoder_dict=dict(),
    ild_dict=dict(),
    pld_dict=dict(),
    grid_dict=dict()
):
    
    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    mices_dict = {k: v.dataset.neurons for k, v in dataloaders.items()}     
   
    grid_dict['source_grid'] = {k: v.dataset.position for k, v in dataloaders.items()}     
    grid_dict['proj_neuron'] = proj_dict['proj_neuron']//2**(len(encoder_decoder_dict['ch_mult'])-1) 
    
    model = diVAE(
        mices_dict,
        proj_dict=proj_dict,
        encoder_decoder_dict=encoder_decoder_dict,
        pld_dict=pld_dict,
        ild_dict=ild_dict,
        grid_dict=grid_dict
    )

    return model

