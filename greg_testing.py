"""
The purpose of this code is to test the Prithvi Climate Model.

conda environment: base
"""
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PrithviWxC.dataloaders.merra2 import Merra2Dataset, input_scalers, output_scalers, static_input_scalers
from PrithviWxC.model import PrithviWxC

DATA_DIR = Path('/home/greg/gitcode/Prithvi-WxC/data')
surf_dir = DATA_DIR / 'merra-2'
vert_dir = DATA_DIR / 'merra-2'
time_range = ('2020-01-01T00:00:00', '2020-01-02T05:59:59')
surf_clim_dir = DATA_DIR / 'climatology'
vert_clim_dir = DATA_DIR / 'climatology'
positional_encoding = 'fourier'


def configure_backends():
    """
    Configure the backends and torch states, including setting the seeds for the random number generators.
    """
    torch.jit.enable_onednn_fusion(True)
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


def set_variables_and_levels():
    """
    The core model expects a fixed set of variables from the MERRA-2 dataset, which are prescribed below. The variables 
    comprise surface variables, surface static variables, and variables at various vertical levels within the 
    atmosphere. More details on the MERRA-2 dataset can be found at (https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/).

    The MERRA-2 dataset includes data at longitudes of -180° and +180°. This represents duplicate data, so we set a
    padding variable to remove it. The input to the core model consists of these variables at two different times. 
    The time difference in hours between these samples is passed to the model and set in the input_time variable.

    The model's task is to predict the fixed set of variables at a target time, given the input data. For example, if 
    the input times are 0900 and 1200, resulting in an input_time of -3, then a lead_time of 6 would result in a target 
    time of 1800.
    """
    # Make sure that all variables set below are available globally.
    global surface_vars, static_surface_vars, vertical_vars, levels, padding, lead_times, input_times, variable_names
    
    surface_vars = [
        'EFLUX',
        'GWETROOT',
        'HFLUX',
        'LAI',
        'LWGAB',
        'LWGEM',
        'LWTUP',
        'PS',
        'QV2M',
        'SLP',
        'SWGNT',
        'SWTNT',
        'T2M',
        'TQI',
        'TQL',
        'TQV',
        'TS',
        'U10M',
        'V10M',
        'Z0M',
    ]
    static_surface_vars = ['FRACI', 'FRLAND', 'FROCEAN', 'PHIS']
    vertical_vars = ['CLOUD', 'H', 'OMEGA', 'PL', 'QI', 'QL', 'QV', 'T', 'U', 'V']
    levels = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]
    padding = {'level': [0, 0], 'lat': [0, -1], 'lon': [0, 0]}
    lead_times = [18]  # This varibale can be changed to change the task
    input_times = [-6]  # This varibale can be changed to change the task
    variable_names = surface_vars + [
        f'{var}_level_{level}' for var in vertical_vars for level in levels
    ]


def download_data():
    """
    Download the data from the Hugging Face Hub.
    """
    snapshot_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        # TODO: The mask below needs to be automated so that it is consistent with the selected time range.
        allow_patterns='merra-2/MERRA2_sfc_2020010[1-2].nc',
        local_dir=DATA_DIR,
    )
    snapshot_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        # TODO: The mask below needs to be automated so that it is consistent with the selected time range.
        allow_patterns='merra-2/MERRA_pres_2020010[1-2].nc',
        local_dir=DATA_DIR
    )


def download_climatology():
    """
    Download the climatology data from the Hugging Face Hub.
    """
    snapshot_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        # TODO: The mask below needs to be automated so that it is consistent with the selected time range.
        allow_patterns='climatology/climate_surface_doy00[1-2]*.nc',
        local_dir=DATA_DIR,
    )
    snapshot_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        # TODO: The mask below needs to be automated so that it is consistent with the selected time range.
        allow_patterns='climatology/climate_vertical_doy00[1-2]*.nc',
        local_dir=DATA_DIR,
    )    


def instantiate_merra2_dataset():
    """
    Instantiate the MERRA-2 dataset.
    """
    dataset = Merra2Dataset(
        time_range=time_range,
        lead_times=lead_times,
        input_times=input_times,
        data_path_surface=surf_dir,
        data_path_vertical=vert_dir,
        climatology_path_surface=surf_clim_dir,
        climatology_path_vertical=vert_clim_dir,
        surface_vars=surface_vars,
        static_surface_vars=static_surface_vars,
        vertical_vars=vertical_vars,
        levels=levels,
        positional_encoding=positional_encoding,
    )
    assert len(dataset) > 0, "There doesn't seem to be any valid data."    


def load_scalers():
    """
    The model takes as static parameters the mean and variance values of the input variables and the variance values of
    the target difference, i.e., the variance between climatology and instantaneous variables. We have provided data 
    files containing these values, and here we load these data.
    """
    global output_sig, in_mu, in_sig, static_mu, static_sig

    surf_in_scal_path = surf_clim_dir / 'musigma_surface.nc'
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename=f'climatology/{surf_in_scal_path.name}',
        local_dir=DATA_DIR,
        )
    vert_in_scal_path =surf_clim_dir / 'musigma_vertical.nc'
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename=f'climatology/{vert_in_scal_path.name}',
        local_dir=DATA_DIR,
    )
    surf_out_scal_path = surf_clim_dir / 'anomaly_variance_surface.nc'
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename=f'climatology/{surf_out_scal_path.name}',
        local_dir=DATA_DIR,
    )
    vert_out_scal_path = surf_clim_dir / 'anomaly_variance_vertical.nc'
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename=f'climatology/{vert_out_scal_path.name}',
        local_dir=DATA_DIR,
    )
    in_mu, in_sig = input_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_in_scal_path,
        vert_in_scal_path,
    )
    output_sig = output_scalers(
        surface_vars,
        vertical_vars,
        levels,
        surf_out_scal_path,
        vert_out_scal_path,
    )
    static_mu, static_sig = static_input_scalers(
        surf_in_scal_path,
        static_surface_vars,
    )    


def setup_shifting():
    """
    Primarily utilized in the decoder, this enables alternate shifting of the attention windows, similar to the SWIN
    model. This option necessitates an even number of decoder blocks and is incompatible with the encoder when masking
    is also employed.
    """
    global residual, masking_mode, encoder_shifting, decoder_shifting, masking_ratio
    residual = 'climate'
    masking_mode = 'global'
    encoder_shifting = True
    decoder_shifting = True
    masking_ratio = 0.0

def initialise_model():
    """
    Initialise the Prithvi Climate Model.
    """
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename='config.yaml',
        local_dir='../data',
    )

    with open('../data/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return PrithviWxC(
        in_channels=config['params']['in_channels'],
        input_size_time=config['params']['input_size_time'],
        in_channels_static=config['params']['in_channels_static'],
        input_scalers_mu=in_mu,
        input_scalers_sigma=in_sig,
        input_scalers_epsilon=config['params']['input_scalers_epsilon'],
        static_input_scalers_mu=static_mu,
        static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=config['params'][
            'static_input_scalers_epsilon'
        ],
        output_scalers=output_sig**0.5,
        n_lats_px=config['params']['n_lats_px'],
        n_lons_px=config['params']['n_lons_px'],
        patch_size_px=config['params']['patch_size_px'],
        mask_unit_size_px=config['params']['mask_unit_size_px'],
        mask_ratio_inputs=masking_ratio,
        mask_ratio_targets=0.0,
        embed_dim=config['params']['embed_dim'],
        n_blocks_encoder=config['params']['n_blocks_encoder'],
        n_blocks_decoder=config['params']['n_blocks_decoder'],
        mlp_multiplier=config['params']['mlp_multiplier'],
        n_heads=config['params']['n_heads'],
        dropout=config['params']['dropout'],
        drop_path=config['params']['drop_path'],
        parameter_dropout=config['params']['parameter_dropout'],
        residual=residual,
        masking_mode=masking_mode,
        encoder_shifting=encoder_shifting,
        decoder_shifting=decoder_shifting,
        positional_encoding=positional_encoding,
        checkpoint_encoder=[],
        checkpoint_decoder=[],
    )


def load_weigths(model_, device_):
    """
    Load the model weights from the Hugging Face Hub.
    """
    weights_path = Path('../data/weights/prithvi.wxc.2300m.v1.pt')
    hf_hub_download(
        repo_id='ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
        filename=weights_path.name,
        local_dir='../data/weights',
    )

    state_dict = torch.load(weights_path, weights_only=False)
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    model_.load_state_dict(state_dict, strict=True)

    if (hasattr(model_, 'device') and model_.device != device_) or not hasattr(
        model, 'device'
    ):
        model = model_.to(device)

if __name__ == '__main__':
    configure_backends()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_variables_and_levels()
    download_data()
    download_climatology()
    instantiate_merra2_dataset()
    load_scalers()
    setup_shifting()
    model = initialise_model()
    load_weigths(model, device)
    print('running finished')
