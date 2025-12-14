import IRBEM
import numpy as np
import pandas as pd
import pyspedas
import pytplot

R_E = 6378.137  # km

def map_elfin(df, alt=110, hemi_flag=0):
    """
    Map ELFIN's location along the magnetic field line to alt using IRBEM.MagFields.find_foot_print.

    Parameters
    ----------
    alt: float
        The mapping altitude in units of kilometers
    hemi_flag: int
        What direction to trace the field line: 
        0 = same magnetic hemisphere as starting point
        +1   = northern magnetic hemisphere
        -1   = southern magnetic hemisphere
        +2   = opposite magnetic hemisphere as starting point
    """
    m = IRBEM.MagFields(kext='T89')
    _all = np.zeros_like(df.loc[:, ['alt', 'lat', 'lon']])

    for i, (time, row) in enumerate(df.iterrows()):
        X = {'Time':time, 'x1':row['alt'], 'x2':row['lat'], 'x3':row['lon']}
        _all[i, :] = m.find_foot_point(X, {'Kp':56}, alt, hemi_flag)['XFOOT']
    _all[_all == -1E31] = np.nan
    mapped_df = df.copy()
    mapped_df.loc[:, ['alt', 'lat', 'lon']] = _all
    return mapped_df

def map_themis(themis_probe, time_range, alt, no_update=False):
    """
    Map the THEMIS location to the ionosphere.
    """
    state_vars = pyspedas.themis.state(probe=themis_probe, trange=time_range, time_clip=True, no_update=no_update)
    state_xr = pytplot.get_data(f'th{themis_probe}_pos_gse')
    state_times = pyspedas.time_datetime(state_xr.times)
    state_times = [state_time.replace(tzinfo=None) for state_time in state_times]
    state_df = pd.DataFrame(
        index=state_times,
        data={component.upper():state_xr.y[:, i]/R_E for i, component in enumerate(['x', 'y', 'z'])}
        )
    
    mapped_df = pd.DataFrame(
        index=state_times,
        data={
            'alt':np.nan*np.zeros(state_df['X'].shape),
            'lat':np.nan*np.zeros(state_df['X'].shape),
            'lon':np.nan*np.zeros(state_df['X'].shape)
            }
    )
    
    irbem_model = IRBEM.MagFields(kext='T89', sysaxes=3)
    for time, row in state_df.iterrows():
        X = {'Time':time, 'x1':row['X'], 'x2':row['Y'], 'x3':row['Z']}
        mapped_df.loc[time, :] = irbem_model.find_foot_point(X, {'Kp':56}, alt, 0)['XFOOT']
    mapped_df[mapped_df == -1E31] = np.nan
    return mapped_df