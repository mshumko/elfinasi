import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import IRBEM
import numpy as np
import pandas as pd
import pyspedas
import pytplot

import pad


R_E = 6378.137  # km
time_range = ('2022-09-04T03:35', '2022-09-04T03:40')
location_codes = ('PINA', 'GILL', 'KAPU', 'TPAS')
themis_probes = ('a', 'd', 'e')
elfin_id = 'b'
alt=110

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

irbem_model = IRBEM.MagFields(kext='T89', sysaxes=3)
def map_themis(themis_probe, time_range, alt):
    """
    Map the THEMIS location to the ionosphere.
    """
    state_vars = pyspedas.themis.state(probe=themis_probe, trange=time_range, time_clip=True)
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

    for time, row in state_df.iterrows():
        X = {'Time':time, 'x1':row['X'], 'x2':row['Y'], 'x3':row['Z']}
        mapped_df.loc[time, :] = irbem_model.find_foot_point(X, {'Kp':56}, alt, 0)['XFOOT']
    mapped_df[mapped_df == -1E31] = np.nan
    return mapped_df

themis_mapped_state = {themis_probe:map_themis(themis_probe, time_range, alt) for themis_probe in themis_probes}

fig = plt.figure(figsize=(10, 5.5))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,
                      left=0.01, right=0.95, bottom=0.25, top=0.95,
                      wspace=0.18, hspace=0.15)
ax = asilib.map.create_map(lon_bounds=(-115, -80), lat_bounds=(41, 65), fig_ax=(fig, gs[:, 0]))
bx = fig.add_subplot(gs[0, 1])
cx = fig.add_subplot(gs[1, 1], sharex=bx, sharey=bx)

plt.suptitle(f'THEMIS-THEMIS-ELFIN Conjunction | {time_range[0][:10]} | T89 model | {alt} km map altitude')

pad_obj = pad.EPD_PAD(
    elfin_id, time_range, start_pa=0, min_counts=None, accumulate=1, spin_time_tol=(2.5, 12),
    lc_exclusion_angle=0
)
transformed_state = pad_obj.transform_state()
transformed_state = transformed_state.loc[time_range[0]:time_range[1]]
pad_obj.plot_omni(bx, labels=True, colorbar=True, vmin=1E2, vmax=1E7, pretty_plot=False)
pad_obj.plot_blc_dlc_ratio(cx, labels=True, colorbar=True, cmap='viridis', vmin=1E-2, vmax=1)
pad_obj.plot_position(cx)
cx.xaxis.set_major_locator(plt.MaxNLocator(5))
cx.xaxis.set_label_coords(-0.15, -0.007*7)
cx.xaxis.label.set_size(10)

mapped_state = map_elfin(transformed_state, alt=alt)
ax.plot(mapped_state['lon'], mapped_state['lat'], c='r', transform=ccrs.PlateCarree())

asis = asilib.Imagers(
    [asilib.asi.themis(location_code=location_code, time_range=time_range, alt=alt) 
    for location_code in location_codes]
    )
gen = asis.animate_map_gen(overwrite=True, ax=ax, min_elevation=5)
for _guide_time, _asi_times, _asi_images, _ in gen:
    if 'vline_bx' in locals():
        # These variables are only used when ELFIN is in the FOV
        vline_bx.remove()
        vline_cx.remove()
        _elfin_loc.remove()
    
    if '_plot_time' in locals():
        # These variables are used for every timestamp.
        _themisa_loc.remove()
        _themisd_loc.remove()
        _themise_loc.remove()
        _plot_time.remove()

    if (
        (_guide_time > mapped_state.index[0]) or 
        (_guide_time < mapped_state.index[-1])
        ):
        vline_bx = bx.axvline(_guide_time, c='r')
        vline_cx = cx.axvline(_guide_time, c='r')

        _elfin_loc = ax.scatter(
            mapped_state.loc[_guide_time, 'lon'], 
            mapped_state.loc[_guide_time, 'lat'], 
            c='r', 
            s=30,
            transform=ccrs.PlateCarree(),
            label='ELFIN'
            )
    _themisa_index = themis_mapped_state['a'].index.get_indexer([_guide_time], method='nearest')
    _themisa_loc = ax.scatter(
        themis_mapped_state['a'].iloc[_themisa_index, :]['lon'], 
        themis_mapped_state['a'].iloc[_themisa_index, :]['lat'], 
        c='orange', 
        s=30,
        marker='^',
        transform=ccrs.PlateCarree(),
        label='THEMIS-A',
        zorder=2.01
        )
    _themisd_index = themis_mapped_state['d'].index.get_indexer([_guide_time], method='nearest')
    _themisd_loc = ax.scatter(
        themis_mapped_state['d'].iloc[_themisd_index, :]['lon'], 
        themis_mapped_state['d'].iloc[_themisd_index, :]['lat'], 
        c='orange', 
        s=30,
        marker='P',
        transform=ccrs.PlateCarree(),
        label='THEMIS-D',
        zorder=2.01
        )
    _themise_index = themis_mapped_state['e'].index.get_indexer([_guide_time], method='nearest')
    _themise_loc = ax.scatter(
        themis_mapped_state['e'].iloc[_themise_index, :]['lon'], 
        themis_mapped_state['e'].iloc[_themise_index, :]['lat'],
        c='orange',
        s=30,
        marker='*',
        transform=ccrs.PlateCarree(),
        label='THEMIS-E',
        zorder=2.01
        )
    if not '_legend' in locals():
        _legend = ax.legend(loc='lower left', ncols=2)
    _plot_time = ax.text(
        0.01, 0.99, f'{_guide_time:%Y-%m-%d %H:%M:%S}', va='top', transform=ax.transAxes, fontsize=18
        )