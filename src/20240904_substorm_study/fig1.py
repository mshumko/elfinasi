"""
Make a THEMIS-GOES location plot.
"""
import string
import pathlib
import dateutil.parser
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pyspedas
import pytplot
import IRBEM
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature
import asilib.asi

import elfinasi
from map import map_elfin, map_themis

save_locations=False  # Save the THEMIS location and a series of ELFIN locations.
R_e = 6378.137  # km
time_range_str = ('2022-09-04T04:18', '2022-09-04T04:29')
time_range = [dateutil.parser.parse(t_i) for t_i in time_range_str]
elfin_times = (
    datetime(2022, 9, 4, 4, 19, 18),
    datetime(2022, 9, 4, 4, 20, 18),
    datetime(2022, 9, 4, 4, 21, 24),
    datetime(2022, 9, 4, 4, 22, 22)
)
# poes_times = (
#     datetime(2022, 9, 4, 4, 24, 0),
#     datetime(2022, 9, 4, 4, 25, 0),
#     datetime(2022, 9, 4, 4, 26, 0),
#     datetime(2022, 9, 4, 4, 27, 0)
# )
trex_image_time = datetime(2022, 9, 4, 4, 21, 24)
elfin_id = 'a'
poes_probe = 'noaa18'
coordinates = 'gsm'
themis_probes = ('a', 'd', 'e')
markers = ("*", "P", "D", "o", 'v')
colors=('orange', 'blue', 'purple', 'k', 'red')
alt = 110
locations={}
location_codes = ('ATHA', 'PINA', 'GILL', 'RABB', 'LUCK')
no_update=False

fig = plt.figure(figsize=(9, 4.5))
gs = fig.add_gridspec(2, 2,
        left=0.07, right=0.99, bottom=0.12, top=0.93,
        wspace=0.02, hspace=0.05,
        height_ratios=(1, 1), 
        width_ratios=(0.75, 1)
        )
x_lim=(-12, 0)
y_lim=(0, 5)
z_lim =(-2.5, 2.5)

ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[1, 0], sharex=ax)
cx = fig.add_subplot(gs[:, 1], projection=ccrs.Orthographic(-100, 50))
cx.add_feature(cartopy.feature.LAND, color='grey')
cx.add_feature(cartopy.feature.OCEAN, color='w')
cx.add_feature(cartopy.feature.COASTLINE, edgecolor='k')
cx.set_extent([-117, -82, 43, 63])
cx.gridlines()

for ax_i, _label in zip([ax, bx, bx], string.ascii_lowercase):
    ax_i.text(0.01, 0.98, f'({_label})', fontsize=15, transform=ax_i.transAxes, va='top')

def trace_field_line(time, X, kp=50, sysaxes=2):
    """
    sysaxes=2 is GSM
    """
    LLA = {key:X[i] for i, key in enumerate(['x1', 'x2', 'x3'])}
    LLA['dateTime'] = time
    maginput = {'Kp':kp}
    model = IRBEM.MagFields(kext='T89', sysaxes=sysaxes)  
    out = model.trace_field_line(LLA, maginput)

    # Transform from GEO to GSM
    _coords_obj = IRBEM.Coords()
    r_gsm = _coords_obj.transform(
        np.full(out['POSIT'][:, 0].shape, LLA['dateTime']), 
        out['POSIT'], 
        1, 2)
    return r_gsm

def footprint(df, alt=110, hemi_flag=0):
    """
    Find the satellite footprint.

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

for _probe, color, marker in zip(themis_probes, colors, markers):
    state_vars = pyspedas.themis.state(probe=_probe, trange=time_range_str, time_clip=True, no_update=no_update)  # 'thd_pos_gsm'
    state_xr = pytplot.get_data(f'th{_probe}_pos_{coordinates}')
    locations[f'themis-{_probe}'] = state_xr.y[0, :]/R_e

    themis_mapped_state = map_themis(_probe, time_range_str, alt, no_update=no_update)

    _themisa_index = themis_mapped_state.index.get_indexer([time_range[0]], method='nearest')
    _themisa_loc = cx.scatter(
        themis_mapped_state.iloc[_themisa_index, :]['lon'], 
        themis_mapped_state.iloc[_themisa_index, :]['lat'], 
        c=color, 
        s=50,
        marker=marker,
        transform=ccrs.PlateCarree(),
        zorder=2.01
        )

    field_line = trace_field_line(time_range[0], locations[f'themis-{_probe}'])
    ax.plot(field_line[:, 0], field_line[:, 1], c='k')
    bx.plot(field_line[:, 0], field_line[:, 2], c='k')

asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code=location_code, time=trex_image_time, alt=alt) 
        for location_code in location_codes]
        )
asis.plot_map(ax=cx, min_elevation=10, pcolormesh_kwargs={'rasterized':True}, asi_label=True)

pad_obj_nflux = elfinasi.EPD_PAD(
    elfin_id, time_range, start_pa=0, min_counts=None, accumulate=1, spin_time_tol=(2.5, 12),
    lc_exclusion_angle=0
)
transformed_state = pad_obj_nflux.transform_state()
transformed_state = transformed_state.loc[time_range[0]:time_range[1]]
mapped_state = map_elfin(transformed_state, alt=alt)
cx.plot(
    mapped_state['lon'], 
    mapped_state['lat'], 
    c='orange', 
    transform=ccrs.PlateCarree(), 
    label=f'ELFIN-{elfin_id.upper()}'
    )

for time in elfin_times:
    elfin_scatter = cx.scatter(
        mapped_state.loc[time, 'lon'], 
        mapped_state.loc[time, 'lat'], 
        c='orange', 
        s=50,
        transform=ccrs.PlateCarree(),
        label=f'ELFIN-{elfin_id.upper()}'
        )
    cx.text(
        mapped_state.loc[time, 'lon']+2*np.cos(np.deg2rad(mapped_state.loc[time, 'lat'])), 
        mapped_state.loc[time, 'lat'],
        f'{time:%H:%M:%S}',
        transform=ccrs.PlateCarree(),
        color='orange',
        va='center',
        ha='left'
        )

for (_probe, _loc), marker, color in zip(locations.items(), markers, colors):
    # THEMIS
    ax.scatter(_loc[0], _loc[1], color=color, marker=marker, zorder=2.01, s=50, label=_probe.upper())
    bx.scatter(_loc[0], _loc[2], color=color, marker=marker, zorder=2.01, s=50)

# # Plot NOAA footprints
# sem_vars = pyspedas.poes.sem(
#     trange=time_range_str, 
#     probe=poes_probe, 
#     time_clip=True, 
#     no_update=True
#     )
# ephemeris_df = pd.DataFrame(
#     index=pyspedas.data_quants['lat'].time,
#     data={
#         'alt':pyspedas.data_quants['alt'].to_numpy(),
#         'lat':pyspedas.data_quants['lat'].to_numpy(),
#         'lon':pyspedas.data_quants['lon'].to_numpy(),
#     }
# )
# poes_footprint = footprint(ephemeris_df, alt=alt)
# cx.plot(
#     poes_footprint['lon'], 
#     poes_footprint['lat'], 
#     c='w',
#     ls='--',
#     transform=ccrs.PlateCarree(), 
#     label=f'NOAA-{poes_probe[-2:]}'
#     )
# ephemeris_indices = poes_footprint.index.get_indexer(poes_times, method='nearest')
# poes_footprint_snapshots = poes_footprint.iloc[ephemeris_indices, :]
# cx.legend(loc='lower left')
# 
# cx.scatter(
#         poes_footprint_snapshots['lon'], 
#         poes_footprint_snapshots['lat'], 
#         c='w', 
#         s=30,
#         transform=ccrs.PlateCarree(),
#         label=poes_probe.upper()
#         )
# for time, mapped_ephemeris_snapshot in poes_footprint_snapshots.iterrows():
#     cx.text(
#             mapped_ephemeris_snapshot['lon']-2*np.cos(np.deg2rad(mapped_ephemeris_snapshot['lat'])), 
#             mapped_ephemeris_snapshot['lat'],
#             f'{time:%H:%M:%S}',
#             transform=ccrs.PlateCarree(),
#             color='white',
#             va='center',
#             ha='right'
#             )

# https://matplotlib.org/stable/users/explain/axes/legend_guide.html#proxy-legend-handles
elfin_legend_line = mlines.Line2D(
    [], [], 
    color='orange', 
    marker='.',                      
    markersize=15, 
    label=f'ELFIN-{elfin_id.upper()}'
    )
cx.legend(handles=[elfin_legend_line], loc='lower left')
ax.legend()
ax.set(
    xlabel=f'$X_{{{coordinates.upper()}}} $ [$R_{{E}}$]', 
    ylabel=f'$Y_{{{coordinates.upper()}}}$ [$R_{{E}}$]', 
    xlim=x_lim, 
    ylim=y_lim
    )
ax.axis('equal')
bx.set(
    xlabel=f'$X_{{{coordinates.upper()}}} $ [$R_{{E}}$]', 
    ylabel=f'$Z_{{{coordinates.upper()}}}$ [$R_{{E}}$]', 
    xlim=x_lim, 
    ylim=z_lim
    )
bx.axhline(0, ls='--', c='k')
bx.axis('equal')

for ax_i in [ax, bx]:
    earth_circle = plt.Circle((0, 0), 1, color='k', fill='grey')
    ax_i.add_patch(earth_circle)

seven_re = plt.Circle((0, 0), 7, color='k', fill=None, ls='--')
ax.add_patch(seven_re)

ax.xaxis.set_visible(False)
plt.suptitle(
    f'THEMIS, TREx, and ELFIN | {time_range[0]:%Y-%m-%d %H:%M} - {time_range[1]:%H:%M}'
    f' | {alt} map altitude | T89 field model'
    )
# plt.tight_layout()

if save_locations:
    save_dir = pathlib.Path(__file__).parent / 'data'
    for themis_probe in themis_probes:
        save_path = save_dir/ f'{time_range[0]:%Y%m%d_%H%M}_themis{themis_probe}_{coordinates}_ephemeris.txt'
        save_path.write_text(', '.join(locations[f'themis_{themis_probe}'].astype(str)))
    transformed_state.to_csv(
        save_dir / f'{time_range[0]:%Y%m%d_%H%M}_{time_range[1]:%H%M}_elfin{elfin_id}_geo_ephemeris.csv', 
        index_label='time'
        )
plt.show()