"""
Make a location plot with ELFIN, THEMIS, TREx, and POES locations.
"""
import string
import pathlib
import dateutil.parser
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.lines as mlines
import pyspedas
import pytplot
import IRBEM
import numpy as np
import pandas as pd
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
keogram_time_range = (datetime(2022, 9, 4, 2, 37, 0, 0), datetime(2022, 9, 4, 10, 0, 0, 0))

trex_image_time = datetime(2022, 9, 4, 4, 21, 24)
elfin_id = 'a'
coordinates = 'gsm'
themis_probes = ('a', 'd', 'e')
markers = ("*", "P", "D", "o", 'v')
colors=('orange', 'blue', 'purple', 'k', 'red')
alt = 110
locations={}
asi_location_codes = ('ATHA', 'PINA', 'GILL', 'RABB', 'LUCK')
no_update=False

supermag = pd.read_csv(elfinasi.data_dir / '20250331-19-15-supermag.csv')
supermag.index = pd.to_datetime(supermag['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
del(supermag['Date_UTC'])
supermag[supermag == 999999] = np.nan

fig = plt.figure(figsize=(8, 9))
n_rows = 7

gs = fig.add_gridspec(
    2, 
    2, 
    left=0.1, right=0.99, bottom=0.07, top=0.95,
    wspace=0.02, hspace=0.15,
    height_ratios=(1.2, 1)
    )
top_gs = gs[0, :].subgridspec(n_rows-2, 1, hspace=0.03)
bottom_gs = gs[1, :].subgridspec(2, 2, wspace=0.01, hspace=0.03, width_ratios=(1, 2))

x_lim=(-12, 0)
y_lim=(0, 5)
z_lim =(-2.5, 2.5)

ax = np.zeros(n_rows-2, dtype=object)
ax[0] = fig.add_subplot(top_gs[0, :])
for i in range(1, n_rows-2):
    ax[i] = fig.add_subplot(top_gs[i, :], sharex=ax[0])
bx = fig.add_subplot(bottom_gs[0, 0])
cx = fig.add_subplot(bottom_gs[1, 0], sharex=bx)
dx = fig.add_subplot(bottom_gs[:, 1], projection=ccrs.Orthographic(-100, 50))
dx.add_feature(cartopy.feature.LAND, color='grey')
dx.add_feature(cartopy.feature.OCEAN, color='w')
dx.add_feature(cartopy.feature.COASTLINE, edgecolor='k')
dx.set_extent([-117, -82, 43, 63])
dx.gridlines()

ax[0].plot(supermag.index, supermag['GSM_Bz'], c='k', label='Bz')
ax[0].set_ylabel(f'IMF $B_{{z}}$ [nT]')
v = np.linalg.norm(supermag[['GSE_Vx', 'GSE_Vy', 'GSM_Vz']], axis=1)
ax[1].plot(supermag.index, v, c='k', label='|V|')
ax[1].set_ylabel(f'$|V|$ [km/s]')
ax[2].plot(supermag.index, supermag.SML, c='k', label='SML')
ax[2].set_ylabel('SML [nT]')
ax[3].plot(supermag.index, supermag.SMR, c='k', label='SMR')
ax[3].set_ylabel('SMR [nT]')

for ax_i in ax:
    if ax_i == ax[-1]:
        ax_i.axvline(elfin_times[0], ls='--', c='w')
    else:
        ax_i.axvline(elfin_times[0], ls='--', c='k')
for ax_i in ax[:-1]:
    # ax_i.get_xaxis().set_visible(False)
    ax_i.set_xticklabels([])
ax[0].axhline(0, c='k', alpha=0.5)
ax[-1].set_xlim(supermag.index[0], supermag.index[-1])

def _format_xaxis(tick_val, tick_pos):
    """
    The tick magic to include the date only on the first tick, and ticks at midnight.
    """
    tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
    if (tick_pos==0) or ((tick_time.hour == 0) and (tick_time.minute == 0)):
        ticks = tick_time.strftime('%H:%M') + '\n' + tick_time.strftime('%Y-%m-%d')
    else:
        ticks = tick_time.strftime('%H:%M')
    return ticks

ax[-1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_xaxis))

for ax_i, _label in zip([*ax, bx, cx, dx], string.ascii_lowercase):
    ax_i.text(0.01, 0.98, f'({_label})', fontsize=15, transform=ax_i.transAxes, va='top')

def trace_field_line(time, X, kp=50, sysaxes=2):
    """
    sysbxes=2 is GSM
    """
    X = {key:X[i] for i, key in enumerate(['x1', 'x2', 'x3'])}
    X['dateTime'] = time
    maginput = {'Kp':kp}
    model = IRBEM.MagFields(kext='T89', sysaxes=sysaxes)  
    out = model.trace_field_line(X, maginput)

    # Transform from GEO to GSM
    _coords_obj = IRBEM.Coords()
    r_gsm = _coords_obj.transform(
        np.full(out['POSIT'][:, 0].shape, X['dateTime']), 
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
    _themisa_loc = dx.scatter(
        themis_mapped_state.iloc[_themisa_index, :]['lon'], 
        themis_mapped_state.iloc[_themisa_index, :]['lat'], 
        c=color, 
        s=50,
        marker=marker,
        transform=ccrs.PlateCarree(),
        zorder=2.01
        )

    field_line = trace_field_line(time_range[0], locations[f'themis-{_probe}'].astype(float))
    bx.plot(field_line[:, 0], field_line[:, 1], c='k')
    cx.plot(field_line[:, 0], field_line[:, 2], c='k')

gill_asi = asilib.asi.trex_rgb('gill', time_range=keogram_time_range)
pina_asi = asilib.asi.trex_rgb('pina', time_range=keogram_time_range)
# Plot keogram along THEMIS's latitude.
_themis_lon = -98.2  # By eye---too lazy to copy the mapping code here.
gill_keogram_path=np.stack((np.linspace(55, 60, num=500), _themis_lon*np.ones(500)), axis=1)
pina_keogram_path=np.stack((np.linspace(45, 55.5, num=500), _themis_lon*np.ones(500)), axis=1)
gill_asi.plot_keogram(ax=ax[-1], aacgm=True, title=False, path=gill_keogram_path, pcolormesh_kwargs={'rasterized':True})
pina_asi.plot_keogram(ax=ax[-1], aacgm=True, title=False, path=pina_keogram_path, pcolormesh_kwargs={'rasterized':True})
ax[-1].set(ylabel=f'AACGM ${{\lambda}}$ [${{\circ}}$]')

asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code=location_code, time=trex_image_time, alt=alt) 
        for location_code in asi_location_codes]
        )
asis.plot_map(ax=dx, min_elevation=10, pcolormesh_kwargs={'rasterized':True}, asi_label=True)
dx.text(0.01, 0.98, f'     {trex_image_time:%H:%M:%S} | {alt} km altitude', fontsize=15, transform=dx.transAxes, va='top')

pad_obj_nflux = elfinasi.EPD_PAD(
    elfin_id, time_range, start_pa=0, min_counts=None, accumulate=1, spin_time_tol=(2.5, 12),
    lc_exclusion_angle=0
)
transformed_state = pad_obj_nflux.transform_state()
transformed_state = transformed_state.loc[time_range[0]:time_range[1]]
mapped_state = map_elfin(transformed_state, alt=alt)
dx.plot(
    mapped_state['lon'], 
    mapped_state['lat'], 
    c='orange', 
    transform=ccrs.PlateCarree(), 
    label=f'ELFIN-{elfin_id.upper()}'
    )

for time in elfin_times:
    elfin_scatter = dx.scatter(
        mapped_state.loc[time, 'lon'], 
        mapped_state.loc[time, 'lat'], 
        c='orange', 
        s=50,
        transform=ccrs.PlateCarree(),
        label=f'ELFIN-{elfin_id.upper()}'
        )
    dx.text(
        mapped_state.loc[time, 'lon']+2*np.cos(np.deg2rad(mapped_state.loc[time, 'lat'])), 
        mapped_state.loc[time, 'lat'],
        f'{time:%H:%M:%S}',
        transform=ccrs.PlateCarree(),
        color='orange',
        va='center',
        ha='left'
        )

themis_scatter_objects = len(locations)*[None]
for i, ((_probe, _loc), marker, color) in enumerate(zip(locations.items(), markers, colors)):
    # THEMIS
    themis_scatter_objects[i] = bx.scatter(
        _loc[0], _loc[1], color=color, marker=marker, zorder=2.01, s=50, label=f'{_probe.upper()}'
        )
    cx.scatter(_loc[0], _loc[2], color=color, marker=marker, zorder=2.01, s=50)

# https://matplotlib.org/stable/users/explain/bxes/legend_guide.html#proxy-legend-handles
elfin_legend_line = mlines.Line2D(
    [], [], 
    color='orange', 
    marker='.',                      
    markersize=15, 
    label=f'ELFIN-{elfin_id.upper()}'
    )
dx.legend(handles=themis_scatter_objects+[elfin_legend_line], loc='lower left', ncol=2)
# bx.legend()
bx.set(
    xlabel=f'$X_{{{coordinates.upper()}}} $ [$R_{{E}}$]', 
    ylabel=f'$Y_{{{coordinates.upper()}}}$ [$R_{{E}}$]', 
    xlim=x_lim, 
    ylim=y_lim
    )
bx.axis('equal')
cx.set(
    xlabel=f'$X_{{{coordinates.upper()}}} $ [$R_{{E}}$]', 
    ylabel=f'$Z_{{{coordinates.upper()}}}$ [$R_{{E}}$]', 
    xlim=x_lim, 
    ylim=z_lim
    )
cx.axhline(0, c='k', alpha=0.5)
cx.axis('equal')

for bx_i in [bx, cx]:
    earth_circle = plt.Circle((0, 0), 1, color='k', fill='grey')
    bx_i.add_patch(earth_circle)

seven_re = plt.Circle((0, 0), 7, color='k', fill=None, ls='--')
bx.add_patch(seven_re)

bx.text(0.01, 0.98, f'     T89', fontsize=15, transform=bx.transAxes, va='top')
cx.text(0.01, 0.98, f'     T89', fontsize=15, transform=cx.transAxes, va='top')

bx.xaxis.set_visible(False)
plt.suptitle(
    f'{time_range[0]:%Y-%m-%d} | Geomagnetic Indices and Observatory Locations'
    )

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