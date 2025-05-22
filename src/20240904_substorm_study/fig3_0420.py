from datetime import datetime
import string

import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches
import matplotlib.dates
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

import elfinasi
from map import map_elfin, map_themis


time_range = ('2022-09-04T04:17:00', '2022-09-04T04:24:00')
plot_times = (
    datetime(2022, 9, 4, 4, 19, 18),
    datetime(2022, 9, 4, 4, 20, 18),
    datetime(2022, 9, 4, 4, 21, 24),
    datetime(2022, 9, 4, 4, 22, 22)
)
magnetospheric_regions = {
    'ps':(datetime(2022, 9, 4, 4, 17, 33), datetime(2022, 9, 4, 4, 20, 0), 'purple'),
    'ib':(datetime(2022, 9, 4, 4, 20, 0), datetime(2022, 9, 4, 4, 20, 25), 'purple'),
    'rb':(datetime(2022, 9, 4, 4, 20, 25), datetime(2022, 9, 4, 4, 21, 48), 'purple'),
}
location_codes = ('ATHA', 'PINA', 'GILL', 'RABB', 'LUCK')
themis_probe = 'a'
elfin_probe='a'
alt=110

elfin_labels=(
    f'Omnidirectional $e^{{-}}$ number flux',
    'Loss cone filling ratio',
)

# sst19_file_path = elfinasi.data_dir / '20220904_0418_0424_elfina_mapping_sst19.txt'
# sst19_df = pd.read_csv(sst19_file_path, delim_whitespace=True, index_col=0, parse_dates=True)
# sst19_df['IBeReEnergy'][sst19_df['IBeReEnergy']==-1000] = np.nan
# # sst19_df.index -= pd.Timedelta(seconds=40)

themis_mapped_state = {themis_probe:map_themis(themis_probe, time_range, alt)}

fig = plt.figure(figsize=(9, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
outer_gridspec = fig.add_gridspec(
    2, 
    1,
    left=0.03, 
    right=0.95, 
    bottom=0.20, 
    top=0.97,
    hspace=0.1,
    height_ratios=(1, 1)
    )
top_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
    1, 
    len(plot_times), 
    subplot_spec=outer_gridspec[0, 0],
    wspace=0.02
    )
bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
    len(elfin_labels), 
    1, 
    subplot_spec=outer_gridspec[1, 0], 
    hspace=0.05
    )

ax = [
    asilib.map.create_map(
        lon_bounds=(-115, -88), 
        lat_bounds=(45, 63.5), 
        fig_ax=(fig, top_gs[0, i]), 
        land_color='grey'
        ) for i in range(top_gs.ncols)
    ]
bx = fig.add_subplot(bottom_gs[0, :])
cx = fig.add_subplot(bottom_gs[1, :], sharex=bx, sharey=bx)

plt.suptitle(f'THEMIS-{themis_probe.upper()}/ELFIN-{elfin_probe.upper()}/TREx Conjunction | {time_range[0][:10]} | T89 model | {alt} km map altitude')

pad_obj_nflux = elfinasi.EPD_PAD(
    elfin_probe, time_range, start_pa=0, min_counts=None, accumulate=1, spin_time_tol=(2.5, 12),
    lc_exclusion_angle=0
)
pad_obj_eflux = elfinasi.EPD_PAD(
    elfin_probe, time_range, start_pa=0, min_counts=None, accumulate=1, spin_time_tol=(2.5, 12),
    lc_exclusion_angle=0, nflux=False
)
kev_erg_factor = 1.6E-9  # The conversion factor from KeV to ergs.
precipitation_solid_angle = 2*np.pi
energy_widths_mev = (pad_obj_eflux.energy_widths[:, 1]-pad_obj_eflux.energy_widths[:, 0])/1E3
eflux_ergs = kev_erg_factor*precipitation_solid_angle*(pad_obj_eflux.blc - pad_obj_eflux.ablc)*energy_widths_mev
relativistic_eflux = np.nansum(eflux_ergs, axis=1)

transformed_state = pad_obj_nflux.transform_state()
transformed_state = transformed_state.loc[time_range[0]:time_range[1]]
p, _ = pad_obj_nflux.plot_omni(
    bx, labels=True, colorbar=False, vmin=1E2, vmax=2E7, pretty_plot=False
    )
_cbar = plt.colorbar(p, ax=bx, shrink=0.9, fraction=0.05, pad=0.01)
_cbar.set_label(label=pad_obj_nflux._flux_units, size=8)

p, _ = pad_obj_nflux.plot_blc_dlc_ratio(cx, labels=True, colorbar=False, vmin=1E-2, vmax=1.5)
cx.contour(pad_obj_nflux.pad.time, pad_obj_nflux.energy, (pad_obj_nflux.blc/pad_obj_nflux.dlc).T, levels=[0.9], colors='k', linewidths=0.5)
# cx.plot(sst19_df.index, sst19_df['IBeReEnergy'], c='r', lw=5, label='SST19 IBeRe Energy')
_cbar = plt.colorbar(p, ax=cx, shrink=0.9, fraction=0.05, pad=0.01)
_cbar.set_label(label=f'$j_{{||}}/j_{{\perp}}$', size=10)

pad_obj_nflux.plot_position(cx)
cx.xaxis.set_major_locator(plt.MaxNLocator(7))
cx.xaxis.set_label_coords(-0.07, -0.007*7)
cx.xaxis.label.set_size(9)

mapped_state = map_elfin(transformed_state, alt=alt)
# file_path = f'{mapped_state.index[0]:%Y%m%d_%H%M}_elfin{elfin_probe}_{alt}km_footprint.csv'
# mapped_state.to_csv(file_path, index_label='time')

for time, ax_i, _label in zip(plot_times, ax, string.ascii_lowercase):
    asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code=location_code, time=time, alt=alt) 
        for location_code in location_codes]
        )
    vmin = min([_imager.get_color_bounds()[0] for _imager in asis.imagers])
    vmax = max([_imager.get_color_bounds()[1] for _imager in asis.imagers])
    asis.plot_map(
        ax=ax_i, 
        min_elevation=10, 
        pcolormesh_kwargs={'rasterized':True}, 
        asi_label=False, 
        color_bounds=(vmin, vmax)
        )
    ax_i.plot(mapped_state['lon'], mapped_state['lat'], c='r', transform=ccrs.PlateCarree())

    _elfin_loc = ax_i.scatter(
        mapped_state.loc[time, 'lon'], 
        mapped_state.loc[time, 'lat'], 
        c='r', 
        s=30,
        transform=ccrs.PlateCarree(),
        label=f'ELFIN-{elfin_probe.upper()}'
        )
    _themisa_index = themis_mapped_state['a'].index.get_indexer([time], method='nearest')
    _themisa_loc = ax_i.scatter(
        themis_mapped_state['a'].iloc[_themisa_index, :]['lon'], 
        themis_mapped_state['a'].iloc[_themisa_index, :]['lat'], 
        c='orange', 
        s=30,
        marker='*',
        transform=ccrs.PlateCarree(),
        label=f'THEMIS-{themis_probe.upper()}',
        zorder=2.01
        )
    if not '_legend' in locals():
        _legend = ax[0].legend(loc='lower left', ncols=2, columnspacing=0.1, handletextpad=0.1, fontsize=8)
    _plot_time = ax_i.text(
        0.01, 0.99, f'({_label}) {time:%H:%M:%S}', va='top', transform=ax_i.transAxes, fontsize=12
        )
    
# Connect the subplots and add vertical lines to bx and cx.
for ax_i, image_time_numeric in zip(ax, matplotlib.dates.date2num(plot_times)):
    line = matplotlib.patches.ConnectionPatch(
        xyA=(0.5, 0), coordsA=ax_i.transAxes,
        xyB=(image_time_numeric, bx.get_ylim()[1]), coordsB=bx.transData, 
        ls='--')
    ax_i.add_artist(line)

    for _other_ax in [bx, cx]:
        _other_ax.axvline(image_time_numeric, c='k', ls='--', alpha=1)

_z = zip([bx, cx], string.ascii_lowercase[len(ax):], elfin_labels)
for _other_ax, letter_label, elfin_label in _z:  # Just need to run this loop once
    _text = _other_ax.text(
        0.01, 
        0.99, 
        f'({letter_label}) {elfin_label}', 
        va='top', 
        transform=_other_ax.transAxes, 
        fontsize=12
        )
    _text.set_bbox(dict(facecolor='white', linewidth=0, pad=0.1, edgecolor='k'))
bx.get_xaxis().set_visible(False)

for ax_i in [bx, cx]:
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("left", size="8%", pad=0.08)
    cax.remove()

mixed_transform = transforms.blended_transform_factory(bx.transData, bx.transAxes)
for _loc, (_st, _ed, _c) in magnetospheric_regions.items():
    bx.text(
        _st+(_ed-_st)/2, 1.05, _loc.upper(), 
        color='k', transform=mixed_transform,
        ha='center', va='center',
        c='w'
        )
    rect = matplotlib.patches.Rectangle(
        (matplotlib.dates.date2num(_st), 1),  # plt.Rectangle can't handle datetime().
        (matplotlib.dates.date2num(_ed)-matplotlib.dates.date2num(_st)), 
        0.15, 
        facecolor=_c,
        edgecolor='w',
        transform=mixed_transform,
        clip_on=False,  # https://stackoverflow.com/a/62747618
        )
    bx.add_patch(rect)

plt.show()