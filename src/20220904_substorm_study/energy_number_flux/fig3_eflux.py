from datetime import datetime
import string

import asilib
import asilib.map
import asilib.asi
import matplotlib.pyplot as plt
import matplotlib.dates
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import numpy as np

import elfinasi
from .map import map_elfin, map_themis


time_range = ('2022-09-04T04:17:00', '2022-09-04T04:24:00')
plot_times = (
    datetime(2022, 9, 4, 4, 19, 18),
    datetime(2022, 9, 4, 4, 20, 18),
    datetime(2022, 9, 4, 4, 21, 24),
    datetime(2022, 9, 4, 4, 22, 22)
)
location_codes = ('ATHA', 'PINA', 'GILL', 'RABB', 'LUCK')
themis_probe = 'a'
elfin_probe='a'
alt=110

elfin_labels=(
    f'Omnidirectional $e^{{-}}$ number flux',
    f'Omnidirectional $e^{{-}}$ energy flux',
    'Loss cone filling ratio',
    f'Total energy flux'
)

themis_mapped_state = {themis_probe:map_themis(themis_probe, time_range, alt)}

fig = plt.figure(figsize=(9, 9))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(5, len(plot_times),
                      left=0.03, right=0.95, bottom=0.15, top=0.97,
                      wspace=0.02, hspace=0.1,
                      height_ratios=(2, 1, 1, 1, 1))

ax = [
    asilib.map.create_map(
        lon_bounds=(-115, -88), 
        lat_bounds=(45, 63.5), 
        fig_ax=(fig, gs[0, i])
        ) for i in range(gs.ncols)
    ]
bx = fig.add_subplot(gs[1, :])
cx = fig.add_subplot(gs[2, :], sharex=bx, sharey=bx)
dx = fig.add_subplot(gs[3, :], sharex=bx)
ex = fig.add_subplot(gs[4, :], sharex=bx)

plt.suptitle(f'THEMIS-A/ELFIN-A/TREx Conjunction | {time_range[0][:10]} | T89 model | {alt} km map altitude')

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

p, _ = pad_obj_eflux.plot_omni(
    cx, labels=True, colorbar=False, vmin=1E2, vmax=1E9, pretty_plot=False
    )
_cbar = plt.colorbar(p, ax=cx, shrink=0.9, fraction=0.05, pad=0.01)
_cbar.set_label(label=pad_obj_eflux._flux_units, size=8)

p, _ = pad_obj_nflux.plot_blc_dlc_ratio(dx, labels=True, colorbar=False, vmin=1E-2, vmax=1)
_cbar = plt.colorbar(p, ax=dx, shrink=0.9, fraction=0.05, pad=0.01)
_cbar.set_label(label=f'$j_{{||}}/j_{{\perp}}$', size=10)

ex.plot(pad_obj_eflux.pad.time, relativistic_eflux, color='k', label=f'>50 keV')
ex.set(
    ylabel=f'Energy flux\n$[ergs/cm^{{2}}s]$',
    yscale='log',
    ylim=(1E-3, 5E1)
    )
ex.legend(loc='upper right')
pad_obj_nflux.plot_position(ex)
ex.xaxis.set_major_locator(plt.MaxNLocator(7))
ex.xaxis.set_label_coords(-0.07, -0.007*7)
ex.xaxis.label.set_size(9)

mapped_state = map_elfin(transformed_state, alt=alt)
# file_path = f'{mapped_state.index[0]:%Y%m%d_%H%M}_elfin{elfin_probe}_{alt}km_footprint.csv'
# mapped_state.to_csv(file_path, index_label='time')

for time, ax_i, _label in zip(plot_times, ax, string.ascii_lowercase):
    asis = asilib.Imagers(
        [asilib.asi.trex_rgb(location_code=location_code, time=time, alt=alt) 
        for location_code in location_codes]
        )
    asis.plot_map(ax=ax_i, min_elevation=10, pcolormesh_kwargs={'rasterized':True}, asi_label=False)
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

    for _other_ax in [bx, cx, dx, ex]:
        _other_ax.axvline(image_time_numeric, c='k', ls='--', alpha=1)

_z = zip([bx, cx, dx, ex], string.ascii_lowercase[len(ax):], elfin_labels)
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
cx.get_xaxis().set_visible(False)
dx.get_xaxis().set_visible(False)

for ax_i in [bx, cx, dx]:
    divider = make_axes_locatable(ax_i)
    cax = divider.append_axes("left", size="8%", pad=0.08)
    cax.remove()

# Make room on the left and right
divider = make_axes_locatable(ex)
cax = divider.append_axes("right", size="6%", pad=0.08)
cax.remove()
dax = divider.append_axes("left", size="8%", pad=0.08)
dax.remove()
plt.show()